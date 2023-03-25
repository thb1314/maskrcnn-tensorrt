# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets.pipelines import Compose
from mmdet.core import bbox2result
from mmdet.datasets import replace_ImageToTensor
import onnxruntime as rt
import warnings
import os
import torch
# from mmdet.datasets.builder import bui


def imgpath2pad_array(image_path):
    
    image = mmcv.imread(image_path, channel_order='rgb')
    height, width = image.shape[:2]
    dst_height, dst_width = 800, 1216
    resize_scale = min(dst_height / height, dst_width / width)
    new_height, new_width = round(height * resize_scale), round(width * resize_scale)
    resize_image = mmcv.imresize(image, (new_width, new_height))
    # pad_height = dst_height - new_height
    # pad_width = dst_width - new_width
    # ((before_1, after_1), ... (before_N, after_N))
    # pad_val = ((0, pad_height), (0, pad_width), ) + tuple([(0, 0), ] * (len(image.shape) - 2))
    return resize_image # np.pad(resize_image, pad_width = pad_val, mode='constant')


# pad numpy array and transpose it to tensor memory map
def array2tensor(array):
    dst_height, dst_width = 800, 1216
    height, width = array.shape[:2]
    pad_height = dst_height - height
    pad_width = dst_width - width
    pad_val = ((0, pad_height), (0, pad_width), ) + tuple([(0, 0), ] * (len(array.shape) - 2))
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32).reshape(1,1,3)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32).reshape(1,1,3)
    array = (array.astype(np.float32) - mean) / std
    pad_array = np.pad(array, pad_width = pad_val, mode='constant')
    return np.transpose(pad_array, axes=[2, 0, 1])



def get_onnx_runner(onnx_filepath):
    # 注册自定义算子
    # get the custom op path
    ort_custom_op_path = ''
    try:
        from mmcv.ops import get_onnxruntime_op_path
        ort_custom_op_path = get_onnxruntime_op_path()
    except (ImportError, ModuleNotFoundError):
        warnings.warn('If input model has custom op from mmcv, \
            you may have to build mmcv with ONNXRuntime from source.')
    session_options = rt.SessionOptions()
    # register custom op for onnxruntime
    if os.path.exists(ort_custom_op_path):
        session_options.register_custom_ops_library(ort_custom_op_path)

    sess = rt.InferenceSession(onnx_filepath, session_options)
    input_names = [item.name for item in sess.get_inputs()]
    label_names = [item.name for item in sess.get_outputs()]
    def runner(input_tensors):
        nonlocal label_names
        nonlocal input_names
        pred_onnx = sess.run(label_names, dict(zip(input_names, input_tensors)))
        return dict(zip(label_names,pred_onnx))
    return runner

if __name__ == "__main__":
    import os
    import mmdet

    dirpath = os.path.dirname
    path_join = os.path.join
    BASE_DIR = path_join(dirpath(dirpath(os.path.realpath(__file__))), 'models', 'mask-rcnn')
    MMDET_DIR = dirpath(dirpath(mmdet.__file__))
    print('BASE_DIR', BASE_DIR)
    print('MMDET_DIR', MMDET_DIR)

    cur_dir = os.path.dirname(__file__)
    jpg_image_path = path_join(MMDET_DIR,  'demo', 'demo.jpg')
    
    from PIL import Image
    def get_image_size(filepath):
        im = Image.open(filepath)
        width, height = im.size
        return (height, width)

    image_array = imgpath2pad_array(jpg_image_path)
    dst_path = os.path.join(cur_dir, '..',  'results', "image_array.jpg")
    os.makedirs(dirpath(dst_path), exist_ok=True)

    mmcv.imwrite(image_array[..., ::-1], dst_path)
    img_metas = [
        {
            'img_shape':tuple(image_array.shape),
            'ori_shape':get_image_size(jpg_image_path)
        },
    ]
    # print(image_array.shape)
    tensor = array2tensor(image_array)
    tensor = np.expand_dims(tensor, axis=0)
    # print(tensor.shape)
    onnx_filepath = '../results/mask_rcnn_r50_fpn_2x_coco.onnx'
    runner = get_onnx_runner(onnx_filepath)
    output_dict = runner([tensor])
    torch.save(output_dict, 'onnx_output_dict.pkl')
    
    import pickle
    with open('./class_names.pickle', 'rb') as f:
        class_names = pickle.load(f)
    score_thr = 0.3

    # print('class_names', class_names)
    for key, value in output_dict.items():
        print(key, value.shape)

    batch_dets = output_dict['dets']
    batch_labels = output_dict['labels']
    batch_masks = output_dict['masks']
    batch_size = tensor.shape[0]

    results = []
    for i in range(batch_size):
        # [N, 5], [N]
        dets, labels = batch_dets[i], batch_labels[i]
        det_mask = dets[:, -1] >= score_thr
        dets = dets[det_mask]
        labels = labels[det_mask]
        
        # [N, 800, 1216]
        masks = batch_masks[i]
        masks = masks[det_mask]
        img_h, img_w = img_metas[i]['img_shape'][:2]
        ori_h, ori_w = img_metas[i]['ori_shape'][:2]

        scale_factor_w = img_w / ori_w
        scale_factor_h = img_h / ori_h
        scale_factor = np.array([scale_factor_w, scale_factor_h, scale_factor_w, scale_factor_h])
        dets[:, :4] /= scale_factor

        dets_results = bbox2result(dets, labels, len(class_names))
        # 去除 padding
        masks = masks[:, :img_h, :img_w]
        if True:
            masks = masks.astype(np.float32)
            masks = torch.from_numpy(masks)
            masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=(ori_h, ori_w))
            masks = masks.squeeze(0).detach().numpy()

        if masks.dtype != bool:
            masks = masks >= 0.5
        segms_results = [[] for _ in range(len(class_names))]
        for j in range(len(dets)):
            segms_results[labels[j]].append(masks[j])
        results.append((dets_results, segms_results))


    # 绘制推理结果
    # wrap onnx model
    from mmdet.core.export.model_wrappers import ONNXRuntimeDetector
    onnx_model = ONNXRuntimeDetector(onnx_filepath, class_names, 0)
    
    onnx_model.show_result(
        jpg_image_path,
        results[0],
        score_thr=score_thr,
        show=True,
        win_name='ONNXRuntime',
        out_file='../results/onnxruntime_result.png')
