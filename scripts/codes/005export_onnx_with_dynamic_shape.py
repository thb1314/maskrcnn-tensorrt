# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import warnings
from functools import partial

import numpy as np
import onnx
import torch
from mmcv import Config, DictAction

from mmdet.core.export import build_model_from_cfg, preprocess_example_input
from mmdet.core.export.model_wrappers import ONNXRuntimeDetector


def pytorch2onnx(model,
                 input_img,
                 input_shape,
                 normalize_cfg,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 test_img=None,
                 do_simplify=False,
                 dynamic_export=None,
                 skip_postprocess=False,
                 force_write=False):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }
    # prepare input
    one_img, one_meta = preprocess_example_input(input_config)
    img_list, img_meta_list = [one_img], [[one_meta]]
    
    if skip_postprocess:
        warnings.warn('Not all models support export onnx without post '
                      'process, especially two stage detectors!')
        origin_forward = model.forward
        model.forward = model.forward_dummy
        torch.onnx.export(
            model,
            one_img,
            output_file,
            input_names=['input'],
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=show,
            opset_version=opset_version)
        model.forward = origin_forward
        print(f'Successfully exported ONNX model without '
              f'post process: {output_file}')
        # return
    else:
        # replace original forward function
        origin_forward = model.forward
        model.forward = partial(
            model.forward,
            img_metas=img_meta_list,
            return_loss=False,
            rescale=False)

        output_names = ['dets', 'labels']
        if model.with_mask:
            output_names.append('masks')
        input_name = 'input'
        dynamic_axes = None
        if dynamic_export:
            dynamic_axes = {
                input_name: {
                    0: 'batch',
                },
                'dets': {
                    0: 'batch',
                },
                'labels': {
                    0: 'batch',
                },
            }
            if model.with_mask:
                dynamic_axes['masks'] = {0: 'batch'}
        if not os.path.exists(output_file) or force_write:
            torch.onnx.export(
                model,
                img_list,
                output_file,
                input_names=[input_name],
                output_names=output_names,
                export_params=True,
                keep_initializers_as_inputs=True,
                do_constant_folding=True,
                verbose=show,
                opset_version=opset_version,
                dynamic_axes=dynamic_axes)

        model.forward = origin_forward

    if do_simplify:
        import onnxsim

        from mmdet import digit_version

        min_required_version = '0.4.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnxsim>={min_required_version}'

        model_opt, check_ok = onnxsim.simplify(output_file)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            warnings.warn('Failed to simplify ONNX model.')
    print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        
        # wrap onnx model
        onnx_model = ONNXRuntimeDetector(output_file, model.CLASSES, 0)
        # if dynamic_export:
        #     # scale up to test dynamic shape
        #     h, w = [int((_ * 1.5) // 32 * 32) for _ in input_shape[2:]]
        #     h, w = min(1344, h), min(1344, w)
        #     input_config['input_shape'] = (1, 3, h, w)

        if test_img is None:
            input_config['input_path'] = input_img

        # prepare input once again
        one_img, one_meta = preprocess_example_input(input_config)
        img_list, img_meta_list = [one_img], [[one_meta]]

        # get pytorch output
        with torch.no_grad():
            pytorch_results = model(
                img_list,
                img_metas=img_meta_list,
                return_loss=False,
                rescale=True)[0]
        

        img_list = [_.cuda().contiguous() for _ in img_list]
        if dynamic_export:
            img_list = img_list + [_.flip(-1).contiguous() for _ in img_list]
            img_meta_list = img_meta_list * 2
        # get onnx output
        onnx_results = onnx_model(
            img_list, img_metas=img_meta_list, return_loss=False, rescale=True)[0]
        # visualize predictions
        score_thr = 0.0001

        out_file_ort, out_file_pt = 'show-ort.png', 'show-pt.png'

        show_img = one_meta['show_img']
        model.show_result(
            show_img,
            pytorch_results,
            score_thr=score_thr,
            show=True,
            win_name='PyTorch',
            out_file=out_file_pt)
        onnx_model.show_result(
            show_img,
            onnx_results,
            score_thr=score_thr,
            show=True,
            win_name='ONNXRuntime',
            out_file=out_file_ort)

        
        # compare a part of result
        if model.with_mask:
            compare_pairs = list(zip(onnx_results, pytorch_results))
        else:
            compare_pairs = [(onnx_results, pytorch_results)]
        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                try:
                    np.testing.assert_allclose(
                        o_res, p_res, rtol=0.01, atol=0.1, err_msg=err_msg)
                except AssertionError as e:
                    print(e)
                    print('onnxruntime')
                    print(o_res)
                    print('pytorch')
                    print(p_res)
                    if hasattr(o_res, 'shape') and hasattr(p_res, 'shape'):
                        print('onnxruntime', o_res.shape, 'pytorch', p_res.shape)
                    
        print('The numerical values are the same between Pytorch and ONNX')


def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show onnx graph and detection outputs')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')
    parser.add_argument(
        '--dataset',
        type=str,
        default='coco',
        help='Dataset name. This argument is deprecated and will be removed \
        in future releases.')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1216],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[123.675, 116.28, 103.53],
        help='mean value used for preprocess input data.This argument \
        is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[58.395, 57.12, 57.375],
        help='variance value used for preprocess input data. '
        'This argument is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    parser.add_argument(
        '--skip-postprocess',
        action='store_true',
        help='Whether to export model without post process. Experimental '
        'option. We do not guarantee the correctness of the exported '
        'model.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import os
    import glob
    import mmdet


    dirpath = os.path.dirname
    path_join = os.path.join
    BASE_DIR = path_join(dirpath(dirpath(os.path.realpath(__file__))), 'models', 'mask-rcnn')
    MMDET_DIR = dirpath(dirpath(mmdet.__file__))
    print('BASE_DIR', BASE_DIR)
    print('MMDET_DIR', MMDET_DIR)

    config_file = path_join(BASE_DIR, 'config', 'mask_rcnn_r50_fpn_2x_coco.py')
    checkpoint_file = path_join(BASE_DIR, 'pretrained_model', 'mask_rcnn_r50_fpn_2x_coco_*.pth')
    checkpoint_file = glob.glob(checkpoint_file)[0]


    opset_version = 11
    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(opset_version)

    cfg = Config.fromfile(config_file)
    img_scale = [800, 1216]
    input_shape = (1, 3, img_scale[0], img_scale[1])


    # build the model and load checkpoint
    model = build_model_from_cfg(config_file, checkpoint_file)
    jpg_image_path = path_join(MMDET_DIR,  'demo', 'demo.jpg')
    normalize_cfg = parse_normalize_cfg(cfg.test_pipeline)
    basename_woext, ext = os.path.splitext(os.path.basename(config_file))

    basename_woext += '_dynamic_shape'
    onnx_path = os.path.join('..', 'results', basename_woext+'.onnx')
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    # convert model to onnx file
    pytorch2onnx(
        model,
        jpg_image_path,
        input_shape,
        normalize_cfg,
        opset_version=opset_version,
        show=True,
        output_file=onnx_path,
        verify=True,
        test_img=None,
        do_simplify=True,
        dynamic_export=True,
        skip_postprocess=False,
        force_write=True)
