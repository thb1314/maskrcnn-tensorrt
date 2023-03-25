from mmdet.apis import init_detector, inference_detector
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

model = init_detector(config_file, checkpoint_file, device='cpu')
jpg_image_path = path_join(MMDET_DIR,  'demo', 'demo.jpg')
det_result = inference_detector(model, jpg_image_path)

basename_woext, ext = os.path.splitext(os.path.basename(jpg_image_path))
result_pred_path = path_join('../results', basename_woext + '_result' + ext)
os.makedirs(dirpath(result_pred_path), exist_ok=True)

model.show_result(jpg_image_path, det_result, out_file=result_pred_path)
