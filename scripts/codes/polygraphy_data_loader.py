import numpy as np
import mmcv
import os

"""
Defines a `load_data` function that returns a generator yielding
feed_dicts so that this script can be used as the argument for
the --data-loader-script command-line parameter.
"""

INPUT_SHAPE = (1, 2, 28, 28)

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


def load_data():
    import mmdet
    dirpath = os.path.dirname
    path_join = os.path.join
    MMDET_DIR = dirpath(dirpath(mmdet.__file__))
    jpg_image_path = path_join(MMDET_DIR,  'demo', 'demo.jpg')

    image_array = imgpath2pad_array(jpg_image_path)

    tensor = array2tensor(image_array)
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
    # Still totally real data
    yield {"input": tensor}