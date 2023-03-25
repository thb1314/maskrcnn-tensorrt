import tensorrt as trt
import numpy as np
import os
import ctypes
# cuda: https://nvidia.github.io/cuda-python/
# pycuda: https://documen.tician.de/pycuda/
import pycuda.driver as cuda
import torch
import pycuda.autoinit
import glob
import mmcv


# 设置一些常量
epsilon = 1.0e-2
np.random.seed(97)
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
so_files = glob.glob(os.path.join('../relaventTensorRTPlugin/build', '*.so'))

for so_file in so_files:
    ctypes.cdll.LoadLibrary(so_file)
    print('load {} success!'.format(os.path.basename(so_file)))

def GiB(val):
    return val * 1 << 30

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    
    def free(self):
        self.host = None
        if self.device is not None:
            self.device.free()
            self.device = None
    
    def __del__(self):
        self.free()
    
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(ori_inputs, ori_outputs, engine, context, stream):
    inputs = []
    outputs = []
    bindings = []
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    
    for i, binding in enumerate(engine):
        size = trt.volume(context.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        try:
            if engine.binding_is_input(binding):
                ori_mem = ori_inputs[i]
            else:
                ori_mem = ori_outputs[i - nInput]
        except:
            ori_mem = None
            
        if ori_mem is not None:
            if ori_mem.host.nbytes >= size:
                host_mem = ori_mem.host
                device_mem = ori_mem.device
                # 避免再次释放
                ori_mem.device = None
            else:
                ori_mem.free()
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
        else:
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings


def build_engine(onnx_file_path, enable_fp16=False, max_batch_size=1, max_workspace_size=10, write_engine=True):
    # 通过加载onnx文件，构建engine
    # :param onnx_file_path: onnx文件路径
    # :return: engine
    onnx_path = os.path.realpath(onnx_file_path) 
    engine_file_path = ".".join(onnx_path.split('.')[:-1] + ['engine'])
    print('engine_file_path', engine_file_path)
    G_LOGGER = trt.Logger(trt.Logger.INFO)
    if os.path.exists(engine_file_path):
        with open(engine_file_path, 'rb') as f, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine, engine_file_path
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(max_workspace_size))
        if enable_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path {} ...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                # print(" parsing error:", parser.get_error(0).code(), "\n",
                #       "function name:", parser.get_error(0).func(), "\n",
                #       "node:", parser.get_error(0).node(), "\n",
                #       "line num:", parser.get_error(0).line(), "\n",
                #       "desc:", parser.get_error(0).desc())
                return None, None
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        # 重点
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 800, 1216), (max_batch_size, 3, 800, 1216), (max_batch_size, 3, 800, 1216))
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            return None, None
        print("Completed creating Engine")
        # 保存engine文件
        if write_engine:
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)
        with trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine, engine_file_path


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class TRTMaskRCNN(object):
    def __init__(self, engine_or_onnx_path):
        self.engine_path = engine_or_onnx_path
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._get_engine()
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = None
        self.outputs = None


    def _get_engine(self):
        # If a serialized engine exists, use it instead of building an engine.
        return build_engine(self.engine_path, enable_fp16=False, max_batch_size=1, write_engine=True)[0]

    def detect(self, image_np_array, cuda_ctx = pycuda.autoinit.context):
        if cuda_ctx:
            cuda_ctx.push()

        batch_size = image_np_array.shape[0]
        # 动态输入
        origin_inputshape = self.context.get_binding_shape(0)
        origin_inputshape[0] = batch_size
        self.context.set_binding_shape(0, (origin_inputshape))
        self.context.set_optimization_profile_async(0, self.stream.handle)
        
        self.inputs, self.outputs, bindings = allocate_buffers(self.inputs, self.outputs, self.engine, self.context, self.stream)
        np_type = trt.nptype(self.engine.get_binding_dtype(0))
        # Do inference
        self.inputs[0].host = np.ascontiguousarray(image_np_array.astype(np_type))
        trt_outputs = do_inference(self.context, bindings=bindings, inputs=self.inputs, outputs=self.outputs,
                                          stream=self.stream)
        
        if cuda_ctx:
            cuda_ctx.pop()
        
        nInput = np.sum([self.engine.binding_is_input(i) for i in range(self.engine.num_bindings)])
        nOutput = self.engine.num_bindings - nInput
        trt_outputs_dict = dict()
        
        for i in range(nOutput):
            shape = self.context.get_binding_shape(nInput + i)
            name = self.engine.get_binding_name(nInput + i)
            trt_outputs_dict[name] = trt_outputs[i].reshape(shape)
        return trt_outputs_dict
    
    def __call__(self, x):
        return self.detect(x)
    
    def __del__(self):
        del self.inputs
        del self.outputs
        del self.stream
        del self.engine
        del self.context


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

def main():
    import mmdet

    trt_maskrcnn = TRTMaskRCNN('../results/mask_rcnn_r50_fpn_2x_coco.onnx')


    with open('onnx_output_dict.pkl', 'rb') as f:
        onnx_outputs = torch.load(f, map_location='cpu')
    
    
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
        return (width, height)

    image_array = imgpath2pad_array(jpg_image_path)
    # img_metas = [
    #     {
    #         'img_shape':tuple(image_array.shape),
    #         'ori_shape':get_image_size(jpg_image_path)
    #     },
    # ]
    tensor = array2tensor(image_array)
    tensor = np.ascontiguousarray(np.expand_dims(tensor, axis=0))

    outputs = trt_maskrcnn.detect(tensor)
    
    for k in outputs:
        diff = (outputs[k] - onnx_outputs[k])
        print('key: {}, diff: {}'.format(k, np.abs(diff).max()))


if __name__ == "__main__":
    main()

