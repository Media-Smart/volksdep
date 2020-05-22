import os

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from ..onnx import torch2onnx
from .calibrator import Calibrator
from ... import utils


__all__ = ['TRTEngine']


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, dtype, shape):
        self.host = host_mem
        self.device = device_mem
        self.dtype = dtype
        self.shape = shape

    def __str__(self):
        return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTEngine:
    def __init__(self, build_from, *args, **kwargs):
        """build tensorRT engine

        Args:
            build_from (string): build engine from specified framework, now only support torch and tensorRT engine

        """

        super(TRTEngine, self).__init__()

        self.engine = getattr(self, 'build_from_{}'.format(build_from))(*args, **kwargs)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    @staticmethod
    def build_from_torch(
            model,
            dummy_input,
            log_level='ERROR',
            max_batch_size=1,
            fp16_mode=False,
            max_workspace_size=100,
            strict_type_constraints=False,
            int8_mode=False,
            int8_calibrator=None,
    ):
        """build trt engine from pytorch model

        Args:
            model (torch.nn.Module): pytorch model
            dummy_input (torch.Tensor or np.ndarray, tuple or list): dummy input into pytorch model.
            log_level (string, default is ERROR): tensorrt logger level, now
                INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
            max_batch_size (int, default is 1): The maximum batch size which can be used at execution time, and also
                the batch size for which the engine will be optimized.
            fp16_mode (bool, default is False): Whether or not 16-bit kernels are permitted. During engine build
                fp16 kernels will also be tried when this mode is enabled.
            max_workspace_size (int, default is 100): The maximum GPU temporary memory which the ICudaEngine can use at
                execution time. MB
            strict_type_constraints (bool, default is False): When strict type constraints is set, TensorRT will choose
                the type constraints that conforms to type constraints. If the flag is not enabled higher precision
                implementation may be chosen if it results in higher performance.
            int8_mode (bool, default is False): Whether Int8 mode is used.
            int8_calibrator (vedasep.converters.Calibrator, default): calibrator for int8 mode, if None, dummy_input
                will be used as calibration data.
        """

        onnx_model = torch2onnx(model, dummy_input)

        logger = trt.Logger(getattr(trt.Logger, log_level))

        builder = trt.Builder(logger)
        builder.max_batch_size = max_batch_size
        builder.max_workspace_size = max_workspace_size * (1 << 16)
        builder.fp16_mode = fp16_mode
        builder.strict_type_constraints = strict_type_constraints

        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)
        with open(onnx_model, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        os.remove(onnx_model)

        if int8_mode:
            builder.int8_mode = True
            if int8_calibrator is None:
                int8_calibrator = Calibrator(data=utils.to(dummy_input, 'numpy'))
            builder.int8_calibrator = int8_calibrator

        engine = builder.build_cuda_engine(network)

        return engine

    @staticmethod
    def build_from_engine(name, log_level='ERROR'):
        """build trt engine from saved engine

        Args:
            name (string): engine file name to load
            log_level (string, default is ERROR): tensorrt logger level, now
                INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        """

        logger = trt.Logger(getattr(trt.Logger, log_level))

        with open(name, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        engine = self.engine
        for binding in engine:
            shape = engine.get_binding_shape(binding)
            size = trt.volume(shape) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem, dtype, shape))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem, dtype, shape))

        return inputs, outputs, bindings, stream

    def feed(self, inputs):
        for engine_inp, inp in zip(self.inputs, inputs):
            engine_inp.host = inp.astype(engine_inp.dtype)
            cuda.memcpy_htod_async(engine_inp.device, engine_inp.host, self.stream)

    def run(self, batch_size=1):
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)

    def fetch(self):
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

    def inference(self, inputs):
        """run inference with inputs

        Args:
            inputs (np.ndarray, torch.Tensor, tuple or list): inputs into trt engine.

        Returns:
            outputs (np.ndarray or list): return np.ndarray if there is only one output, else return list
        """

        inputs = utils.to(inputs, 'numpy')
        inputs = utils.flatten(inputs)
        assert len(inputs) == len(self.inputs)

        num = inputs[0].shape[0]
        max_batch_size = self.engine.max_batch_size

        outputs = []
        for current_index in range(0, num, max_batch_size):
            end_index = min(current_index+max_batch_size, num)
            batch_size = end_index - current_index
            batch_data = [inp[current_index:end_index] for inp in inputs]

            # feed batch data into GPU
            self.feed(batch_data)
            # engine context run based on batch data
            self.run(batch_size)
            # fetch result from GPU
            self.fetch()

            # Synchronize the stream
            self.stream.synchronize()

            # Return only the host outputs.
            for i, out in enumerate(self.outputs):
                valid_out = out.host.reshape(max_batch_size, *out.shape)[:batch_size]
                if current_index == 0:
                    outputs.append(valid_out)
                else:
                    outputs[i] = np.concatenate([outputs[i], valid_out], axis=0)

        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def save(self, name):
        with open(name, 'wb') as f:
            f.write(self.engine.serialize())
