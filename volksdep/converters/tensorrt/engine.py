import os
import copy
import uuid

import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda

from ..onnx import torch2onnx
from .calibrator import EntropyCalibrator2
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

    def __del__(self):
        try:
            del self.host
            del self.device
            del self.dtype
            del self.shape
        except:
            pass


class TRTEngine:
    def __init__(self, build_from, *args, **kwargs):
        """build tensorRT engine

        Args:
            build_from (string): build engine from specified framework, now support onnx, torch and tensorRT engine
            args, kwargs are parameters in build_from_function
        """

        super(TRTEngine, self).__init__()

        torch.cuda.init()

        self.engine = getattr(self, 'build_from_{}'.format(build_from))(*args, **kwargs)
        if self.engine.has_implicit_batch_dimension:
            raise AttributeError('Model has implicit batch dimension, now only support explicit batch dimension.')
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    @staticmethod
    def build_from_onnx(
            model,
            log_level='ERROR',
            max_workspace_size=100,
            fp16_mode=False,
            strict_type_constraints=False,
            int8_mode=False,
            int8_calibrator=None,
    ):
        """build trt engine from onnx model

        Args:
            model (string): onnx model name
            log_level (string, default is ERROR): tensorrt logger level, now
                INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
            max_workspace_size (int, default is 100): The maximum GPU temporary memory which the ICudaEngine can use at
                execution time. MB
            fp16_mode (bool, default is False): Whether or not 16-bit kernels are permitted. During engine build
                fp16 kernels will also be tried when this mode is enabled.
            strict_type_constraints (bool, default is False): When strict type constraints is set, TensorRT will choose
                the type constraints that conforms to type constraints. If the flag is not enabled higher precision
                implementation may be chosen if it results in higher performance.
            int8_mode (bool, default is False): Whether Int8 mode is used.
            int8_calibrator (vedasep.converters.calibrator.BaseCalibrator, default is None): calibrator for int8 mode,
                if None, default calibrator will be used as calibration data.
        """

        logger = trt.Logger(getattr(trt.Logger, log_level))

        builder = trt.Builder(logger)
        builder.max_workspace_size = max_workspace_size * (1 << 16)
        builder.fp16_mode = fp16_mode
        builder.strict_type_constraints = strict_type_constraints

        network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(model, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        # re-order output tensor
        output_tensors = [network.get_output(i) for i in range(network.num_outputs)]
        [network.unmark_output(tensor) for tensor in output_tensors]
        for tensor in output_tensors:
            identity_out_tensor = network.add_identity(tensor).get_output(0)
            identity_out_tensor.name = 'identity_{}'.format(tensor.name)
            network.mark_output(tensor=identity_out_tensor)

        max_batch_size = max([network.get_input(i).shape[0] for i in range(network.num_inputs)])
        builder.max_batch_size = max_batch_size

        if int8_mode:
            builder.int8_mode = True
            if int8_calibrator is None:
                input_shapes = [network.get_input(i).shape for i in range(network.num_inputs)]
                int8_calibrator = EntropyCalibrator2(data=utils.gen_ones_data(input_shapes))
            builder.int8_calibrator = int8_calibrator

        engine = builder.build_cuda_engine(network)

        return engine

    @staticmethod
    def build_from_torch(
            model,
            dummy_input,
            log_level='ERROR',
            max_workspace_size=100,
            fp16_mode=False,
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
            max_workspace_size (int, default is 100): The maximum GPU temporary memory which the ICudaEngine can use at
                execution time. MB
            fp16_mode (bool, default is False): Whether or not 16-bit kernels are permitted. During engine build
                fp16 kernels will also be tried when this mode is enabled.
            strict_type_constraints (bool, default is False): When strict type constraints is set, TensorRT will choose
                the type constraints that conforms to type constraints. If the flag is not enabled higher precision
                implementation may be chosen if it results in higher performance.
            int8_mode (bool, default is False): Whether Int8 mode is used.
            int8_calibrator (vedasep.converters.calibrator.BaseCalibrator, default is None): calibrator for int8 mode,
                if None, default calibrator will be used as calibration data.
        """

        onnx_model = f'/tmp/{uuid.uuid4()}.onnx'

        try:
            torch2onnx(model, dummy_input, onnx_model)

            if int8_mode and int8_calibrator is None:
                int8_calibrator = EntropyCalibrator2(data=utils.to(dummy_input, 'numpy'))

            engine = TRTEngine.build_from_onnx(onnx_model, log_level, max_workspace_size, fp16_mode, strict_type_constraints, int8_mode, int8_calibrator)
        except Exception as e:
            raise e
        finally:
            if os.path.exists(onnx_model):
                os.remove(onnx_model)

        return engine

    @staticmethod
    def build_from_engine(model, log_level='ERROR'):
        """build trt engine from saved engine

        Args:
            model (string): engine file name to load
            log_level (string, default is ERROR): tensorrt logger level, now
                INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        """

        logger = trt.Logger(getattr(trt.Logger, log_level))

        with open(model, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(binding)[1:]
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem, dtype, shape))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem, dtype, shape))

        return inputs, outputs, bindings, stream

    def feed(self, inputs):
        for engine_inp, inp in zip(self.inputs, inputs):
            engine_inp.host = np.ascontiguousarray(inp.astype(engine_inp.dtype))
            cuda.memcpy_htod_async(engine_inp.device, engine_inp.host, self.stream)

    def run(self):
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

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
            self.run()
            # fetch result from GPU
            self.fetch()
            # Synchronize the stream
            self.stream.synchronize()

            # Return only the host outputs.
            for i, out in enumerate(self.outputs):
                valid_out = copy.deepcopy(out.host.reshape(max_batch_size, *out.shape)[:batch_size])
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

    def __del__(self):
        """Free Cuda Memory"""
        try:
            del self.engine
            del self.context
            del self.inputs
            del self.outputs
            del self.bindings
            del self.stream
        except:
            pass
