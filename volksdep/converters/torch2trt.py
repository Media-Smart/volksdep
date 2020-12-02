import io

from .torch2onnx import torch2onnx
from .onnx2trt import onnx2trt


def torch2trt(
        model,
        dummy_input,
        log_level='ERROR',
        max_batch_size=1,
        min_input_shapes=None,
        max_input_shapes=None,
        max_workspace_size=1,
        fp16_mode=False,
        strict_type_constraints=False,
        int8_mode=False,
        int8_calibrator=None,
        opset_version=9,
        do_constant_folding=False,
        verbose=False):

    """build TensorRT model from PyTorch model.

    Args:
        model (torch.nn.Module): PyTorch model.
        dummy_input (torch.Tensor, tuple or list): dummy input.
        log_level (string, default is ERROR): TensorRT logger level,
            INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        max_batch_size (int, default=1): The maximum batch size which can be
            used at execution time, and also the batch size for which the
            ICudaEngine will be optimized.
        min_input_shapes (list, default is None): Minimum input shapes, should
            be provided when shape is dynamic. For example, [(3, 224, 224)] is
            for only one input.
        max_input_shapes (list, default is None): Maximum input shapes, should
            be provided when shape is dynamic. For example, [(3, 224, 224)] is
            for only one input.
        max_workspace_size (int, default is 1): The maximum GPU temporary
            memory which the ICudaEngine can use at execution time. default is
            1GB.
        fp16_mode (bool, default is False): Whether or not 16-bit kernels are
            permitted. During engine build fp16 kernels will also be tried when
            this mode is enabled.
        strict_type_constraints (bool, default is False): When strict type
            constraints is set, TensorRT will choose the type constraints that
            conforms to type constraints. If the flag is not enabled higher
            precision implementation may be chosen if it results in higher
            performance.
        int8_mode (bool, default is False): Whether Int8 mode is used.
        int8_calibrator (volksdep.calibrators.base.BaseCalibrator, default is
            None): calibrator for int8 mode, if None, default calibrator will
            be used as calibration data.
        opset_version (int, default is 9): Onnx opset version.
        do_constant_folding (bool, default False): If True, the
            constant-folding optimization is applied to the model during
            export. Constant-folding optimization will replace some of the ops
            that have all constant inputs, with pre-computed constant nodes.
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
    """

    assert not (bool(min_input_shapes) ^ bool(max_input_shapes))

    f = io.BytesIO()
    dynamic_shape = bool(min_input_shapes) and bool(max_input_shapes)
    torch2onnx(model, dummy_input, f, dynamic_shape, opset_version,
               do_constant_folding, verbose)
    f.seek(0)

    trt_model = onnx2trt(f, log_level, max_batch_size, min_input_shapes,
                         max_input_shapes, max_workspace_size, fp16_mode,
                         strict_type_constraints, int8_mode, int8_calibrator)

    return trt_model
