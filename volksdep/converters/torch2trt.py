import io

from .torch2onnx import torch2onnx
from .onnx2trt import onnx2trt


def torch2trt(
        model,
        dummy_input,
        log_level='ERROR',
        max_batch_size=1,
        max_workspace_size=1,
        fp16_mode=False,
        strict_type_constraints=False,
        int8_mode=False,
        int8_calibrator=None,
):
    """build TensorRT model from pytorch model.

    Args:
        model (torch.nn.Module): pytorch model
        dummy_input (torch.Tensor, tuple or list): dummy input of pytorch model.
        log_level (string, default is ERROR): tensorrt logger level, now
            INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        max_batch_size (int, default=1): The maximum batch size which can be used at execution time, and also the
            batch size for which the ICudaEngine will be optimized.
        max_workspace_size (int, default is 1): The maximum GPU temporary memory which the ICudaEngine can use at
            execution time. default is 1GB.
        fp16_mode (bool, default is False): Whether or not 16-bit kernels are permitted. During engine build
            fp16 kernels will also be tried when this mode is enabled.
        strict_type_constraints (bool, default is False): When strict type constraints is set, TensorRT will choose
            the type constraints that conforms to type constraints. If the flag is not enabled higher precision
            implementation may be chosen if it results in higher performance.
        int8_mode (bool, default is False): Whether Int8 mode is used.
        int8_calibrator (volksdep.calibrators.base.BaseCalibrator, default is None): calibrator for int8 mode,
            if None, default calibrator will be used as calibration data.
    """

    f = io.BytesIO()
    torch2onnx(model, dummy_input, f)
    f.seek(0)

    trt_model = onnx2trt(f, log_level, max_batch_size, max_workspace_size, fp16_mode, strict_type_constraints,
                         int8_mode, int8_calibrator)

    return trt_model
