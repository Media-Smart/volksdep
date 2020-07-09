import warnings
warnings.filterwarnings("ignore")

import torch

from .. import utils


def torch2onnx(
        model,
        dummy_input,
        onnx_model_name,
        opset_version=9,
        do_constant_folding=False,
        verbose=False
):
    """convert pytorch model to onnx

    Args:
        model (torch.nn.Module): pytorch model
        dummy_input (torch.Tensor, tuple or list): dummy input into pytorch
            model.
        onnx_model_name (string or io object): saved onnx model name.
        opset_version (int, default is 9): onnx opset version.
        do_constant_folding (bool, default False): If True, the
            constant-folding optimization is applied to the model during
            export. Constant-folding optimization will replace some of the ops
            that have all constant inputs, with pre-computed constant nodes.
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.

    Returns:
        onnx_model_name (string): saved onnx model name
    """

    if isinstance(dummy_input, tuple):
        dummy_input = list(dummy_input)
    dummy_input = utils.to(dummy_input, 'cuda')
    model.cuda().eval()
    with torch.no_grad():
        output = model(dummy_input)

    assert not isinstance(dummy_input, dict), 'input type should not be dict.'
    assert not isinstance(output, dict), 'output type should not be dict'

    input_names = utils.get_names(dummy_input, 'input')
    output_names = utils.get_names(output, 'output')
    dynamic_axes = {name: [0] for name in input_names + output_names}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
        dynamic_axes=dynamic_axes,
    )

    torch.cuda.empty_cache()
