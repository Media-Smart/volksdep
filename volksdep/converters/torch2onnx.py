import warnings
warnings.filterwarnings("ignore")

import torch

from .. import utils


def torch2onnx(
        model,
        dummy_input,
        onnx_model_name,
        dynamic_shape=False,
        opset_version=9,
        do_constant_folding=False,
        verbose=False):

    """convert PyTorch model to Onnx

    Args:
        model (torch.nn.Module): PyTorch model.
        dummy_input (torch.Tensor, tuple or list): dummy input.
        onnx_model_name (string or io object): saved Onnx model name.
        dynamic_shape (bool, default is False): if False, only first dimension
            will be dynamic; if True, all dimensions will be dynamic.
        opset_version (int, default is 9): Onnx opset version.
        do_constant_folding (bool, default False): If True, the
            constant-folding optimization is applied to the model during
            export. Constant-folding optimization will replace some of the ops
            that have all constant inputs, with pre-computed constant nodes.
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
    """

    if isinstance(dummy_input, tuple):
        dummy_input = list(dummy_input)
    dummy_input = utils.to(dummy_input, 'cuda')
    model.eval().cuda()
    with torch.no_grad():
        output = model(dummy_input)

    assert not isinstance(dummy_input, dict), 'input should not be dict.'
    assert not isinstance(output, dict), 'output should not be dict'

    input_names = utils.get_names(dummy_input, 'input')
    output_names = utils.get_names(output, 'output')

    dynamic_axes = dict()
    for name, tensor in zip(input_names+output_names,
                            utils.flatten(dummy_input)+utils.flatten(output)):
        dynamic_axes[name] = list(range(tensor.dim())) if dynamic_shape else [0]

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
        dynamic_axes=dynamic_axes)

    torch.cuda.empty_cache()
