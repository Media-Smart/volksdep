import uuid
import copy
import warnings
warnings.filterwarnings("ignore")

import torch

from ... import utils


__all__ = ['torch2onnx']


def torch2onnx(
        model,
        dummy_input,
        onnx_model_name=None,
        opset_version=9,
        do_constant_folding=False,
        verbose=False
):
    """convert pytorch model to onnx

    Args:
        model (torch.nn.Module): pytorch model
        dummy_input (torch.Tensor or np.ndarray, tuple or list): dummy input into pytorch model.
        onnx_model_name (string, default is None): saved onnx model name, if None, onnx model will be saved in /tmp/
            with a unique name.
        opset_version (int, default is 9): by default we export the model to the opset version of the onnx submodule.
            Since ONNX’s latest opset may evolve before next stable release, by default we export to one stable opset
            version. Right now, supported stable opset version is 9
        do_constant_folding (bool, default False): If True, the constant-folding optimization is applied to the model
            during export. Constant-folding optimization will replace some of the ops that have all constant inputs,
            with pre-computed constant nodes.
        verbose (bool, default False): if specified, we will print out a debug description of the trace being exported.

    Returns:
        onnx_model_name (string): saved onnx model name
    """

    dummy_input = copy.deepcopy(dummy_input)
    dummy_input = utils.to(dummy_input, 'torch')
    dummy_input = utils.to(dummy_input, 'cuda')
    dummy_input = utils.to(dummy_input, torch.float32)

    model = copy.deepcopy(model).cuda().to(torch.float32).eval()

    output = model(dummy_input)

    input_names = utils.get_names(dummy_input, 'input')
    output_names = utils.get_names(output, 'output')
    dynamic_axes = {name: {0: 'batch'} for name in input_names+output_names}

    if onnx_model_name is None:
        onnx_model_name = '/tmp/{}.onnx'.format(uuid.uuid4())

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
    )

    output = utils.to(output, 'cpu')
    model = model.cpu()
    dummy_input = utils.to(dummy_input, 'cpu')

    del output
    del model
    del dummy_input
    torch.cuda.empty_cache()

    return onnx_model_name
