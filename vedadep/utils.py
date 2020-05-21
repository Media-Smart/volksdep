import torch
import numpy as np


def get_names(inp, prefix):
    if not isinstance(inp, (tuple, list)):
        inp = [inp]

    names = []

    for i in range(len(inp)):
        x = inp[i]
        sub_prefix = '{}.{}'.format(prefix, i)
        if isinstance(x, torch.Tensor):
            names.append(sub_prefix)
        elif isinstance(x, (list, tuple)):
            names += get_names(x, sub_prefix)
        else:
            raise TypeError('Input/Output only support variables, tuple ans list')

    return names


def get_form(inp):
    if not isinstance(inp, (tuple, list)):
        return 'x'

    data_format = []
    for x in inp:
        data_format.append(get_form(x))

    return data_format


def to(inp, device_or_dtype):
    if not isinstance(inp, (tuple, list)):
        if isinstance(inp, (torch.Tensor, torch.nn.Module)):
            if device_or_dtype == 'numpy':
                inp = inp.cpu().numpy()
            elif device_or_dtype == 'torch':
                inp = inp
            else:
                inp = inp.to(device_or_dtype)
        elif isinstance(inp, np.ndarray):
            if device_or_dtype == 'numpy':
                inp = inp
            elif device_or_dtype == 'torch':
                inp = torch.from_numpy(inp)
            else:
                inp = inp.astype(device_or_dtype)
        else:
            raise TypeError('Unsupported type {}'.format(type(inp)))

        return inp
    else:
        out = []
        for x in inp:
            out.append(to(x, device_or_dtype))
        out = type(inp)(out)

        return out


def flatten(inp):
    if not isinstance(inp, (tuple, list)):
        return [inp]
    else:
        out = []
        for x in inp:
            out += flatten(x)

        return out


def flatten_reform(inp, form):
    assert len(flatten(inp)) == len(flatten(form))

    if not isinstance(form, (tuple, list)):
        if isinstance(inp, (tuple, list)):
            assert len(inp) == 1
            return inp[0]
        else:
            return inp

    out = []
    index = 0
    for sub_form in form:
        if isinstance(sub_form, (tuple, list)):
            sub_form_len = len(flatten(sub_form))
            out.append(flatten_reform(inp[:sub_form_len], sub_form))
            index += sub_form_len
        else:
            out.append(inp[index])
            index += 1

    return out
