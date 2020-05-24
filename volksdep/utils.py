import torch
import numpy as np


def get_names(inp, prefix):
    if not isinstance(inp, (tuple, list)):
        inp = [inp]

    names = []
    for i in range(len(inp)):
        sub_inp = inp[i]
        sub_prefix = '{}.{}'.format(prefix, i)
        if isinstance(sub_inp, (list, tuple)):
            names += get_names(sub_inp, sub_prefix)
        else:
            names.append(sub_prefix)

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
        if type(inp).__module__ == torch.__name__:
            if device_or_dtype == 'torch':
                pass
            elif device_or_dtype == 'numpy':
                inp = inp.detach().cpu().numpy()
            else:
                inp = inp.to(device_or_dtype)
        elif type(inp).__module__ == np.__name__:
            if not isinstance(inp, np.ndarray):
                inp = np.array(inp)

            if device_or_dtype == 'torch':
                inp = torch.from_numpy(inp)
            elif device_or_dtype == 'numpy':
                pass
            else:
                inp = inp.astype(device_or_dtype)
        elif isinstance(inp, (int, float)):
            if device_or_dtype == 'torch':
                inp = torch.tensor(inp)
            elif device_or_dtype == 'numpy':
                inp = np.array(inp)
        else:
            raise TypeError('Unsupported type {}, expect int, float, np.ndarray or torch.Tensor'.format(type(inp)))

        return inp
    else:
        out = []
        for x in inp:
            out.append(to(x, device_or_dtype))

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


def add_batch_dim(inp):
    if not isinstance(inp, (list, tuple)):
        return inp[None, ...]

    out = []
    for x in inp:
        out.append(add_batch_dim(x))

    return out


def cat(x, y, dim=0):
    if isinstance(x, (tuple, list)):
        assert isinstance(y, (tuple, list)) and (len(x) == len(y))

        out = []
        for sub_x, sub_y in zip(x, y):
            out.append(cat(sub_x, sub_y, dim))
        return out
    elif isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor)

        return torch.cat([x, y], dim=dim)
    elif isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray)

        return np.concatenate([x, y], axis=dim)
    else:
        raise TypeError('Unsupported data type {}, expect np.ndarray or torch.Tensor'.format(type(x)))
