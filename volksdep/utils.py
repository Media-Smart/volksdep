import torch
import numpy as np


def get_names(inp, prefix):
    if not isinstance(inp, (tuple, list)):
        inp = [inp]

    names = []
    for i, sub_inp in enumerate(inp):
        sub_prefix = '{}.{}'.format(prefix, i)
        if isinstance(sub_inp, (list, tuple)):
            names.extend(get_names(sub_inp, sub_prefix))
        else:
            names.append(sub_prefix)

    return names


def get_forms(inp):
    if not isinstance(inp, (tuple, list)):
        return 'x'

    forms = []
    for sub_inp in inp:
        forms.append(get_forms(sub_inp))

    return forms


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
            raise TypeError(('Unsupported type {}, expect int, float, '
                             'np.ndarray or torch.Tensor').format(type(inp)))

        return inp

    out = []
    for sub_inp in inp:
        out.append(to(sub_inp, device_or_dtype))

    return out


def flatten(inp):
    if not isinstance(inp, (tuple, list)):
        return [inp]

    out = []
    for sub_inp in inp:
        out.extend(flatten(sub_inp))

    return out


def reconstruct(inp, forms):
    assert len(flatten(inp)) == len(flatten(forms))

    if not isinstance(forms, (tuple, list)):
        if isinstance(inp, (tuple, list)):
            assert len(inp) == 1
            return inp[0]
        else:
            return inp

    out = []
    index = 0
    for sub_form in forms:
        if isinstance(sub_form, (tuple, list)):
            sub_form_len = len(flatten(sub_form))
            out.append(reconstruct(inp[:sub_form_len], sub_form))
            index += sub_form_len
        else:
            out.append(inp[index])
            index += 1

    return out


def add_batch_dim(inp):
    if not isinstance(inp, (list, tuple)):
        return inp[None, ...]

    out = []
    for sub_inp in inp:
        out.append(add_batch_dim(sub_inp))

    return out


def cat(x, y, dim=0):
    x = list(x) if isinstance(x, (tuple, list)) else x
    y = list(y) if isinstance(y, (tuple, list)) else y

    assert type(x) == type(y)

    if isinstance(x, list):
        assert len(x) == len(y)

        out = []
        for sub_x, sub_y in zip(x, y):
            out.append(cat(sub_x, sub_y, dim))
        return out
    elif isinstance(x, torch.Tensor):
        return torch.cat([x, y], dim=dim)
    elif isinstance(x, np.ndarray):
        return np.concatenate([x, y], axis=dim)
    else:
        raise TypeError(('Unsupported data type {}, '
                         'expect np.ndarray or torch.Tensor').format(type(x)))


def gen_ones(shape):
    if isinstance(shape[0], int):
        return torch.ones(*shape)

    data = []
    for sub_shape in shape:
        data.append(gen_ones(sub_shape))

    return data


def fetch_batch(data, start, end):
    if isinstance(data, torch.Tensor):
        return data[start:end]

    assert not isinstance(data, (tuple, list)), (
        'Unsupported data type {}, only torch.Tensor, '
        'tuple or list are supported').format(type(data))

    batch = []
    for sub_data in data:
        batch.append(fetch_batch(sub_data, start, end))

    return batch

