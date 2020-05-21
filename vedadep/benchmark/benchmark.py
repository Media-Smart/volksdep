import time
import copy
import gc

import numpy as np
import torch
import tensorrt as trt

from ..converters import TRTEngine
from .. import utils


__all__ = ['benchmark']


template = '| {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} |'


np_dtypes = {
    'fp32': np.float32,
    'fp16': np.float16,
    'int8': np.int8,
}

torch_dtypes = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'int8': torch.int8,
}


def torch_benchmark(model, dummy_input, dtype, iters=100, dataset=None):
    dummy_input = utils.to(dummy_input, 'torch')
    dummy_input = utils.to(dummy_input, 'cuda')
    dummy_input = utils.to(dummy_input, torch_dtypes[dtype])

    model = copy.deepcopy(model).cuda().to(torch_dtypes[dtype]).eval()

    # warm up
    for _ in range(10):
        model(dummy_input)
    torch.cuda.synchronize()

    with torch.no_grad():
        # throughput evaluate
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(dummy_input)
        torch.cuda.synchronize()
        t1 = time.time()
        throughput = int(1.0 * iters / (t1 - t0))

        # latency evaluate
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(dummy_input)
            torch.cuda.synchronize()
        t1 = time.time()
        latency = round(1000.0 * (t1 - t0) / iters, 2)

        # metric evaluate
        if dataset is not None:
            input_form = utils.get_form(dummy_input)
            data = utils.flatten(dataset.data)
            num = data[0].shape[0]

            pred = []
            for i in range(num):
                batch_data = [torch.from_numpy(d[i:i+1]).cuda().to(torch_dtypes[dtype]) for d in data]
                batch_data = utils.flatten_reform(batch_data, input_form)
                batch_out = model(batch_data)
                batch_out = utils.to(batch_out, 'numpy')
                batch_out = utils.flatten(batch_out)
                if i == 0:
                    pred = batch_out
                else:
                    pred = [np.concatenate([prev_p, batch_p], axis=0) for prev_p, batch_p in zip(pred, batch_out)]
            if len(pred) == 1:
                pred = pred[0]

            metric_value = dataset.metric(pred)
        else:
            metric_value = '-' * 3

    # recycle memory
    dummy_input = utils.to(dummy_input, 'cpu')
    model = model.cpu()

    del dummy_input
    del model

    torch.cuda.empty_cache()
    gc.collect()

    return throughput, latency, metric_value


def trt_benchmark(model, dummy_input, dtype, iters=100, int8_calibrator=None, dataset=None):
    dummy_input = utils.to(dummy_input, 'numpy')
    max_batch_size = utils.flatten(dummy_input)[0].shape[0]

    if dtype == 'fp32':
        engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input, max_batch_size=max_batch_size)
    elif dtype == 'fp16':
        engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input, max_batch_size=max_batch_size, fp16_mode=True)
    elif dtype == 'int8':
        engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input, max_batch_size=max_batch_size, int8_mode=True, int8_calibrator=int8_calibrator)
    else:
        raise TypeError('Unsupported dtype {}'.format(dtype))

    engine.feed(dummy_input)

    # warm up
    for _ in range(10):
        engine.run(max_batch_size)
    engine.stream.synchronize()

    # throughput evaluate
    engine.stream.synchronize()
    t0 = time.time()
    for _ in range(iters):
        engine.run(max_batch_size)
    engine.stream.synchronize()
    t1 = time.time()
    throughput = int(1.0 * iters / (t1 - t0))

    # latency evaluate
    engine.stream.synchronize()
    t0 = time.time()
    for _ in range(iters):
        engine.run(max_batch_size)
        engine.stream.synchronize()
    t1 = time.time()
    latency = round(1000.0 * (t1 - t0) / iters, 2)

    # metric evaluate
    if dataset is not None:
        pred = engine.inference(dataset.data)
        metric_value = dataset.metric(pred)
    else:
        metric_value = '-' * 3

    # recycle memory
    del dummy_input
    del model

    torch.cuda.empty_cache()
    gc.collect()

    return throughput, latency, metric_value


def generate_dummy_input(shapes):
    if isinstance(shapes[0], int):
        return np.random.randn(*shapes).astype(np.float32)

    dummy_input = []
    for shape in shapes:
        dummy_input.append(generate_dummy_input(shape))

    return dummy_input


def benchmark(
        model,
        shape,
        dtypes=('fp32', 'fp16', 'int8'),
        int8_calibrator=None,
        dataset=None,
        iters=100,
):
    """generate benchmark with given model

    Args:
        model (torch.nn.Module): pytorch model
        shape (tuple, list): pytorch model input shapes, data format must match pytorch model input format, for example:
            pytorch model need input format is (x,(y,z)), then shape should be ((b,c,h,w), ((b,c,h,w), (b,c,h,w))). if
            input format is x, then shape should be (b,c,h,w)
        dtypes (tuple or list, default is ('fp32', 'fp16', 'int8')): dtypes need to be evaluated.
        int8_calibrator (vedadep.converters.Calibrator, default is None): if not None, it will be used when int8 dtype
            in dtypes.
        dataset (vedadep.benchmark.Dataset): if not None, benchmark will contain correspoding metric results.
        iters (int, default is 100): larger iters gives more stable performance and cost more time to run.
    """

    for dtype in dtypes:
        if dtype not in ['fp32', 'fp16', 'int8']:
            raise TypeError('Unsupported dtype {}, valid dtpyes are fp32, fp16, int8 '.format(dtype))

    if dataset is None:
        metric_name = 'no metric'
    else:
        metric_name = dataset.metric_name

    print(template.format('framework', 'framework_version', 'input_shape', 'dtype', 'throughput(FPS)', 'latency(ms)', metric_name))

    dummy_input = generate_dummy_input(shape)
    for dtype in dtypes:
        if dtype not in ['fp32', 'fp16']:
            pass
        else:
            throughput, latency, metric = torch_benchmark(model, dummy_input, dtype, iters, dataset)
            print(template.format('pytorch', torch.__version__, str(shape), dtype, throughput, latency, str(metric)))

        throughput, latency, metric = trt_benchmark(model, dummy_input, dtype, iters, int8_calibrator, dataset)
        print(template.format('tensorRT', trt.__version__, str(shape), dtype, throughput, latency, str(metric)))
