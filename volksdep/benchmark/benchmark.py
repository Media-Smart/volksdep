import time

import torch
import numpy as np
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


def metric_evaluation(dataset, dtype, metric, model):
    if dataset is not None and metric is not None:
        preds = []
        tgts = []
        for i in range(len(dataset)):
            inputs, targets = dataset[i]
            inputs, targets = utils.to(inputs, 'numpy'), utils.to(targets, 'numpy')
            inputs, targets = utils.add_batch_dim(inputs), utils.add_batch_dim(targets)

            if isinstance(model, torch.nn.Module):
                inputs = utils.to(inputs, 'torch')
                inputs = utils.to(inputs, 'cuda')
                inputs = utils.to(inputs, torch_dtypes[dtype])
                outs = model(inputs)
                outs = utils.to(outs, 'numpy')
            elif isinstance(model, TRTEngine):
                outs = model.inference(inputs)
            else:
                raise TypeError('Unsupported model type {}'.format(type(model)))
            outs = utils.flatten(outs)

            if i == 0:
                preds = outs
                tgts = targets
            else:
                preds = utils.cat(preds, outs)
                tgts = utils.cat(tgts, targets)
        if len(preds) == 1:
            preds = preds[0]

        metric_value = metric.metric(preds, tgts)
    else:
        metric_value = '-' * 3

    return metric_value


def torch_benchmark(model, dummy_input, dtype, iters=100, dataset=None, metric=None):
    dummy_input = utils.to(dummy_input, 'torch')
    dummy_input = utils.to(dummy_input, 'cuda')
    dummy_input = utils.to(dummy_input, torch_dtypes[dtype])

    model = model.cuda().to(torch_dtypes[dtype]).eval()

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
        metric_value = metric_evaluation(dataset, dtype, metric, model)

    # recycle memory
    dummy_input = utils.to(dummy_input, 'cpu')
    model = model.cpu()

    del dummy_input
    del model

    torch.cuda.empty_cache()

    return throughput, latency, metric_value


def trt_benchmark(model, dummy_input, dtype, iters=100, int8_calibrator=None, dataset=None, metric=None):
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
    metric_value = metric_evaluation(dataset, dtype, metric, engine)

    # recycle memory
    del dummy_input

    return throughput, latency, metric_value


def benchmark(
        model,
        shape,
        dtypes=('fp32', 'fp16', 'int8'),
        iters=100,
        int8_calibrator=None,
        dataset=None,
        metric=None,
):
    """generate benchmark with given model

    Args:
        model (torch.nn.Module): pytorch model
        shape (tuple, list): pytorch model input shapes, data format must match pytorch model input format, for example:
            pytorch model need input format is (x,(y,z)), then shape should be ((b,c,h,w), ((b,c,h,w), (b,c,h,w))). if
            input format is x, then shape should be (b,c,h,w)
        dtypes (tuple or list, default is ('fp32', 'fp16', 'int8')): dtypes need to be evaluated.
        iters (int, default is 100): larger iters gives more stable performance and cost more time to run.
        int8_calibrator (vedadep.converters.Calibrator, default is None): if not None, it will be used when int8 dtype
            in dtypes.
        dataset (vedadep.benchmark.dataset.BaseDataset): if not None, benchmark will contain correspoding metric results.
        metric (vedadep.benchmark.metric.BaseMetric): if not None, benchmark will contain correspoding metric results.
    """

    for dtype in dtypes:
        if dtype not in ['fp32', 'fp16', 'int8']:
            raise TypeError('Unsupported dtype {}, valid dtpyes are fp32, fp16, int8 '.format(dtype))

    if dataset is None or metric is None:
        metric_name = '-' * 3
    else:
        metric_name = metric.metric_name()

    print(template.format('framework', 'framework_version', 'input_shape', 'dtype', 'throughput(FPS)', 'latency(ms)', metric_name))

    dummy_input = utils.gen_ones_data(shape)
    for dtype in dtypes:
        if dtype not in ['fp32', 'fp16']:
            pass
        else:
            throughput, latency, metric_value = torch_benchmark(model, dummy_input, dtype, iters, dataset, metric)
            print(template.format('pytorch', torch.__version__, str(shape), dtype, throughput, latency, str(metric_value)))

        throughput, latency, metric_value = trt_benchmark(model, dummy_input, dtype, iters, int8_calibrator, dataset, metric)
        print(template.format('tensorRT', trt.__version__, str(shape), dtype, throughput, latency, str(metric_value)))
