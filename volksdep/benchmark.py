import time
from functools import partial

import torch
import numpy as np
import tensorrt as trt

from . import utils
from .converters import onnx2trt, torch2trt
from .calibrators import EntropyCalibrator2
from .datasets import CustomDataset


__all__ = ['benchmark']


TEMPLATE = '| {:^10} | {:^10} | {:^20} | {:^25} | {:^20} | {:^15} | {:^20} |'

NP_DTYPES = {
    'fp32': np.float32,
    'fp16': np.float16,
    'int8': np.int8,
}
TORCH_DTYPES = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'int8': torch.int8,
}

VALID_DTYPES = ['fp32', 'fp16', 'int8']
TRT_VERSION = trt.__version__
TORCH_VERSION = torch.__version__


def metric_evaluation(model, dtype, dataset, metric):
    value = 'none'
    if dataset and metric:
        preds = []
        tgts = []
        for i in range(len(dataset)):
            inputs, targets = dataset[i]
            inputs = utils.to(inputs, 'torch')
            inputs = utils.to(inputs, 'cuda')
            inputs = utils.to(inputs, TORCH_DTYPES[dtype])
            targets = utils.to(targets, 'numpy')
            inputs, targets = utils.add_batch_dim(inputs), utils.add_batch_dim(targets)

            outs = model(inputs)
            outs = utils.to(outs, 'numpy')
            outs = utils.flatten(outs)

            if i == 0:
                preds = outs
                tgts = targets
            else:
                preds = utils.cat(preds, outs)
                tgts = utils.cat(tgts, targets)
        if len(preds) == 1:
            preds = preds[0]

        preds = utils.to(preds, NP_DTYPES['fp32'])
        value = metric(preds, tgts)

    return value


def speed_evaluation(model, dummy_input, iters=100):
    with torch.no_grad():
        # warm up
        for _ in range(10):
            model(dummy_input)

        # throughput evaluate
        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(dummy_input)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        throughput = int(1.0 * iters / (t1 - t0))

        # latency evaluate
        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(dummy_input)
            torch.cuda.current_stream().synchronize()
        t1 = time.time()
        latency = round(1000.0 * (t1 - t0) / iters, 2)

    return throughput, latency


def torch_benchmark(model, dummy_input, dtype, iters=100, dataset=None, metric=None):
    dummy_input = utils.to(dummy_input, 'cuda')
    dummy_input = utils.to(dummy_input, TORCH_DTYPES[dtype])
    model.cuda().eval().to(TORCH_DTYPES[dtype])

    throughput, latency = speed_evaluation(model, dummy_input, iters)
    value = metric_evaluation(model, dtype, dataset, metric)

    return throughput, latency, value


def trt_benchmark(model, dummy_input, framework, dtype, iters=100, int8_calibrator=None, dataset=None, metric=None):
    dummy_input = utils.to(dummy_input, 'cuda')
    dummy_input = utils.to(dummy_input, TORCH_DTYPES['fp32'])

    if framework == 'torch':
        model.cuda().eval().to(TORCH_DTYPES['fp32'])
        trt_engine = partial(torch2trt, model, dummy_input)
    elif framework == 'onnx':
        trt_engine = partial(onnx2trt, model)
    else:
        raise ValueError('Unsupported framework {}, now only support torch, onnx'.format(framework))

    if dtype == 'fp32':
        model = trt_engine()
    elif dtype == 'fp16':
        model = trt_engine(fp16_mode=True)
    elif dtype == 'int8':
        model = trt_engine(int8_mode=True, int8_calibrator=int8_calibrator)
    else:
        raise TypeError('Unsupported dtype {}'.format(dtype))

    throughput, latency = speed_evaluation(model, dummy_input, iters)
    value = metric_evaluation(model, 'fp32', dataset, metric)

    return throughput, latency, value


def benchmark(
        model,
        shape,
        framework='torch',
        dtypes=('fp32', 'fp16', 'int8'),
        iters=100,
        int8_calibrator=None,
        dataset=None,
        metric=None,
):
    """generate benchmark with given model

    Args:
        model (torch.nn.Module or string): pytorch or onnx model
        shape (tuple, list): model input shapes
        framework (string, default torch): model framework, only torch and onnx are valid strings.
        dtypes (tuple or list, default is ('fp32', 'fp16', 'int8')): dtypes need to be evaluated.
        iters (int, default is 100): larger iters gives more stable performance and cost more time to run.
        int8_calibrator (volksdep.calibrators.base.BaseCalibrator, default is None): calibrator for int8 mode
        dataset (volksdep.datasets.base.Dataset, default is None): used for metric calculation
        metric (volksdep.metrics.base.BaseMetric, default is None): used for metric calculation
    """

    assert set(dtypes).issubset(set(VALID_DTYPES)), 'Unsupported dtypes {}, valid dtpyes are {}'.format(set(dtypes)-set(VALID_DTYPES), VALID_DTYPES)

    metric_name = str(metric) if dataset and metric else 'no metric'
    print(TEMPLATE.format('framework', 'version', 'input shape', 'data type', 'throughput(FPS)', 'latency(ms)', metric_name))
    print(TEMPLATE.format(*[':-:' for _ in range(TEMPLATE.count('|') - 1)]))

    dummy_input = utils.gen_ones_data(shape)
    for dtype in dtypes:
        if framework == 'torch':
            if dtype not in ['fp32', 'fp16']:
                pass
            else:
                throughput, latency, value = torch_benchmark(model, dummy_input, dtype, iters, dataset, metric)
                print(TEMPLATE.format('pytorch', TORCH_VERSION, str(shape), dtype, throughput, latency, value))

        if dtype == 'int8':
            if int8_calibrator is None:
                int8_calibrators = [EntropyCalibrator2(CustomDataset(dummy_input))]
            elif not isinstance(int8_calibrator, (list, tuple)):
                int8_calibrators = [int8_calibrator]
            else:
                int8_calibrators = int8_calibrator

            for int8_calibrator in int8_calibrators:
                throughput, latency, value = trt_benchmark(model, dummy_input, framework, dtype, iters, int8_calibrator, dataset, metric)
                print(TEMPLATE.format('tensorrt', TRT_VERSION, str(shape), '{}({})'.format(dtype, str(int8_calibrator)), throughput, latency, str(value)))
        else:
            throughput, latency, value =  trt_benchmark(model, dummy_input, framework, dtype, iters, int8_calibrator, dataset, metric)
            print(TEMPLATE.format('tensorrt', TRT_VERSION, str(shape), dtype, throughput, latency, str(value)))

        torch.cuda.empty_cache()
