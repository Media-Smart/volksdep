## Introduction
volksdep is an open-source toolbox for deploying and accelerating PyTorch, Onnx and Tensorflow models with TensorRT.

## Features
- **Auto conversion and acceleration**\
    volksdep can accelerate PyTorch, Onnx and Tensorflow models using TensorRT with 
    only some few codes.

- **Benchmark of throughput, latency and metric**\
    volksdep can generate benchmark of throughput, latency and metric with given model.

## License
This project is released under [Apache 2.0 license](https://github.com/Media-Smart/volksdep/blob/master/LICENSE).

## Installation
### Requirements

- Linux
- Python 3.6 or higher
- TensorRT 7.1.0.16 or higher
- PyTorch 1.4.0 or higher
- CUDA 10.2 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- Python 3.6.9
- TensorRT 7.1.3.4
- PyTorch 1.4.0
- CUDA: 10.2

### Install volksdep

1. If your platform is x86 or x64, you can create a conda virtual environment and activate it.

```shell
conda create -n volksdep python=3.6.9 -y
conda activate volksdep
```

2. Install TensorRT following the [official instructions](https://developer.nvidia.com/tensorrt/)

3. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/)

4. Setup.

```shell
pip install "git+https://github.com/Media-Smart/volksdep.git"
```

## Known Issues
1. PyTorch Upsample operation is supported with specified size, nearest mode and align_corners being None.

## Usage
### Convert

#### PyTorch to TensorRT
```shell
import torch
import torchvision
from volksdep.converters import torch2trt
from volksdep.calibrators import EntropyCalibrator2
from volksdep.datasets import CustomDataset

dummy_input = torch.ones(1, 3, 224, 224).cuda()
model = torchvision.models.resnet18().cuda().eval()

## build trt model with fp32 mode
trt_model = torch2trt(model, dummy_input)
## build trt model with fp16 mode
# trt_model = torch2trt(model, dummy_input, fp16_mode=True)
## build trt model with int8 mode
# trt_model = torch2trt(model, dummy_input, int8_mode=True)
## build trt model with int8 mode and provided data using EntropyCalibrator2
# dummy_calibrator = EntropyCalibrator2(CustomDataset(torch.randn(4, 3, 224, 224)))
# trt_model = torch2trt(model, dummy_input, int8_mode=True, int8_calibrator=dummy_calibrator)
```
More available arguments of torch2trt are detailed in 
[volksdep/converters/torch2trt.py](https://github.com/Media-Smart/volksdep/blob/master/volksdep/converters/torch2trt.py)

#### Onnx to TensorRT
```shell
import torch
from volksdep.converters import onnx2trt
from volksdep.calibrators import EntropyCalibrator2
from volksdep.datasets import CustomDataset

model = 'resnet18.onnx'

## build trt model with fp32 mode
trt_model = onnx2trt(model)
## build trt model with fp16 mode
# trt_model = onnx2trt(model, fp16_mode=True)
## build trt model with int8 mode
# trt_model = onnx2trt(model, int8_mode=True)
## build trt model with int8 mode and provided data using EntropyCalibrator2
# dummy_calibrator = EntropyCalibrator2(CustomDataset(torch.randn(4, 3, 224, 224)))
# trt_model = onnx2trt(model, int8_mode=True, int8_calibrator=dummy_calibrator)
```
More available arguments of onnx2trt are detailed in 
[volksdep/converters/onnx2trt.py](https://github.com/Media-Smart/volksdep/blob/master/volksdep/converters/onnx2trt.py)

#### Other frameworks to Onnx
1. PyTorch to Onnx
```shell
import torch
import torchvision
from volksdep.converters import torch2onnx

dummy_input = torch.ones(1, 3, 224, 224).cuda()
model = torchvision.models.resnet18().cuda().eval()
torch2onnx(model, dummy_input, 'resnet18.onnx')
```
More available arguments of torch2onnx are detailed in 
[volksdep/converters/torch2onnx.py](https://github.com/Media-Smart/volksdep/blob/master/volksdep/converters/torch2onnx.py)

2. [Tensorflow to Onnx](https://github.com/onnx/tensorflow-onnx)

3. [Keras to Onnx](https://github.com/onnx/keras-onnx)

### Execute inference
```shell
with torch.no_grad():
    trt_output = trt_model(dummy_input)
    print(trt_output.shape)
```

### Save and load
#### Save
```shell
from volksdep.converters import save

save(trt_model, 'resnet18.engine')
```
#### Load
```shell
from volksdep.converters import load

trt_model = load('resnet18.engine')
```

### Benchmark
#### PyTorch benchmark
```shell
import torch
import torchvision
from volksdep import benchmark
from volksdep.calibrators import EntropyCalibrator, EntropyCalibrator2, MinMaxCalibrator
from volksdep.datasets import CustomDataset
from volksdep.metrics import Accuracy

model = torchvision.models.resnet18()

## simple benchmark, only test throughput and latency
benchmark(model, (1, 3, 224, 224), dtypes=['fp32', 'fp16', 'int8'])
## benchmark with provided test dataset and metric
# dummy_inputs = torch.randn(100, 3, 224, 224)
# dummy_targets = torch.randint(0, 1001, size=(100,))
# dummy_dataset = CustomDataset(dummy_inputs, dummy_targets)
# metric = Accuracy()
# benchmark(model, (1, 3, 224, 224), dataset=dummy_dataset, metric=metric)
## benchmark with provided test dataset, metric and  data for int8 calibration
# dummy_data = torch.randn(10, 3, 224, 224)
# dummy_calibrators = [
#     EntropyCalibrator(CustomDataset(dummy_data)),
#     EntropyCalibrator2(CustomDataset(dummy_data)),
#     MinMaxCalibrator(CustomDataset(dummy_data))
# ]
# dummy_dataset = CustomDataset(torch.randn(100, 3, 224, 224), torch.randint(0, 1001, size=(100,)))
# metric = Accuracy()
# benchmark(model, (1, 3, 224, 224), int8_calibrator=dummy_calibrators, dataset=dummy_dataset, metric=metric)
```

#### Onnx benchmark
```shell
import torch
import torchvision
from volksdep import benchmark
from volksdep.calibrators import EntropyCalibrator, EntropyCalibrator2, MinMaxCalibrator
from volksdep.datasets import CustomDataset
from volksdep.metrics import Accuracy

model = 'resnet18.onnx'

## simple benchmark, only test throughput and latency
benchmark(model, (1, 3, 224, 224), framework='onnx', dtypes=['fp32', 'fp16', 'int8'])
## benchmark with provided test dataset and metric
# dummy_inputs = torch.randn(100, 3, 224, 224)
# dummy_targets = torch.randint(0, 1001, size=(100,))
# dummy_dataset = CustomDataset(dummy_inputs, dummy_targets)
# metric = Accuracy()
# benchmark(model, (1, 3, 224, 224), framework='onnx', dataset=dummy_dataset, metric=metric)
## benchmark with provided test dataset, metric and  data for int8 calibration
# dummy_data = torch.randn(10, 3, 224, 224)
# dummy_calibrators = [
#     EntropyCalibrator(CustomDataset(dummy_data)),
#     EntropyCalibrator2(CustomDataset(dummy_data)),
#     MinMaxCalibrator(CustomDataset(dummy_data))
# ]
# dummy_dataset = CustomDataset(torch.randn(100, 3, 224, 224), torch.randint(0, 1001, size=(100,)))
# metric = Accuracy()
# benchmark(model, (1, 3, 224, 224), framework='onnx', int8_calibrator=dummy_calibrators, dataset=dummy_dataset, metric=metric)
```

We can define our own dataset and metric for int8 calibration and metric calculation.
```shell
import numpy as np
import torch
import torchvision
from volksdep.datasets import Dataset
from volksdep.calibrators import EntropyCalibrator2
from volksdep.metrics import Metric
from volksdep import benchmark


class DatasetForCalibration(Dataset):
    def __init__(self):
        super(DatasetForCalibration, self).__init__()

        self.dummy_inputs = torch.randn(10, 3, 224, 224)

    def __getitem__(self, idx):
        return self.dummy_inputs[idx]

    def __len__(self):
        return len(self.dummy_inputs)


class DatasetForMetric(Dataset):
    def __init__(self):
        super(DatasetForMetric, self).__init__()

        self.dummy_inputs = torch.randn(100, 3, 224, 224)
        self.dummy_targets = torch.randint(0, 1001, size=(100,))

    def __getitem__(self, idx):
        return self.dummy_inputs[idx], self.dummy_targets[idx]

    def __len__(self):
        return len(self.dummy_inputs)


class MyMetric(Metric):
    def __init__(self):
        super(MyMetric, self).__init__()

    def __call__(self, preds, targets):
        pred = np.argmax(preds, axis=-1)
        acc = 1.0 * np.sum(pred == targets) / len(targets.flatten())

        return acc

    def __str__(self):
        return 'my_metric'


dummy_input = torch.randn(1, 3, 224, 224).cuda()
model = torchvision.models.resnet18().cuda().eval()
calibrator = EntropyCalibrator2(DatasetForCalibration())
dataset = DatasetForMetric()
metric = MyMetric()

benchmark(model, (1, 3, 224, 224), int8_calibrator=calibrator, dataset=dataset, metric=metric)
```

## Contact
This repository is currently maintained by Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).
