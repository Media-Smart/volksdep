## Introduction
volksdep is an open-source toolbox for deploying and accelerating PyTorch model with TensorRT in x86_64 and aarch64 platform.

## Features
- **Auto transformation and acceleration**\
    volksdep can automatically transform and accelerate PyTorch model with TensorRT by writing only some few 
    codes.

- **Auto benchmark**\
    volksdep can automatically generate benchmark with given PyTorch model.

## License
This project is released under [Apache 2.0 license](https://github.com/Media-Smart/volksdep/blob/master/LICENSE).

## Installation
### Requirements

- Linux
- Python 3.6.x
- TensorRT 6.0.x.x
- PyTorch 1.2.0
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- Python 3.6.9
- TensorRT 6.0.1.5
- PyTorch 1.2.0
- CUDA: 10.1

### Install volksdep

1. Install TensorRT following the [official instructions](https://developer.nvidia.com/tensorrt/)

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/)

3. If your platform is x86, you can create a conda virtual environment and activate it.

```shell
conda create -n volksdep python=3.6.9 -y
conda activate volksdep
```

4. Clone the volksdep repository.

```shell
git clone https://github.com/Media-Smart/volksdep.git
cd volksdep
```

5. Setup.

```shell
python setup.py install
```

## Known Issues
1. Dynamic shape input is not supported.
2. PyTorch Upsample operation is supported with specified size, nereast mode and align_corners being False.

## Usage
### Convert
```shell
import numpy as np
import torch
import torchvision
from volksdep.converters import TRTEngine, Calibrator

# create dummy input for tensorRT engine building.
dummy_input = torch.ones(1, 3, 224, 224)
# create pytorch model
model = torchvision.models.resnet18()

# build engine with fp32 mode
engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input)
# build engine with fp16 mode
# engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input, fp16_mode=True)
# build engine with int8 mode
# engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input, int8_mode=True)
# build engine with int8 mode and calibrator
# dummy_calibrator = Calibrator(data=np.ones((2, 3, 224, 224)).astype(np.float32))
# engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input, int8_mode=True, int8_calibrator=dummy_calibrator)
```
### Execute
```shell
model = model.cuda().eval()
torch_output = model(dummy_input.cuda()).detach().cpu().numpy()
# inference input can be numpy data or torch.Tensor data
trt_output = engine.inference(dummy_input.cpu().numpy())

print(np.max(np.abs(torch_output-trt_output)))
```
### Save and load
We can save the builded engine
```shell
engine.save('resnet18.engine')
```
We can load the saved engine
```shell
from volksdep.converters import TRTEngine

engine = TRTEngine(build_from='engine', name='resnet18.engine')
```
### Benchmark
```shell
import numpy as np
import torchvision
from volksdep.converters import Calibrator
from volksdep.benchmark import benchmark
from volksdep.benchmark.dataset import CustomDataset
from volksdep.benchmark.metric import Accuracy


# create pytorch model
model = torchvision.models.resnet18()

# simple benchmark, only test throughput and latency
benchmark(model=model, shape=(1, 3, 224, 224), dtypes=['fp32', 'fp16', 'int8'])

# benchmark with specified metric, should provide test dataset
dummy_inputs = np.random.randn(100, 3, 224, 224).astype(np.float32)
dummy_targets = np.random.randint(0, 1001, size=(100,))
dummy_dataset = CustomDataset(data=(dummy_inputs, dummy_targets))
metric = Accuracy()
benchmark(model=model, shape=(1, 3, 224, 224), dataset=dummy_dataset, metric=metric)

# when int8 in dtypes, we can also add calibration data for int8 calibration
dummy_calibrator = Calibrator(data=np.random.randn(10, 3, 224, 224).astype(np.float32))
benchmark(model=model, shape=(1, 3, 224, 224), int8_calibrator=dummy_calibrator, dataset=dummy_dataset, metric=metric)
```
We can define our own dataset.
```shell
import numpy as np
from volksdep.benchmark.dataset import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self):
        super(MyDataset, self).__init__()

        self.dummy_inputs = np.random.randn(100, 3, 224, 224).astype(np.float32)
        self.dummy_targets = np.random.randint(0, 1001, size=(100,))

    def __getitem__(self, index):
        return self.dummy_inputs[index], self.dummy_targets[index]

    def __len__(self):
        return len(self.dummy_inputs)
```
We can define our own metric.
```shell
from volksdep.benchmark.metric import BaseMetric

class MyMetric(BaseMetric):
    def __init__(self):
        super(MyMetric, self).__init__()

    def metric(self, pred, target):
        pred = np.argmax(pred, axis=-1)
        acc = 1.0 * np.sum(pred == target) / len(target.flatten())

        return acc

    def metric_name(self):
        return 'my_metric'
```

## Contact
This repository is currently maintained by Hongxiang Cai ([@hxcai](http://github.com/hxcai)).
