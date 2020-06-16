## Introduction
volksdep is an open-source toolbox for deploying and accelerating PyTorch, Onnx and Tensorflow models with TensorRT.

## Features
- **Auto transformation and acceleration**\
    volksdep can automatically transform and accelerate PyTorch, Onnx and Tensorflow models with TensorRT by writing 
    only some few codes.

- **Auto benchmark**\
    volksdep can automatically generate benchmark with given model.

## License
This project is released under [Apache 2.0 license](https://github.com/Media-Smart/volksdep/blob/master/LICENSE).

## Installation
### Requirements

- Linux
- Python 3.6 or higher
- TensorRT 7.0.0.11 or higher
- PyTorch 1.2.0 or higher
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- Python 3.6.9
- TensorRT 7.0.0.11
- PyTorch 1.2.0
- CUDA: 10.2

### Install volksdep

1. Install TensorRT following the [official instructions](https://developer.nvidia.com/tensorrt/)

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/)

3. If your platform is x86 or x64, you can create a conda virtual environment and activate it.

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
1. Input should be with explicit batch and shape.
2. PyTorch Upsample operation is supported with specified size, nereast mode and align_corners being None.

## Usage
### Convert
More available arguments of build_from_torch, build_from_onnx and build_from_engine are detailed in 
[volksdep/converters/tensorrt/engine.py](https://github.com/Media-Smart/volksdep/blob/master/volksdep/converters/tensorrt/engine.py)

#### PyTorch to TensorRT engine
```shell
import numpy as np
import torch
import torchvision
from volksdep.converters import TRTEngine
from volksdep.converters import EntropyCalibrator2

# create dummy input for tensorrt engine building.
dummy_input = torch.ones(1, 3, 224, 224)
# create pytorch model
model = torchvision.models.resnet18()

# build engine with fp32 mode
engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input)
# build engine with fp16 mode
# engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input, fp16_mode=True)
# build engine with int8 mode
# engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input, int8_mode=True)
# build engine with int8 mode and provided data using EntropyCalibrator2
# dummy_calibrator = EntropyCalibrator2(data=np.ones((2, 3, 224, 224)).astype(np.float32))
# engine = TRTEngine(build_from='torch', model=model, dummy_input=dummy_input, int8_mode=True, int8_calibrator=dummy_calibrator)
```

#### Onnx to TensorRT engine
##### First convert other frameworks to onnx(optional)

PyTorch to Onnx
```shell
import torch
import torchvision
from volksdep.converters import torch2onnx

# create dummy input.
dummy_input = torch.ones(1, 3, 224, 224)
# create pytorch model
model = torchvision.models.resnet18()
# use torch2onnx to convert pytorch model to onnx.
torch2onnx(model, dummy_input, 'resnet18.onnx')
```
More available arguments of torch2onnx are detailed in 
[volksdep/converters/onnx/converter.py](https://github.com/Media-Smart/volksdep/blob/master/volksdep/converters/onnx/converter.py)

[Tensorflow to Onnx](https://github.com/onnx/tensorflow-onnx)

[Keras to Onnx](https://github.com/onnx/keras-onnx)


##### Then convert Onnx to TensorRT engine
```shell
import numpy as np
from volksdep.converters import TRTEngine
from volksdep.converters import EntropyCalibrator2

onnx_model = 'resnet18.onnx'

# build engine with fp32 mode
engine = TRTEngine(build_from='onnx', model=onnx_model)
# build engine with fp16 mode
# engine = TRTEngine(build_from='onnx', model=onnx_model, fp16_mode=True)
# build engine with int8 mode
# engine = TRTEngine(build_from='onnx', model=onnx_model, int8_mode=True)
# build engine with int8 mode and provided data using EntropyCalibrator2
# dummy_calibrator = EntropyCalibrator2(data=np.ones((2, 3, 224, 224)).astype(np.float32))
# engine = TRTEngine(build_from='onnx', model=onnx_model, int8_mode=True, int8_calibrator=dummy_calibrator)
```

### Execute inference
```shell
# input can be numpy data or torch.Tensor data
trt_output = engine.inference(dummy_input.numpy())

print(trt_output.shape)
```

### Save and load
#### Save
```shell
engine.save('resnet18.engine')
```
#### Load
```shell
from volksdep.converters import TRTEngine

engine = TRTEngine(build_from='engine', model='resnet18.engine')
```

### Benchmark
#### PyTorch benchmark
```shell
import numpy as np
import torchvision
from volksdep.converters import EntropyCalibrator, EntropyCalibrator2, MinMaxCalibrator
from volksdep.benchmark import benchmark
from volksdep.benchmark.dataset import CustomDataset
from volksdep.benchmark.metric import Accuracy


# create pytorch model
model = torchvision.models.resnet18()

# simple benchmark, only test throughput and latency
benchmark(model=model, shape=(1, 3, 224, 224), dtypes=['fp32', 'fp16', 'int8'])

# benchmark with provided test dataset and specified metric
dummy_inputs = np.random.randn(100, 3, 224, 224).astype(np.float32)
dummy_targets = np.random.randint(0, 1001, size=(100,))
dummy_dataset = CustomDataset(data=(dummy_inputs, dummy_targets))
metric = Accuracy()
benchmark(model=model, shape=(1, 3, 224, 224), dataset=dummy_dataset, metric=metric)

# when int8 in dtypes, we can also add calibration data for int8 calibration
dummy_calibration_data = np.random.randn(10, 3, 224, 224).astype(np.float32)
dummy_calibrators = [EntropyCalibrator(data=dummy_calibration_data), EntropyCalibrator2(data=dummy_calibration_data), MinMaxCalibrator(data=dummy_calibration_data)]
benchmark(model=model, shape=(1, 3, 224, 224), int8_calibrator=dummy_calibrators, dataset=dummy_dataset, metric=metric)
```
#### Onnx benchmark
```shell
import numpy as np
from volksdep.converters import EntropyCalibrator, EntropyCalibrator2, MinMaxCalibrator
from volksdep.benchmark import benchmark
from volksdep.benchmark.dataset import CustomDataset
from volksdep.benchmark.metric import Accuracy

onnx_model = 'resnet18.onnx'

# simple benchmark, only test throughput and latency
benchmark(model=onnx_model, shape=(1, 3, 224, 224), build_from='onnx', dtypes=['fp32', 'fp16', 'int8'])

# benchmark with specified metric, should provide test dataset
dummy_inputs = np.random.randn(100, 3, 224, 224).astype(np.float32)
dummy_targets = np.random.randint(0, 1001, size=(100,))
dummy_dataset = CustomDataset(data=(dummy_inputs, dummy_targets))
metric = Accuracy()
benchmark(model=onnx_model, shape=(1, 3, 224, 224), build_from='onnx', dataset=dummy_dataset, metric=metric)

# when int8 in dtypes, we can also add calibration data for int8 calibration
dummy_calibration_data = np.random.randn(10, 3, 224, 224).astype(np.float32)
dummy_calibrators = [EntropyCalibrator(data=dummy_calibration_data), EntropyCalibrator2(data=dummy_calibration_data), MinMaxCalibrator(data=dummy_calibration_data)]
benchmark(model=onnx_model, shape=(1, 3, 224, 224), build_from='onnx', int8_calibrator=dummy_calibrators, dataset=dummy_dataset, metric=metric)
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
