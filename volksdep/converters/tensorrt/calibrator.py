import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from ... import utils

__all__ = ['Calibrator']


class Calibrator(trt.IInt8Calibrator):
    def __init__(
            self,
            data,
            batch_size=1,
            algorithm='ENTROPY_CALIBRATION_2',
            cache_file=None,
    ):
        """build int8 calibrator

        Args:
            data (np.ndarray, torch.Tensor, tuple or list): data for int8 calibration
            batch_size (int, default is 1): int8 calibrate batch
            algorithm (string, default is ENTROPY_CALIBRATION_2): int8 calibrate algorithm, now support
                LEGACY_CALIBRATION, ENTROPY_CALIBRATION, ENTROPY_CALIBRATION_2, MINMAX_CALIBRATION
            cache_file (string, default is None): int8 calibrate file. if not None, cache file will be written if file
                not exists and load if file exists.
        """

        super(Calibrator, self).__init__()

        self.batch_size = batch_size
        self.algorithm = getattr(trt.CalibrationAlgoType, algorithm)
        self.cache_file = cache_file

        data = utils.to(data, 'numpy')
        data = utils.flatten(data)
        self.data = data
        self.num = data[0].shape[0]
        self.current_index = 0
        self.device_mems = []
        self.bindings = []
        for inp in data:
            device_mem = cuda.mem_alloc(inp[0].nbytes * self.batch_size)
            self.device_mems.append(device_mem)
            self.bindings.append(int(device_mem))

    def get_algorithm(self):
        return self.algorithm

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > self.num:
            return None

        batch_data = [inp[self.current_index:self.current_index + self.batch_size] for inp in self.data]
        [cuda.memcpy_htod(device_mem, data) for device_mem, data in zip(self.device_mems, batch_data)]
        self.current_index += self.batch_size

        return self.bindings

    def read_calibration_cache(self):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        if self.cache_file is not None:
            with open(self.cache_file, "wb") as f:
                f.write(cache)
