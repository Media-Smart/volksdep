import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from ... import utils

__all__ = ['LegacyCalibrator', 'EntropyCalibrator', 'EntropyCalibrator2', 'MinMaxCalibrator']


class BaseCalibrator(object):
    def __init__(
            self,
            data,
            batch_size=1,
            cache_file=None,
    ):
        """base int8 calibrator

        Args:
            data (np.ndarray, torch.Tensor, tuple or list): data for int8 calibration.
            batch_size (int, default is 1): int8 calibrate batch.
            cache_file (string, default is None): int8 calibrate file. if not  None, cache file will be written if file
                not exists and load if file exists.
        """

        super(BaseCalibrator, self).__init__()

        self.batch_size = batch_size
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

    def __str__(self):
        return self.__class__.__name__


class LegacyCalibrator(BaseCalibrator, trt.IInt8LegacyCalibrator):
    def __init__(
            self,
            data,
            batch_size=1,
            cache_file=None,
            quantile=None,
            regression_cutoff=None,
    ):
        """LegacyCalibrator

        Args:
            data (np.ndarray, torch.Tensor, tuple or list): data for int8 calibration.
            batch_size (int, default is 1): int8 calibrate batch
            cache_file (string, default is None): int8 calibrate file. if not None, cache file will be written if file
                not exists and load if file exists.
            quantile (float, default is None): The quantile (between 0 and 1) that will be used to select the region
                maximum when the quantile method is in use. See the user guide for more details on how the quantile is used.
            regression_cutoff (float, default is None): The fraction (between 0 and 1) of the maximum used to define the
                regression cutoff when using regression to determine the region maximum. See the user guide for more details
                on how the regression cutoff is used.
        """

        BaseCalibrator.__init__(self, data, batch_size, cache_file)
        trt.IInt8LegacyCalibrator.__init__(self)

        if quantile:
            self.quantile = quantile

            def get_quantile():
                return self.quantile
            setattr(self, 'get_quantile', get_quantile)

        if regression_cutoff:
            self.regression_cutoff = regression_cutoff

            def get_regression_cutoff():
                return self.regression_cutoff
            setattr(self, 'get_regression_cutoff', get_regression_cutoff)

    def read_histogram_cache(self, length):
        pass

    def write_histogram_cache(self, data, length):
        pass


class EntropyCalibrator(BaseCalibrator, trt.IInt8EntropyCalibrator):
    def __init__(
            self,
            data,
            batch_size=1,
            cache_file=None,

    ):
        """EntropyCalibrator

        Args:
            data (np.ndarray, torch.Tensor, tuple or list): data for int8 calibration
            batch_size (int, default is 1): int8 calibrate batch
            cache_file (string, default is None): int8 calibrate file. if not None, cache file will be written if file
                not exists and load if file exists.
        """

        BaseCalibrator.__init__(self, data, batch_size, cache_file)
        trt.IInt8EntropyCalibrator.__init__(self)


class EntropyCalibrator2(BaseCalibrator, trt.IInt8EntropyCalibrator2):
    def __init__(
            self,
            data,
            batch_size=1,
            cache_file=None,

    ):
        """EntropyCalibrator

        Args:
            data (np.ndarray, torch.Tensor, tuple or list): data for int8 calibration
            batch_size (int, default is 1): int8 calibrate batch
            cache_file (string, default is None): int8 calibrate file. if not None, cache file will be written if file
                not exists and load if file exists.
        """

        BaseCalibrator.__init__(self, data, batch_size, cache_file)
        trt.IInt8EntropyCalibrator2.__init__(self)


class MinMaxCalibrator(BaseCalibrator, trt.IInt8MinMaxCalibrator):
    def __init__(
            self,
            data,
            batch_size=1,
            cache_file=None,

    ):
        """MinMaxCalibrator

        Args:
            data (np.ndarray, torch.Tensor, tuple or list): data for int8 calibration
            batch_size (int, default is 1): int8 calibrate batch
            cache_file (string, default is None): int8 calibrate file. if not None, cache file will be written if file
                not exists and load if file exists.
        """

        BaseCalibrator.__init__(self, data, batch_size, cache_file)
        trt.IInt8MinMaxCalibrator.__init__(self)
