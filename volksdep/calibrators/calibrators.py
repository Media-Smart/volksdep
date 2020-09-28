import tensorrt as trt

from .base import BaseCalibrator


class LegacyCalibrator(BaseCalibrator, trt.IInt8LegacyCalibrator):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,
            quantile=None,
            regression_cutoff=None,
    ):
        """LegacyCalibrator
        Args:
            dataset (volksdep.datasets.base.Dataset): dataset for int8
                calibration
            batch_size (int, default is 1): int8 calibrate batch size.
            cache_file (string, default is None): int8 calibrate file. if not
                None, cache file will be written if file not exists and load
                if file exists.
            quantile (float, default is None): The quantile (between 0 and 1)
                that will be used to select the region maximum when the
                quantile method is in use. See the user guide for more details
                on how the quantile is used.
            regression_cutoff (float, default is None): The fraction
                (between 0 and 1) of the maximum used to define the regression
                cutoff when using regression to determine the region maximum.
                See the user guide for more details on how the regression
                cutoff is used.
        """

        BaseCalibrator.__init__(self, dataset, batch_size, cache_file)
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

    def __str__(self):
        return 'legacy'


class EntropyCalibrator(BaseCalibrator, trt.IInt8EntropyCalibrator):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,

    ):
        """EntropyCalibrator

        Args:
            dataset (volksdep.datasets.base.Dataset): dataset for int8
                calibration
            batch_size (int, default is 1): int8 calibrate batch size.
            cache_file (string, default is None): int8 calibrate file. if not
                None, cache file will be written if file not exists and load
                if file exists.
        """

        BaseCalibrator.__init__(self, dataset, batch_size, cache_file)
        trt.IInt8EntropyCalibrator.__init__(self)

    def __str__(self):
        return 'entropy'


class EntropyCalibrator2(BaseCalibrator, trt.IInt8EntropyCalibrator2):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,

    ):
        """EntropyCalibrator2

        Args:
            dataset (volksdep.datasets.base.Dataset): dataset for int8
                calibration
            batch_size (int, default is 1): int8 calibrate batch size.
            cache_file (string, default is None): int8 calibrate file. if not
                None, cache file will be written if file not exists and load
                if file exists.
        """

        BaseCalibrator.__init__(self, dataset, batch_size, cache_file)
        trt.IInt8EntropyCalibrator2.__init__(self)

    def __str__(self):
        return 'entropy_2'


class MinMaxCalibrator(BaseCalibrator, trt.IInt8MinMaxCalibrator):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,

    ):
        """MinMaxCalibrator

        Args:
            dataset (volksdep.datasets.base.Dataset): dataset for int8
                calibration
            batch_size (int, default is 1): int8 calibrate batch size.
            cache_file (string, default is None): int8 calibrate file. if not
                None, cache file will be written if file not exists and load
                if file exists.
        """

        BaseCalibrator.__init__(self, dataset, batch_size, cache_file)
        trt.IInt8MinMaxCalibrator.__init__(self)

    def __str__(self):
        return 'minmax'
