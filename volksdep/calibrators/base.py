import os

import torch

from .. import utils


class BaseCalibrator(object):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,
    ):
        """base int8 calibrator

        Args:
            dataset (volksdep.datasets.base.Dataset): dataset for int8
                calibration
            batch_size (int, default is 1): int8 calibrate batch size.
            cache_file (string, default is None): int8 calibrate file. if not
                None, cache file will be written if file not exists and load
                if file exists.
        """

        super(BaseCalibrator, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.cache_file = cache_file

        # create buffers that will hold data batches
        self.buffers = []
        for data in utils.flatten(self.dataset[0]):
            size = (self.batch_size,) + tuple(data.shape)
            buf = torch.zeros(
                size=size, dtype=data.dtype, device='cuda').contiguous()
            self.buffers.append(buf)

        self.num_batch = len(dataset) // self.batch_size
        self.batch_idx = 0

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self, *args, **kwargs):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache, *args, **kwargs):
        if self.cache_file is not None:
            with open(self.cache_file, "wb") as f:
                f.write(cache)

    def get_batch(self, *args, **kwargs):
        if self.batch_idx < self.num_batch:
            for i in range(self.batch_size):
                inputs = utils.flatten(
                    self.dataset[self.batch_idx*self.batch_size+i])
                for buffer, inp in zip(self.buffers, inputs):
                    buffer[i].copy_(inp)
            self.batch_idx += 1

            return [int(buf.data_ptr()) for buf in self.buffers]
        else:
            return []

    def __str__(self):
        raise NotImplemented
