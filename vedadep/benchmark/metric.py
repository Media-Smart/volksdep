from abc import ABCMeta, abstractmethod

import numpy as np


__all__ = ['BaseMetric', 'Accuracy']


class BaseMetric(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        """base metric class
        """

        super(BaseMetric, self).__init__()

    @abstractmethod
    def metric(self, preds, targets):
        """calculate metric result

        Args:
            preds (np.numpy, list or tuple): outputs from pytorch model or tensorRT engine, and will be flatten
                automatecally. Please make sure outputs order from pytorch model is the same as outputs order from
                tensorRT egnine.
            targets (np.numpy, list or tuple): targets given by user.
        """
        pass

    @abstractmethod
    def metric_name(self):
        """metric name used for display
        """
        pass


class Accuracy(BaseMetric):
    def __init__(self, ignore_index=-1):
        """calculate accuracy

        Args:
            ignore_index (int, default is -1): samples with label equal to ignore_index will be ignored.
        """

        super(Accuracy, self).__init__()

        self.ignore_index = ignore_index

    def metric(self, preds, targets):
        preds = np.argmax(preds, axis=-1)

        mask = targets != self.ignore_index

        true_count = np.sum((targets == preds) & mask)
        total_count = np.sum(mask)

        acc = 1.0 * true_count / total_count

        return acc

    def metric_name(self):
        return 'acc'
