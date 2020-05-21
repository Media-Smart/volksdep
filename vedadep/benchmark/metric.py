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
    def metric(self, pred, target):
        """calculate metric result

        Args:
            pred (np.numpy, list or tuple): outputs from pytorch model or tensorRT engine, and will be flatten
                automatecally. For example, if pytorch output like (x,(y,z)), then the pred form is (x,y,z).
            target (np.numpy, list or tuple): target given by user.
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

    def metric(self, pred, target):
        pred = np.argmax(pred, axis=-1)

        mask = target != self.ignore_index

        true_count = np.sum((target == pred) & mask)
        total_count = np.sum(mask)

        acc = 1.0 * true_count / total_count

        return acc

    def metric_name(self):
        return 'acc'
