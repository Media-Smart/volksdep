import numpy as np

from .base import Metric


class Accuracy(Metric):
    def __init__(self, ignore_index=-1):
        """calculate accuracy

        Args:
            ignore_index (int, default is -1): samples with label equal to
                ignore_index will be ignored.
        """

        super(Accuracy, self).__init__()

        self.ignore_index = ignore_index

    def __call__(self, preds, targets):
        preds = np.argmax(preds, axis=-1)
        targets = targets.squeeze()

        mask = targets != self.ignore_index

        true_count = np.sum((targets == preds) & mask)
        total_count = np.sum(mask)

        acc = 1.0 * true_count / total_count

        return acc

    def __str__(self):
        return 'acc'
