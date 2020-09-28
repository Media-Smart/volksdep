from abc import ABCMeta, abstractmethod


class Metric(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        """base metric class
        """

        super(Metric, self).__init__()

    @abstractmethod
    def __call__(self, preds, targets):
        """calculate metric result

        Args:
            preds (np.numpy, list or tuple): outputs from PyTorch model or
                TensorRT engine, and will be flatten automatecally. Please
                make sure outputs order from PyTorch model is the same as
                outputs order from TensorRT egnine.
            targets (np.numpy, list or tuple): targets given by user.
        """
        pass

    @abstractmethod
    def __str__(self):
        pass
