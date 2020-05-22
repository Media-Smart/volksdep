from abc import ABCMeta, abstractmethod

from .. import utils

__all__ = ['BaseDataset', 'CustomDataset']


class BaseDataset(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        """base dataset
        """

        super(BaseDataset, self).__init__()

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class CustomDataset(BaseDataset):
    def __init__(self, data):
        """dataset for metric calculation

        Args:
            data (tuple, list): data used for pytorch model and tensorRT engine. data form should be like
                (inputs, targets).
        """

        super(CustomDataset, self).__init__()

        inputs, targets = data

        self.inputs_form = utils.get_form(inputs)
        self.targets_form = utils.get_form(targets)

        self.inputs = utils.flatten(inputs)
        self.targets = utils.flatten(targets)

    def __getitem__(self, index):
        inputs = [inp[index] for inp in self.inputs]
        targets = [target[index] for target in self.targets]

        inputs = utils.flatten_reform(inputs, self.inputs_form)
        targets = utils.flatten_reform(targets, self.targets_form)

        return inputs, targets

    def __len__(self):
        return self.inputs[0].shape[0]
