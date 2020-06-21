from abc import ABCMeta, abstractmethod


class Dataset(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        """base dataset
        """

        super(Dataset, self).__init__()

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass
