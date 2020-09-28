from .base import Dataset
from .. import utils


class CustomDataset(Dataset):
    def __init__(self, inputs, targets=None):
        """custom dataset

        Args:
            inputs (torch.Tensor, tuple or list)
            targets (torch.Tensor, tuple or list)
        """

        super(CustomDataset, self).__init__()

        self.inputs_forms = utils.get_forms(inputs)
        self.inputs = utils.flatten(inputs)

        if targets is not None:
            self.targets_forms = utils.get_forms(targets)
            self.targets = utils.flatten(targets)

    def __getitem__(self, idx):
        inputs = [inp[idx] for inp in self.inputs]
        inputs = utils.reconstruct(inputs, self.inputs_forms)

        if hasattr(self, 'targets'):
            targets = [target[idx] for target in self.targets]
            targets = utils.reconstruct(targets, self.targets_forms)

            return inputs, targets

        return inputs

    def __len__(self):
        return self.inputs[0].shape[0]
