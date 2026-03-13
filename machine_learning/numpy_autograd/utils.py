from autograd import Tensor
import numpy as np


def stack(tensors: list[Tensor]) -> Tensor:
    """
    Stack a list of tensors along a new axis.
    Doesn't support autograd since it's only used for batching data, not for model computations.
    """
    data = np.stack([tensor.data for tensor in tensors], dtype=tensors[0].data.dtype)
    return Tensor(data, requires_grad=False)


class Dataset:
    """Base class for datasets."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class DataLoader:
    """Simple data loader for batching and shuffling."""

    def __init__(self, dataset, batch_size=32, shuffle=True, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[start : start + self.batch_size]
            batch = [self.dataset[int(idx)] for idx in batch_indices]
            if self.collate_fn:
                yield self.collate_fn(batch)
            else:
                yield self.default_collate_fn(batch)

    def default_collate_fn(self, batch):
        if isinstance(batch[0], dict):
            collated = {}
            for key in batch[0].keys():
                collated[key] = stack([item[key] for item in batch])
            return collated
        elif isinstance(batch[0], (list, tuple)):
            return tuple(stack([items[i] for items in batch]) for i in range(len(batch[0])))
        else:
            try:
                return stack(batch)
            except Exception as e:
                raise ValueError(f"Cannot collate batch of type {type(batch[0])}: {e}")
