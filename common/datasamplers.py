import numpy as np
from abc import ABCMeta
from abc import abstractmethod


class DataSampler(metaclass=ABCMeta):
    """ABC for data sampling"""

    @abstractmethod
    def sample(self, dataset):
        """Should return index samples of the dataset."""


class SequentialDataSampler(DataSampler):
    """Sequential sampler, implementation of ABC DataSampler"""

    def sample(self, dataset):
        """Samples a sequential order of data points."""
        return np.array(range(dataset.size))


class RandomDataSampler(DataSampler):
    """Random sampler, implementation of ABC DataSampler"""

    def sample(self, dataset):
        """Samples a random permutation of the dataset points."""
        return np.random.permutation(range(dataset.size))


class BatchDataSampler:
    """Batch sampler.

    Attributes:
        data_sampler (DataSampler): A data sampler
        batch_size (int, optional): The batch size, defaults to 1
        drop_last (bool, optional): When True, keeps last index batch if smaller than batch size, defaults to False
    """

    def __init__(self, dataset, data_sampler, batch_size=1, drop_last=True):
        self.dataset = dataset
        self.data_sampler = data_sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def sample(self):
        """Samples batches of data point indices based on the `data_sampler`."""
        sample = self.data_sampler.sample(self, dataset=self.dataset)

        if self.batch_size < len(sample):
            # Splitting the array to sub-arrays with batch-size size
            batch = np.array_split(sample, np.arange(self.batch_size, len(sample), self.batch_size))

            # Dropping the last batch if we request for it
            return batch[:-1] if self.drop_last else batch
        else:
            return None
