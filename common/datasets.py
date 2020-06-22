from abc import ABCMeta
from abc import abstractmethod

from .datareaders import DataReader


class Dataset(metaclass=ABCMeta):
    """Dataset interface.

    Attributes:
        num_readers (int): The number of data readers
        data_readers (list of DataReader): A list of different data readers
    """

    def __init__(self, data_readers):
        """Constructor"""

        "Check if data_readers is a list or tuple"
        if isinstance(data_readers, (list, tuple)):
            for reader in data_readers:

                "Assert that every reader is an object of DataReader"
                assert isinstance(reader, DataReader)
            self.num_readers = len(data_readers)

        else:
            raise TypeError('data_reader should be a tuple/list of data readers.')

        self.data_readers = data_readers

    @property
    def size(self):
        """Get the size of the dataset."""
        data_size = 0
        for i in range(self.num_readers):
            data_size += self.data_readers[i].size
        return data_size

    @abstractmethod
    def init_readers(self):
        """Initialize readers."""

    @abstractmethod
    def get(self, idxs, reader_idx):
        """Get a batch of a dataset with indices `idxs` from data reader `reader_idx`."""
