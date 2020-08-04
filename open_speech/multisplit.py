from .datasplit import DataSplit
from .util import *

class MultiSplit:
    """Represents data split (eg, training, validation or test) in multiple
    datasets.

    Attributes:
        name: name of the data split
    """

    def __init__(self, splits, name):
        self._splits = splits
        self.name = name

    @property
    def sample_rate(self):
        """Returns sample rate of audio data"""
        sample_rate = self._splits[0].sample_rate
        assert all([ sample_rate == split.sample_rate for split in self._splits ])
        return sample_rate

    @property
    def dtype(self):
        """Returns type of audio data samples"""
        dtype = self._splits[0].dtype
        assert all([ dtype == split.dtype for split in self._splits ])
        return dtype

    @property
    def files(self):
        """Returns list of files comprising the data split"""
        return [ file for split in self._splits for file in split.files ]

    def get_dataset(self, num_parallel_reads=AUTOTUNE):
        """Returns TFRecordDataset of audio data and UUID of matching labels"""
        return get_dataset(self.files, num_parallel_reads)

    @property
    def dataset(self):
        """Returns TFRecordDataset of audio data and UUID of matching labels"""
        return self.get_dataset()

    @property
    def labels(self):
        """Returns dictionary of labels in the form "UUID": "text" """
        return { uuid: text for split in self._splits for uuid, text in split.labels.items() }