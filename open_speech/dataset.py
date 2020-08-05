from .datasplit import DataSplit
from .util import *

class DataSet:
    """Represents single dataset.

    Attributes:
        path: path to data split files
        name: name of the data split
    """

    def __init__(self, path, name):
        self.path = path
        self.name = name

        self.train = DataSplit(path, "train")
        self.test  = DataSplit(path, "test" )
        self.valid = DataSplit(path, "valid")

    @property
    def sample_rate(self):
        """Returns sample rate of audio data"""
        sample_rate = self.train.sample_rate
        assert sample_rate == self.test.sample_rate
        assert sample_rate == self.valid.sample_rate
        return sample_rate

    @property
    def dtype(self):
        """Returns type of the audio data samples"""
        dtype = self.train.dtype
        assert dtype == self.test.dtype
        assert dtype == self.valid.dtype
        return dtype

    @property
    def files(self):
        """Returns list of all files comprising the dataset"""
        return self.train.files + self.test.files + self.valid.files

    def get_train_dataset(self, num_parallel_reads=AUTOTUNE):
        """Returns training TFRecordDataset of audio data and UUID of matching labels"""
        return self.train.get_dataset(num_parallel_reads)

    @property
    def train_dataset(self):
        """Returns training TFRecordDataset of audio data and UUID of matching labels"""
        return self.get_train_dataset()

    def get_test_dataset(self, num_parallel_reads=AUTOTUNE):
        """Returns test TFRecordDataset of audio data and UUID of matching labels"""
        return self.test.get_dataset(num_parallel_reads)

    @property
    def test_dataset(self):
        """Returns test TFRecordDataset of audio data and UUID of matching labels"""
        return self.get_test_dataset()

    def get_valid_dataset(self, num_parallel_reads=AUTOTUNE):
        """Returns validation TFRecordDataset of audio data and UUID of matching labels"""
        return self.valid.get_dataset(num_parallel_reads)

    @property
    def valid_dataset(self):
        """Returns validation TFRecordDataset of audio data and UUID of matching labels"""
        return self.get_valid_dataset()

    @property
    def train_labels(self):
        """Returns dictionary of training labels in the form "UUID": "text" """
        return self.train.labels

    @property
    def test_labels(self):
        """Returns dictionary of test labels in the form "UUID": "text" """
        return self.test.labels

    @property
    def valid_labels(self):
        """Returns dictionary of validation labels in the form "UUID": "text" """
        return self.valid.labels

    @property
    def labels(self):
        """Returns dictionary of all dataset labels in the form "UUID": "text" """
        return { **self.train_labels, **self.test_labels, **self.valid_labels }
