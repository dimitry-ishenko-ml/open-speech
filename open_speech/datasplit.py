from .util import *

class DataSplit:
    """Represents data split (eg, training, validation or test) in a dataset.

    Attributes:
        path: path to data split files
        name: name of the data split
    """

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self._metadata = None

    def _get_metadata(self):
        if self._metadata is None:
            self._metadata = get_metadata(self.path + "/" + self.name + ".pickle")
            self._metadata["files"] = [
                self.path + "/" + file for file in self._metadata["files"]
            ]
        return self._metadata

    @property
    def sample_rate(self):
        """Returns sample rate of audio data"""
        return self._get_metadata()["sample_rate"]

    @property
    def dtype(self):
        """Returns type of audio data samples"""
        return self._get_metadata()["dtype"]

    @property
    def files(self):
        """Returns list of files comprising the data split"""
        return self._get_metadata()["files"]

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
        return self._get_metadata()["labels"]