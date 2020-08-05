from .multisplit import MultiSplit
from .util import *

class MultiSet:
    """Collection of datasets.

    `MultiSet` contains a number of `DataSet` instances and acts as a single
    dataset.

    Datasets within the collection can be accessed by iterating over it, eg:

    for dataset in open_speech.datasets:
        print(dataset.name)

    Data splits (training, validation and test) within the collection are
    represented by instances of `MultiSplit` and can be accessed using `train`,
    `valid` and `test` attributes.

    Labels for each split can be accessed directly using `train_labels`,
    `valid_label` and `test_labels` properties.

    And, recordsets can be accessed using `train_recordset`, `valid_recordset`,
    `test_recordset` properties.

    Attributes:
        train: Training data split.
        valid: Validation data split.
        test: Test data split.
    """

    def __init__(self, datasets):
        self._datasets = datasets

        self.train = MultiSplit([ ds.train for ds in datasets ], "train")
        self.test  = MultiSplit([ ds.test  for ds in datasets ], "test" )
        self.valid = MultiSplit([ ds.valid for ds in datasets ], "valid")

    def __iter__(self):
        for dataset in self._datasets: yield dataset

    @property
    def sample_rate(self):
        """Returns audio sample rate."""

        sample_rate = self.train.sample_rate
        assert sample_rate == self.test.sample_rate
        assert sample_rate == self.valid.sample_rate
        return sample_rate

    @property
    def dtype(self):
        """Returns data type of audio samples."""

        dtype = self.train.dtype
        assert dtype == self.test.dtype
        assert dtype == self.valid.dtype
        return dtype

    @property
    def files(self):
        """Returns list of all `TFRecord` files comprising the collection."""

        return self.train.files + self.test.files + self.valid.files

    @property
    def train_labels(self):
        """Returns `dict` of training labels in the form "uuid": "label"."""

        return self.train.labels

    @property
    def valid_labels(self):
        """Returns `dict` of validation labels in the form "uuid": "label"."""

        return self.valid.labels

    @property
    def test_labels(self):
        """Returns `dict` of test labels in the form "uuid": "label"."""

        return self.test.labels

    @property
    def labels(self):
        """Returns `dict` of all collection labels in the form "uuid": "label"."""

        return { **self.train_labels, **self.test_labels, **self.valid_labels }

    def get_train_recordset(self, num_parallel_reads=AUTOTUNE):
        """Returns training split recordset (see `MultiSet.train_recordset`)."""

        return self.train.get_recordset(num_parallel_reads)

    @property
    def train_recordset(self):
        """Returns training split recordset.

        Recordset is an instance of `tf.data.TFRecordDataset` containing uuids and
        audio data in serialized form.

        Use `parse_serial` to extract (uuid, audio) tuples and `lookup_table` to
        translate uuids into labels.

        Example:

        table = lookup_table(open_speech.labels)

        ds = open_speech.train_recordset # instance of MultiSet
        ds = ds.map(parse_serial) # serial -> (uuid, audio)
        ds = ds.map(lambda uuid, audio: (audio, table.lookup(uuid))) # (uuid, audio) -> (audio, label)

        Returns:
            An instance of `tf.data.TFRecordDataset`.
        """

        return self.get_train_recordset()

    def get_valid_recordset(self, num_parallel_reads=AUTOTUNE):
        """Returns validation split recordset (see `MultiSet.valid_recordset`)."""
        return self.valid.get_recordset(num_parallel_reads)

    @property
    def valid_recordset(self):
        """Returns validation split recordset.

        Recordset is an instance of `tf.data.TFRecordDataset` containing uuids and
        audio data in serialized form.

        Use `parse_serial` to extract (uuid, audio) tuples and `lookup_table` to
        translate uuids into labels.

        Example:

        table = lookup_table(open_speech.labels)

        ds = open_speech.valid_recordset
        ds = ds.map(parse_serial) # serial -> (uuid, audio)
        ds = ds.map(lambda uuid, audio: (audio, table.lookup(uuid))) # (uuid, audio) -> (audio, label)

        Returns:
            An instance of `tf.data.TFRecordDataset`.
        """

        return self.get_valid_recordset()

    def get_test_recordset(self, num_parallel_reads=AUTOTUNE):
        """Returns test split recordset (see `MultiSet.test_recordset`)."""
        return self.test.get_recordset(num_parallel_reads)

    @property
    def test_recordset(self):
        """Returns test split recordset.

        Recordset is an instance of `tf.data.TFRecordDataset` containing uuids and
        audio data in serialized form.

        Use `parse_serial` to extract (uuid, audio) tuples and `lookup_table` to
        translate uuids into labels.

        Example:

        table = lookup_table(open_speech.labels)

        ds = open_speech.test_recordset
        ds = ds.map(parse_serial) # serial -> (uuid, audio)
        ds = ds.map(lambda uuid, audio: (audio, table.lookup(uuid))) # (uuid, audio) -> (audio, label)

        Returns:
            An instance of `tf.data.TFRecordDataset`.
        """

        return self.get_test_recordset()
