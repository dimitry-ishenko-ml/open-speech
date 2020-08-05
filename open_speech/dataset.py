from .datasplit import DataSplit
from .util import *

class DataSet:
    """Represents single dataset.

    Dataset consists of training, validation and test data splits represented
    by instances of `DataSplit`, each one with its own metadata and `TFRecord`
    files.

    Data splits can be accessed using `train`, `valid` and `test` attributes,
    and their corresponding recordsets with `train_recordset`, `valid_recordset`
    and `test_recordset` properties respectively.

    Attributes:
        path: Path to dataset files.
        name: Name of the dataset.

        train: Training data split.
        valid: Validation data split.
        test: Test data split.
    """

    def __init__(self, path, name):
        self.path = path
        self.name = name

        self.train = DataSplit(path, "train")
        self.test  = DataSplit(path, "test" )
        self.valid = DataSplit(path, "valid")

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
        """Returns list of all `TFRecord` files comprising the dataset."""

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
        """Returns `dict` of all dataset labels in the form "uuid": "label"."""

        return { **self.train_labels, **self.test_labels, **self.valid_labels }

    def get_train_recordset(self, num_parallel_reads=AUTOTUNE):
        """Returns training split recordset (see `DataSet.train_recordset`)."""

        return self.train.get_recordset(num_parallel_reads)

    @property
    def train_recordset(self):
        """Returns training split recordset.

        Recordset is an instance of `tf.data.TFRecordDataset` containing uuids and
        audio data in serialized form.

        Use `parse_serial` to extract (uuid, audio) tuples and `lookup_table` to
        translate uuids into labels.

        Example:

        cv = open_speech.common_voice # instance of DataSet
        table = lookup_table(cv.labels)

        ds = cv.train_recordset
        ds = ds.map(parse_serial) # serial -> (uuid, audio)
        ds = ds.map(lambda uuid, audio: (audio, table.lookup(uuid))) # (uuid, audio) -> (audio, label)

        Returns:
            An instance of `tf.data.TFRecordDataset`.
        """

        return self.get_train_recordset()

    def get_valid_recordset(self, num_parallel_reads=AUTOTUNE):
        """Returns validation split recordset (see `DataSet.valid_recordset`)."""

        return self.valid.get_recordset(num_parallel_reads)

    @property
    def valid_recordset(self):
        """Returns validation split recordset.

        Recordset is an instance of `tf.data.TFRecordDataset` containing uuids and
        audio data in serialized form.

        Use `parse_serial` to extract (uuid, audio) tuples and `lookup_table` to
        translate uuids into labels.

        Example:

        cv = open_speech.common_voice # instance of DataSet
        table = lookup_table(cv.labels)

        ds = cv.valid_recordset
        ds = ds.map(parse_serial) # serial -> (uuid, audio)
        ds = ds.map(lambda uuid, audio: (audio, table.lookup(uuid))) # (uuid, audio) -> (audio, label)

        Returns:
            An instance of `tf.data.TFRecordDataset`.
        """

        return self.get_valid_recordset()

    def get_test_recordset(self, num_parallel_reads=AUTOTUNE):
        """Returns test split recordset (see `DataSet.test_recordset`)."""

        return self.test.get_recordset(num_parallel_reads)

    @property
    def test_recordset(self):
        """Returns test split recordset.

        Recordset is an instance of `tf.data.TFRecordDataset` containing uuids and
        audio data in serialized form.

        Use `parse_serial` to extract (uuid, audio) tuples and `lookup_table` to
        translate uuids into labels.

        Example:

        cv = open_speech.common_voice # instance of DataSet
        table = lookup_table(cv.labels)

        ds = cv.test_recordset
        ds = ds.map(parse_serial) # serial -> (uuid, audio)
        ds = ds.map(lambda uuid, audio: (audio, table.lookup(uuid))) # (uuid, audio) -> (audio, label)

        Returns:
            An instance of `tf.data.TFRecordDataset`.
        """

        return self.get_test_recordset()
