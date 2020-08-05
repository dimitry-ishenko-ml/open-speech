from .datasplit import DataSplit
from .util import *

class MultiSplit:
    """Collection of data splits in multiple datasets.

    `MultiSplit` contains a number of `DataSplit` instances and acts as a single
    data split (training, validation or test) within a `MultiSet`.

    Attributes:
        name: Name of the split.
    """

    def __init__(self, splits, name):
        self._splits = splits
        self.name = name

    @property
    def sample_rate(self):
        """Returns audio sample rate."""

        sample_rate = self._splits[0].sample_rate
        assert all([ sample_rate == split.sample_rate for split in self._splits ])
        return sample_rate

    @property
    def dtype(self):
        """Returns data type of audio samples."""

        dtype = self._splits[0].dtype
        assert all([ dtype == split.dtype for split in self._splits ])
        return dtype

    @property
    def files(self):
        """Returns list of `TFRecord` files comprising the split."""

        return [ file for split in self._splits for file in split.files ]

    @property
    def labels(self):
        """Returns `dict` of labels in the form "uuid": "label"."""

        return { uuid: text for split in self._splits for uuid, text in split.labels.items() }

    def get_recordset(self, num_parallel_reads=AUTOTUNE):
        """Returns recordset for this split (see `MultiSplit.recordset`)."""

        return get_recordset(self.files, num_parallel_reads)

    @property
    def recordset(self):
        """Returns recordset for this split.

        Recordset is an instance of `tf.data.TFRecordDataset` containing uuids and
        audio data in serialized form.

        Use `parse_serial` to extract (uuid, audio) tuples and `lookup_table` to
        translate uuids into labels.

        Example:

        train = open_speech.datasets.train # instance of MultiSplit
        table = lookup_table(train.labels)

        ds = train.recordset
        ds = ds.map(parse_serial) # serial -> (uuid, audio)
        ds = ds.map(lambda uuid, audio: (audio, table.lookup(uuid))) # (uuid, audio) -> (audio, label)

        Returns:
            An instance of `tf.data.TFRecordDataset`.
        """

        return self.get_recordset()
