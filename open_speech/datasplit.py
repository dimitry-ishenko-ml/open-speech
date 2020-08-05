from .util import *

class DataSplit:
    """Represents data split (training, validation or test) in a dataset.

    Data split consists of a metadata file (stored as a .pickle) containing
    labels and other parameters, and a number of `TFRecord` files with audio
    data.

    Attributes:
        path: Path to data split files.
        name: Name of the split.
    """

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self._metadata = None

    def _get_metadata(self):
        """Reads and caches the metadata file."""

        if self._metadata is None:
            self._metadata = read_data(self.path + "/" + self.name + ".pickle")
            self._metadata["files"] = [
                self.path + "/" + file for file in self._metadata["files"]
            ]
        return self._metadata

    @property
    def sample_rate(self):
        """Returns audio sample rate."""

        return self._get_metadata()["sample_rate"]

    @property
    def dtype(self):
        """Returns data type of audio samples."""

        return self._get_metadata()["dtype"]

    @property
    def files(self):
        """Returns list of `TFRecord` files comprising the split."""

        return self._get_metadata()["files"]

    @property
    def labels(self):
        """Returns `dict` of labels in the form "uuid": "label"."""

        return self._get_metadata()["labels"]

    def get_recordset(self, num_parallel_reads=AUTOTUNE):
        """Returns recordset for this split (see `DataSplit.recordset`)."""

        return get_recordset(self.files, num_parallel_reads)

    @property
    def recordset(self):
        """Returns recordset for this split.

        Recordset is an instance of `tf.data.TFRecordDataset` containing uuids and
        audio data in serialized form.

        Use `parse_serial` to extract (uuid, audio) tuples and `lookup_table` to
        translate uuids into labels.

        Example:

        train = open_speech.common_voice.train # instance of DataSplit
        table = lookup_table(train.labels)

        ds = train.recordset
        ds = ds.map(parse_serial) # serial -> (uuid, audio)
        ds = ds.map(lambda uuid, audio: (audio, table.lookup(uuid))) # (uuid, audio) -> (audio, label)

        Returns:
            An instance of `tf.data.TFRecordDataset`.
        """

        return self.get_dataset()
