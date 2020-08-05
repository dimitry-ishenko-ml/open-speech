"""`open-speech` is a collection of popular speech datasets.

Datasets included in the collection are:
    - Mozilla Common Voice Dataset (`cv-corpus-5.1-2020-06-22`)
    - VoxForge
    - LibriSpeech

Datasets have been pre-processed as follows:
    - Audio files have been resampled to 16kHz.
    - Audio files longer than 68kB (~21.25 seconds) have been discarded.
    - Data has been sharded into ~256MB TFRecord files.

`open-speech` can either be used as one large dataset or individual datasets can
be acccessed and used on their own, eg:

    os = open_speech.datasets # one large dataset
    cv = open_speech.common_voice
    ls = open_speech.librispeech
    vf = open_speech.voxforge

Each dataset has the following properties:

    sample_rate:
        Audio sample rate (eg, 160000).

    dtype:
        Data type of audio samples (ef, `tf.float32`).

    files:
        List of all `TFRecord` files comprising the dataset.

    train_labels, valid_labels, test_labels:
        Dictionary of training, validation and test split labels in the form
        "uuid": "label".

    labels:
        Dictionary of all labels in the dataset.

    train_recordset, valid_recordset, test_recordset:
        Training, validation and test split recordsets. Each recordset is an
        instance of `tf.data.TFRecordDataset` containing uuids and audio data in
        serialized form.

        Use `open_speech.parse_serial` to extract (uuid, audio) tuples and
        `open_speech.lookup_table` to translate uuids into labels.

        Example:

        cv = open_speech.common_voice
        table = open_speech.lookup_table(cv.labels)

        dataset = cv.train_recordset
        dataset = dataset.map(open_speech.parse_serial) # serial -> (uuid, audio)
        dataset = dataset.map(lambda uuid, audio: (audio, table.lookup(uuid))) # (uuid, audio) -> (audio, label)

The above properties can also be applied directly to the `open_speech` module,
in which case they are passed through to `open_speech.datasets`. In other words:

    table = open_speech.lookup_table(open_speech.labels) # == open_speech.datasets.labels

    dataset = open_speech.train_recordset # == open_speech.datasets.train_recordset
    dataset = dataset.map(open_speech.parse_serial)
    dataset = dataset.map(lambda uuid, audio: (audio, table.lookup(uuid)))
"""

from .dataset import DataSet
from .multiset import MultiSet
from .util import AUTOTUNE, parse_serial, lookup_table, clean

common_voice = DataSet(path="gs://open-speech/v5.0/common-voice/en", name="common_voice")
voxforge     = DataSet(path="gs://open-speech/v5.0/voxforge/en"    , name="voxforge"    )
librispeech  = DataSet(path="gs://open-speech/v5.0/librispeech/en" , name="librispeech" )

datasets = MultiSet([ common_voice, voxforge, librispeech ])

_attrs = [
    "sample_rate", "dtype", "files",
    "train_recordset", "valid_recordset", "test_recordset",
    "train_labels", "valid_labels", "test_labels",
    "labels"
]

def __dir__():
    return sorted(list(globals().keys()) + _attrs)

def __getattr__(name):
    if name in _attrs:
        return getattr(datasets, name)

    else: raise AttributeError(
        "module '{}' has no attribute '{}'".format(__name__, name)
    )

def get_train_recordset(num_parallel_reads=AUTOTUNE):
    """Returns training split recordset."""

    return datasets.get_train_recordset(num_parallel_reads)

def get_valid_recordset(num_parallel_reads=AUTOTUNE):
    """Returns validation split recordset."""

    return datasets.get_valid_recordset(num_parallel_reads)

def get_test_recordset(num_parallel_reads=AUTOTUNE):
    """Returns test split recordset."""

    return datasets.get_test_recordset(num_parallel_reads)
