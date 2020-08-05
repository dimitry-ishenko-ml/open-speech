from .dataset import DataSet
from .multiset import MultiSet
from .util import AUTOTUNE, parse_example, lookup_table, clean_label

common_voice = DataSet(path="gs://open-speech/v5.0/common-voice/en", name="common_voice")
voxforge     = DataSet(path="gs://open-speech/v5.0/voxforge/en"    , name="voxforge"    )
librispeech  = DataSet(path="gs://open-speech/v5.0/librispeech/en" , name="librispeech" )

datasets = MultiSet([ common_voice, voxforge, librispeech ])

_attrs = [
    "sample_rate", "dtype", "files",
    "train_dataset", "test_dataset", "valid_dataset",
    "train_labels", "test_labels", "valid_labels",
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

def get_train_dataset(num_parallel_reads=AUTOTUNE):
    """Returns training TFRecordDataset of audio data and UUID of matching labels"""
    return datasets.get_train_dataset(num_parallel_reads)

def get_test_dataset(num_parallel_reads=AUTOTUNE):
    """Returns test TFRecordDataset of audio data and UUID of matching labels"""
    return datasets.get_test_dataset(num_parallel_reads)

def get_valid_dataset(num_parallel_reads=AUTOTUNE):
    """Returns validation TFRecordDataset of audio data and UUID of matching labels"""
    return datasets.get_valid_dataset(num_parallel_reads)
