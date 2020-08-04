from io import BytesIO as _BytesIO
from pickle import load as _load
from unidecode import unidecode as _unidecode

import tensorflow as _tf

AUTOTUNE = _tf.data.experimental.AUTOTUNE

def get_metadata(path):
    """Reads metadata from .pickle file"""

    with _BytesIO(_tf.io.read_file(path).numpy()) as file:
        return _load(file)

def get_dataset(files, num_parallel_reads):
    """Returns dataset built from list of TFRecord files"""

    file_dataset = _tf.data.Dataset.from_tensor_slices(files)
    file_dataset = file_dataset.shuffle(buffer_size=len(files),
        reshuffle_each_iteration=True
    )
    dataset = _tf.data.TFRecordDataset(file_dataset,
        num_parallel_reads=num_parallel_reads
    )
    options = _tf.data.Options()
    options.experimental_deterministic = False

    return dataset.with_options(options)

def parse_example(example):
    """Parse serialized example and extract audio and UUID data from it"""

    data = _tf.io.parse_single_example(serialized=example, features={
        "uuid": _tf.io.FixedLenFeature([], _tf.string),
        "audio": _tf.io.VarLenFeature(_tf.float32),
    })
    uuid = data["uuid"]
    audio = _tf.sparse.to_dense(data["audio"])

    return uuid, audio

_remove_table = str.maketrans("", "", "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~")

def clean_label(label):
    """Clean label.

    - Convert to lower case.
    - Strip punctuation except for `'` (apostrophe);
    - Where possible, convert unicode characters their ASCII equivalents.
    """

    label = _unidecode(label)
    label = label.strip()
    label = label.lower()
    label = label.translate(_remove_table)
    return label
