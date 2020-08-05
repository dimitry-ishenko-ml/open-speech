from io import BytesIO as _BytesIO
from pickle import load as _load
from unidecode import unidecode as _unidecode

import tensorflow as _tf

AUTOTUNE = _tf.data.experimental.AUTOTUNE

def read_data(path):
    """Reads data from .pickle file.

    Args:
        path: Path tp .pickle file to read.

    Returns:
        Python object.
    """

    with _BytesIO(_tf.io.read_file(path).numpy()) as file:
        return _load(file)

def get_recordset(files, num_parallel_reads):
    """Creates dataset from a list of `TFRecord` files.

    Args:
        files: A list of `TFRecord` files to read from.
        num_parallel_reads: Number of files to read in parallel.

    Returns:
        An instance of `tf.data.TFRecordDataset`.
    """

    file_dataset = _tf.data.Dataset.from_tensor_slices(files)
    file_dataset = file_dataset.shuffle(buffer_size=len(files),
        reshuffle_each_iteration=True
    )
    recordset = _tf.data.TFRecordDataset(file_dataset,
        num_parallel_reads=num_parallel_reads
    )
    options = _tf.data.Options()
    options.experimental_deterministic = False

    return recordset.with_options(options)

def parse_serial(example):
    """Parses serialized `Example` and extracts audio and uuid.

    Args:
        example: Single serialized instance of `Example`.

    Returns:
        A `tuple` (uuid: `tf.string`, audio: [`tf.float32`]) extracted from the
        example.
    """

    data = _tf.io.parse_single_example(serialized=example, features={
        "uuid": _tf.io.FixedLenFeature([], _tf.string),
        "audio": _tf.io.VarLenFeature(_tf.float32),
    })
    uuid = data["uuid"]
    audio = _tf.sparse.to_dense(data["audio"])

    return uuid, audio

def lookup_table(labels, default_value=""):
    """Creates translation table to replace uuids with labels.

    Usage example:

    table = lookup_table({
        "42": "foo",
        "69": "bar"
    })

    label = table.lookup(tf.constant("42"))
    print(label)

    Args:
        labels: A `dict` in the form "uuid": "label".
        default_value: The value to use for unmatched uuids.

    Returns:
        An instance of `tf.lookup.StaticHashTable`.
    """

    return _tf.lookup.StaticHashTable(
        initializer=_tf.lookup.KeyValueTensorInitializer(
            keys=list(labels.keys()), values=list(labels.values())
        ),
        default_value=default_value
    )

_remove_table = str.maketrans("", "", "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~")

def clean(label):
    """Cleans the label by doing the following:

    - Convert unicode characters their ascii equivalents.
    - Strip leading and trailing whitespace.
    - Convert to lower case.
    - Strip all punctuation except for the apostrophe (').
    """

    label = _unidecode(label)
    label = label.strip()
    label = label.lower()
    label = label.translate(_remove_table)
    return label
