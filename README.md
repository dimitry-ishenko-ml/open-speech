# Open-speech

`open-speech` is a collection of popular speech datasets. Datasets included in
the collection are:

- [Mozilla Common Voice Dataset (`cv-corpus-5.1-2020-06-22`)](https://voice.mozilla.org/en/datasets)
- [VoxForge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/)
- [LibriSpeech](http://www.openslr.org/12)

Datasets have been pre-processed as follows:

- Audio files have been resampled to 16kHz.
- Audio files longer than 68kB (~21.25 seconds) have been discarded.
- Data has been sharded into ~256MB TFRecord files.

## Usage examples

`open-speech` can either be used as one large dataset or individual datasets can
be accessed and used on their own.

### Get data on each dataset:

```python
import open_speech

for dataset in open_speech.datasets:
    print("         name:", dataset.name)
    print("  sample_rate:", dataset.sample_rate)
    print("        dtype:", dataset.dtype)
    print("   # of files:", len(dataset.files))
    print("# of examples:",
        "train=", len(dataset.train_labels),
        "valid=", len(dataset.valid_labels), "test=", len(dataset.test_labels)
    )
    print()
```

### Use entire collection as one large dataset:

```python
import open_speech
import tensorflow as tf

print("  sample_rate:", open_speech.sample_rate)
print("        dtype:", open_speech.dtype)
print("   # of files:", len(open_speech.files))
print("# of examples:",
    "train=", len(open_speech.train_labels),
    "valid=", len(open_speech.valid_labels), "test=", len(open_speech.test_labels)
)
print()

# get a clean set of labels:
#    - convert unicode characters their ascii equivalents
#    - strip leading and trailing whitespace
#    - convert to lower case
#    - strip all punctuation except for the apostrophe (')
#
clean_labels = {
    uuid: open_speech.clean(label) for uuid, label in open_speech.labels.items()
}

chars = set()
for label in clean_labels.values(): chars |= set(label)
print("alphabet:", sorted(chars))

max_len = len(max(clean_labels.values(), key=len))
print("longest sentence:", max_len, "chars")
print()

def transform(dataset):
    # use open_speech.parse_serial to de-serialize examples;
    # this function will return tuples of (uuid, audio)
    dataset = dataset.map(open_speech.parse_serial)

    # use open_speech.lookup_table to look up and replace uuids
    # with corresponding labels
    table = open_speech.lookup_table(open_speech.labels)
    dataset = dataset.map(lambda uuid, audio: (audio, table.lookup(uuid)))

    # ... do something ...

    return dataset

train_dataset = transform( open_speech.train_recordset )
valid_dataset = transform( open_speech.valid_recordset )

hist = model.fit(x=train_dataset, validation_data=valid_dataset,
    # ... other parameters ...
)

test_dataset = transform( open_speech.test_recordset )

loss, metrics = model.evaluate(x=test_dataset,
    # ... other parameters ...
)
```

### Use individual dataset

```python
import open_speech
from open_speech import common_voice
import tensorflow as tf

print("name:", common_voice.name)
print("sample_rate:", common_voice.sample_rate)
print("dtype:", common_voice.dtype)

def transform(dataset):
    # use open_speech.parse_serial to de-serialize examples;
    # this function will return tuples of (uuid, audio)
    dataset = dataset.map(open_speech.parse_serial)

    # use open_speech.lookup_table to look up and replace uuids
    # with corresponding labels
    table = open_speech.lookup_table(common_voice.labels)
    dataset = dataset.map(lambda uuid, audio: (audio, table.lookup(uuid)))

    # ... do something ...

    return dataset

train_dataset = transform( common_voice.train_recordset )
valid_dataset = transform( common_voice.valid_recordset )

hist = model.fit(x=train_dataset, validation_data=valid_dataset,
    # ... other parameters ...
)
```

## Authors

* **Dimitry Ishenko** - dimitry (dot) ishenko (at) (gee) mail (dot) com

## License

This project is distributed under the GNU GPL license. See the
[LICENSE.md](LICENSE.md) file for details.
