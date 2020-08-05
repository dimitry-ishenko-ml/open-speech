# Open-speech

`open-speech` is a collection of popular speech datasets:

- [Mozilla Common Voice Dataset (`cv-corpus-5.1-2020-06-22`)](https://voice.mozilla.org/en/datasets)

- [VoxForge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/)

- [LibriSpeech](http://www.openslr.org/12)

Datasets have been pre-processed as follows:

- Audio files have been resampled to 16kHz.
- Audio files longer than 68kB (~21.25 seconds) have been discarded.
- Data has been sharded into ~256MB TFRecord files.

## Usage examples

### Use the entire collection as one large dataset:

```python
import open_speech
import tensorflow as tf

print("datasets:")
for dataset in open_speech.datasets:
    print(dataset.name)
print()

# these may be a little slow to execute as they read metadata
# for each dataset in the collection;
# the metadata is cached and will be reused for subsequent calls
print("training set size:", len(open_speech.train_labels))
print("validation set size:", len(open_speech.valid_labels))
print("test set size:", len(open_speech.test_labels))
print()

# get a clean set of labels:
# - convert to lower case
# - strip punctuation except for apostrophe (')
# - where possible, convert unicode characters their ASCII equivalents
clean_labels = {
    uuid: open_speech.clean_label(label) for uuid, label in open_speech.labels.items()
}

# show all chars used in clean labels
chars = set()
for sent in clean_labels.values(): chars |= set(sent)
print("alphabet:", sorted(chars))

max_len = len(max(labels.values(), key=len))
print("longest sentence:", max_len, "chars")
print()

def transform(dataset):
    # use open_speech.parse_example to de-serialize examples;
    # this function will return tuples of (uuid, audio)
    dataset = dataset.map(open_speech.parse_example)

    # use open_speech.lookup_table to look up and replace uuid
    # with its label in the labels dictionary
    dataset = dataset.map(open_speech.lookup_table(labels))

    # ... do something ...

    return dataset

train_dataset = transform( open_speech.train_dataset )
valid_dataset = transform( open_speech.valid_dataset )

hist = model.fit(x=train_dataset, validation_data=valid_dataset,
    # ... other parameters ...
)

test_dataset = transform( open_speech.test_dataset )

loss, metrics = model.evaluate(x=test_dataset,
    # ... other parameters ...
)
```

### Use individual dataset

```python
import open_speech
from open_speech import common_voice
import tensorflow as tf

print("dataset:", common_voice.name)

def transform(dataset):
    # use open_speech.parse_example to de-serialize examples;
    # this function will return tuples of (uuid, audio)
    dataset = dataset.map(open_speech.parse_example)

    # use open_speech.lookup_table to look up and replace uuid
    # with its label in the labels dictionary
    dataset = dataset.map(open_speech.lookup_table(common_voice.labels))

    # ... do something ...

    return dataset

train_dataset = transform( common_voice.train_dataset )
valid_dataset = transform( common_voice.valid_dataset )

hist = model.fit(x=train_dataset, validation_data=valid_dataset,
    # ... other parameters ...
)
```

## Authors

* **Dimitry Ishenko** - dimitry (dot) ishenko (at) (gee) mail (dot) com

## License

This project is distributed under the GNU GPL license. See the
[LICENSE.md](LICENSE.md) file for details.
