# Open Speech Datasets

Open Speech Datasets is a collection of popular speech datasets.

Datasets have been pre-processed as follows:

- Audio files have been resampled to 16kHz and trimmed to 15 seconds.
- Sentences have been converted to lower case and stripped of punctuation with
  the exception of `'` (apostrophe).
- Where possible, unicode characters have been converted to their ASCII
  equivalents.
- Original sentences have also been preserved.
- Data has been sharded into ~200MB TFRecord files.

Each individual dataset is located in its own directory consisting of a number
of TFRecord files and one of more JSON files containing metadata.

Following are some examples of how to use this collection.

Use the entire collection as one dataset:

```python
import open_speech
import tensorflow as tf

# these may take a little while to execute as they read metadata
# for each dataset in the collection;
# the metadata is cached and will be used for subsequent calls
print("Training set size:", len(open_speech.train))
print("Validation set size:", len(open_speech.valid))
print("Test dataset size:", len(open_speech.test))

# will use cached metadata from above
metadata = open_speech.meta()

# show all chars used in processed sentences
chars = set()
for sent in metadata["sentence"]: chars |= set(sent)
print("Alphabet:", sorted(chars))

max_len = metadata["sentence"].apply(len).max()
print("Longest sentence:", max_len, "chars")

def transform(dataset):
    # use open_speech.parse_example() to de-serialize examples;
    # this function will return tuples of (audio, sentence, original)
    dataset = dataset.map(open_speech.parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # ...
    dataset = dataset.prefetch(2)
    return dataset

train_dataset = transform( open_speech.train.dataset() )
valid_dataset = transform( open_speech.valid.dataset() )

hist = model.fit(x=train_dataset,
    validation_data=valid_dataset,
    # ...
)

test_dataset = transform( open_speech.test.dataset() )

loss, metrics = model.evaluate(x=test_dataset,
    # ...
)
```

List datasets included in the collection:
```python
import open_speech
print("Available datasets:")
for dataset in open_speech.datasets:
    print(dataset.name)
```

Use an individual dataset:
```python
from open_speech import common_voice_en
import tensorflow as tf

print("Dataset:", common_voice_en.name)

def transform(dataset):
    # use open_speech.parse_example() to de-serialize examples;
    # this function will return tuples of (audio, sentence, original)
    dataset = dataset.map(open_speech.parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # ...
    dataset = dataset.prefetch(2)
    return dataset

train_dataset = transform( common_voice_en.train.dataset() )
valid_dataset = transform( common_voice_en.valid.dataset() )

hist = model.fit(x=train_dataset,
    validation_data=valid_dataset,
    # ...
)
```

Following datasets are currently included in the collection:

- [Mozilla Common Voice Dataset (`cv-corpus-5.1-2020-06-22`)](https://voice.mozilla.org/en/datasets)

- [VoxForge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/)

## Authors

* **Dimitry Ishenko** - dimitry (dot) ishenko (at) (gee) mail (dot) com

## License

This project is distributed under the GNU GPL license. See the
[LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
