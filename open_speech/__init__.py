import pandas as _pd
import tensorflow as _tf

from io import BytesIO as _BytesIO

####################
def _get_dataset(files, num_parallel_reads):
    file_dataset = _tf.data.Dataset.from_tensor_slices(files)
    file_dataset = file_dataset.shuffle(buffer_size=len(files),
        reshuffle_each_iteration=True
    )
    dataset = _tf.data.TFRecordDataset(file_dataset,
        num_parallel_reads=num_parallel_reads
    )
    options = _tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)
    return dataset

####################
class DatasetSplit:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self._meta= None

    def _get_meta(self):
        if self._meta is None and self.name is not None:
            path = self.path + "/" + self.name + ".json"
            with _BytesIO(_tf.io.read_file(path).numpy()) as file:
                meta = _pd.read_json(file, orient="table")
                index = meta.index
                self._meta = meta.set_index(_pd.MultiIndex.from_tuples(
                    [ (self.path + "/" + path, num) for path, num in index ],
                    names=index.names)
                )
        return self._meta if self._meta is not None else _pd.DataFrame()

    def size(self): return len(self._get_meta())
    def __len__(self): return self.size()

    def meta(self): return self._get_meta().copy()

    def files(self):
        paths = self._get_meta().index.unique(level=0)
        return paths.to_list()

    def dataset(self, num_parallel_reads=_tf.data.experimental.AUTOTUNE):
        return _get_dataset(self.files(), num_parallel_reads)

####################
class Dataset:
    def __init__(self, name, path):
        self.name = name
        self.path = path

        self.train = DatasetSplit(self.path, "train")
        self.test  = DatasetSplit(self.path,  "test")
        self.valid = DatasetSplit(self.path, "valid")

    def size(self): return len(self.train) + len(self.valid) + len(self.test)
    def __len__(self): return self.size()

    def meta(self): return _pd.concat([
        self.train.meta(), self.valid.meta(), self.test.meta()
    ])

    def files(self):
        return self.train.files() + self.valid.files() + self.test.files()

####################
class DatasetSplits:
    def __init__(self, splits):
        self._sps = splits

    def size(self): return sum([ len(sp) for sp in self._sps ])
    def __len__(self): return self.size()

    def meta(self): return _pd.concat([ sp.meta() for sp in self._sps ])

    def files(self):
        return [file for sp in self._sps for file in sp.files()]

    def dataset(self, num_parallel_reads=_tf.data.experimental.AUTOTUNE):
        return _get_dataset(self.files(), num_parallel_reads)

####################
class Datasets:
    def __init__(self, datasets):
        self._dss  = datasets
        self.train = DatasetSplits([ ds.train for ds in self._dss ])
        self.test  = DatasetSplits([ ds.test  for ds in self._dss ])
        self.valid = DatasetSplits([ ds.valid for ds in self._dss ])

    def size(self): return len(self.train) + len(self.valid) + len(self.test)
    def __len__(self): return self.size()

    def __iter__(self):
        for ds in self._dss: yield ds

    def meta(self): return _pd.concat([
        self.train.meta(), self.valid.meta(), self.test.meta()
    ])

    def files(self):
        return self.train.files() + self.valid.files() + self.test.files()

####################
common_voice = Dataset(name="common_voice", path="gs://open-speech/common-voice/en")
voxforge     = Dataset(name="voxforge"    , path="gs://open-speech/voxforge/en"    )
librispeech  = Dataset(name="librispeech" , path="gs://open-speech/librispeech/en" )

datasets = Datasets([ common_voice, voxforge, librispeech ])

train = datasets.train
test  = datasets.test
valid = datasets.valid

def size(): return len(datasets)
def meta(): return datasets.meta()
def files(): return datasets.files()

####################
def parse_example(example):
    data = _tf.io.parse_single_example(serialized=example, features={
        "audio": _tf.io.VarLenFeature(_tf.float32),
        "sentence": _tf.io.FixedLenFeature([], _tf.string),
        "original": _tf.io.FixedLenFeature([], _tf.string),
    })
    audio = _tf.sparse.to_dense(data["audio"])
    sentence = data["sentence"]
    original = data["original"]
    return audio, sentence, original
