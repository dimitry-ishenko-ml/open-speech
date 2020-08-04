#!/usr/bin/python3

# Shard sentences and their matching WAV files into TFRecord files.

import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm.auto import tqdm
from uuid import uuid1

vox_path = "~/tensorflow_datasets/manual/voxforge/en"
vox_path = Path(vox_path).expanduser()

extracted_path = vox_path / "extracted"
json_names = [ "train.json", "valid.json", "test.json" ]

data_path = vox_path / "data"
data_path.mkdir(parents=True, exist_ok=True)

max_shard_size = 256 * (1024 ** 2) # 256MB

metadata = {
    "name": "voxforge",
    "license": "GPL-3",
    "sample_rate": 16000,
    "dtype": tf.float32,
}

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def read_audio(path):
    data = tf.io.read_file(filename=str(path))
    audio = tf.audio.decode_wav(contents=data).audio
    return audio.numpy().squeeze().tolist()

def write_shard(shard, tfrec_name):
    tfrec_path = data_path / tfrec_name
    with tf.io.TFRecordWriter(path=str(tfrec_path)) as file:
        done = {}
        for path, size, label in shard.itertuples(index=False):
            audio = read_audio(extracted_path / path)
            uuid = str(uuid1())

            example = tf.train.Example(features=tf.train.Features(feature={
                "uuid": bytes_feature([uuid.encode("utf-8")]),
                "audio": float_feature(audio),
            }))
            file.write(record=example.SerializeToString())

            done.update({ uuid: label })

    return done

def write_data(data, tfrec_templ, files_per_shard):
    groups = data.groupby(np.arange(len(data)) // files_per_shard)
    total = len(groups)

    indices, shards = zip(*groups)
    tfrec_names = [ tfrec_templ.format(index, total) for index in indices ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        dones = list(tqdm(pool.map(write_shard, shards, tfrec_names), total=total))

    labels = { uuid: label for done in dones for uuid, label in done.items() }

    return tfrec_names, labels

for json_name in json_names:
    audio_json = extracted_path / json_name
    metadata_pickle = (data_path / json_name).with_suffix(".pickle")

    print("\nReading:", audio_json)
    data = pd.read_json(audio_json, orient="table")
    print("Total examples:", len(data))

    print("Shuffling:")
    data = data.sample(frac=1)

    median_audio_size = data["size"].median()
    print("Median audio file size:", median_audio_size)

    # audio is going to be converted to float32 by tf.audio.decode_wav(),
    # which will cause their size to double (16bit -> float32)
    files_per_shard = int(max_shard_size / (2 * median_audio_size))
    print("Audio files per shard:", files_per_shard)

    print("Writing data to shards:")
    tfrec_templ = metadata_pickle.stem + "-{:04d}-of-{:04d}.tfrec"
    files, labels = write_data(data, tfrec_templ, files_per_shard)

    metadata["files"] = files
    metadata["labels"] = labels

    print("Saving metadata to:", metadata_pickle)
    with open(metadata_pickle, "wb") as file:
        pickle.dump(metadata, file)
