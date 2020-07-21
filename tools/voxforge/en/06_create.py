#!/usr/bin/python3

# Shard sentences and their matching WAV files into TFRecord files.

import numpy as np
import os
import pandas as pd
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm.auto import tqdm

vox_path = "~/tensorflow_datasets/manual/voxforge/en"
vox_path = Path(vox_path).expanduser()

extracted_path = vox_path / "extracted"
json_names = [ "train.json", "valid.json", "test.json" ]

data_path = vox_path / "data"
data_path.mkdir(parents=True, exist_ok=True)

max_shard_size = 256 * (1024 ** 2) # 256MB

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
        done = []
        for path, size, sentence, original in shard.itertuples(index=False):
            audio = read_audio(extracted_path / path)

            example = tf.train.Example(features=tf.train.Features(feature={
                "audio"   : float_feature(audio),
                "sentence": bytes_feature([sentence.encode("utf-8")]),
                "original": bytes_feature([original.encode("utf-8")])
            }))
            file.write(record=example.SerializeToString())

            done.append({ "sentence": sentence, "original": original })

    return pd.DataFrame(done, index=pd.MultiIndex.from_product(
        [[tfrec_name], range(len(done))], names=["path", "#"]
    ))

def write_data(data, tfrec_templ, files_per_shard):
    groups = data.groupby(np.arange(len(data)) // files_per_shard)
    total = len(groups)

    indices, shards = zip(*groups)
    tfrec_names = [ tfrec_templ.format(index, total) for index in indices ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        dones = list(tqdm(pool.map(write_shard, shards, tfrec_names), total=total))

    return pd.concat(dones)

for json_name in json_names:
    audio_json = extracted_path / json_name
    data_json = data_path / json_name

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
    tfrec_templ = data_json.stem + "-{:04d}-of-{:04d}.tfrec"
    dones = write_data(data, tfrec_templ, files_per_shard)

    print("Saving metadata to:", data_json)
    dones.to_json(data_json, orient="table")
