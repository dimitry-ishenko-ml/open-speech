#!/usr/bin/python3

# Shard sentences and their matching WAV files into TFRecord files.
#
# Some files in the dataset have long silence at the end, which may cause OOM
# during training. These files will be trimmed to 15 seconds.

import numpy as np
import os
import pandas as pd
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm.auto import tqdm

cv_path = "~/tensorflow_datasets/manual/common-voice/en"
cv_path = Path(cv_path).expanduser()

audio_path = cv_path / "audio"
json_names = [ "dev.json", "test.json", "train.json" ]

data_path = cv_path / "data"
data_path.mkdir(parents=True, exist_ok=True)

sample_rate = 16000 # 16kHz
max_audio_len = 15 * sample_rate # 15s
max_audio_size = 2 * max_audio_len # 2 == 16bit

max_shard_size = 200 * (1024 ** 2) # 200MB

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
        for path, sentence, original, size in shard.itertuples(index=False):
            path = audio_path / path
            audio = read_audio(path)[ : max_audio_len] # trim long audio

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

    # we are I/O-bound, so using more than 2 workers won't be of much use;
    # in fact, it may cause slowdown due to multiple concurrent R/W
    with ThreadPoolExecutor(max_workers=2) as pool:
        dones = list(tqdm(pool.map(write_shard, shards, tfrec_names), total=total))

    return pd.concat(dones)

for json_name in json_names:
    audio_json = audio_path / json_name
    data_json = data_path / json_name

    print("\nReading:", audio_json)
    data = pd.read_json(audio_json, orient="table")
    print("Total examples:", len(data))

    data = data.sample(frac=1)

    # files are going to be trimmed inside write_shard(),
    # so we need to "clip" their size, when computing median
    median_audio_size = data["size"].clip(upper=max_audio_size).median()
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
