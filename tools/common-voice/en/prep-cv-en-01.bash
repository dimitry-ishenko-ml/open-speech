#!/bin/bash

# This script will download and extract [Mozilla Common
# Voice](https://voice.mozilla.org) dataset, and will convert MP3 files listed
# in `dev.tsv`, `test.tsv` and `train.tsv` into WAV format, so they can be read
# by `tf.audio.decode_wav()`.
#
# You have to manually get the download URL from
# https://voice.mozilla.org/en/datasets and enter it into the `download_url`
# variable below.
#
# The dataset will be downloaded into `cv_path` directory and extracted into
# `extracted_path` directory, which will have the following structure:
# ```
# clips/
# dev.tsv
# invalidated.tsv
# other.tsv
# test.tsv
# train.tsv
# validated.tsv
# ```
# The `clips` directory will contain all the MP3 files.
#
# Next, the appropriate MP3 files (specified in `tsv_names`) will be converted
# to 16-bit mono WAV format using ffmpeg and placed into the `audio_path`
# directory.

####################
echo
echo "Downloading:"

cv_path="$HOME/tensorflow_datasets/manual/common-voice/en"
mkdir -p "$cv_path"

download_url="..."
download_path="$cv_path/en.tar.gz"

wget -cO "$download_path" "$download_url"

####################
echo
echo "Extracting:"

extract_path="$cv_path/extracted"

if [[ ! -e "$extract_path" ]]; then
    mkdir -p "$extract_path"
    tar -xvzf "$download_path" -C "$extract_path"
else
    echo "Path $extract_path exists"
fi

####################
echo
echo "Converting:"

sample_rate=16000
channels=1
format=pcm_s16le

tsv_names=( "dev.tsv" "test.tsv" "train.tsv" )
mp3_path="$extract_path/clips"

audio_path="$cv_path/audio"

if [[ ! -e "$audio_path" ]]; then
    mkdir -p "$audio_path"

    for tsv_name in ${tsv_names[*]}; do
        tsv_path="$extract_path/$tsv_name"
        echo "Processing: $tsv_path"

        cat "$tsv_path" | cut -d$'\t' -f2 | cut -d'.' -f1 | tail -n+2 | xargs -P`nproc` -I{} -n1 \
            ffmpeg -hide_banner -i "$mp3_path/{}.mp3" -ar $sample_rate -ac $channels -c:a $format "$audio_path/{}.wav"
    done
else
    echo "Path $audio_path exists"
fi
