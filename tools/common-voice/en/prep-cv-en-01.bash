#!/bin/bash

# Download and extract [Mozilla Common Voice](https://voice.mozilla.org) dataset
# and convert MP3 files into 16-bit 16kHz mono WAV format.
#
# You have to manually get the download URL from
# https://voice.mozilla.org/en/datasets and enter it into the `download_url`
# variable below.

####################
cv_path="$HOME/tensorflow_datasets/manual/common-voice/en"
mkdir -p "$cv_path"

download_url=""
download_path="$cv_path/en.tar.gz"

echo "Downloading:"
wget -cO "$download_path" "$download_url"
echo

####################
extracted_path="$cv_path/extracted"

if [[ ! -e "$extracted_path" ]]; then
    echo "Extracting:"
    mkdir -p "$extracted_path"

    tar -xvzf "$download_path" -C "$extracted_path"
    echo
else
    echo "Path $extracted_path exists"
fi

####################
tsv_names=( "dev.tsv" "test.tsv" "train.tsv" )
mp3_path="$extracted_path/clips"

audio_path="$cv_path/audio"

if [[ ! -e "$audio_path" ]]; then
    echo "Converting:"
    mkdir -p "$audio_path"

    for tsv_name in ${tsv_names[*]}; do
        tsv_path="$extracted_path/$tsv_name"
        echo "Processing: $tsv_path"

        cat "$tsv_path" | cut -d$'\t' -f2 | cut -d'.' -f1 | tail -n+2 | xargs -P`nproc` -I{} -n1 \
            ffmpeg -hide_banner -i "$mp3_path/{}.mp3" -ar 16000 -ac 1 -c:a pcm_s16le "$audio_path/{}.wav"
    done
    echo
else
    echo "Path $audio_path exists"
fi
