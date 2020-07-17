#!/bin/bash

# Convert FLAC files into 16-bit 16kHz mono WAV format.

####################
vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"
extracted_path="$vox_path/extracted"

echo "Renaming:"
find "$extracted_path" -type d -name "flac" | sort | \
    xargs -P`nproc` -I{} -n1 sh -c 'basename $(dirname "{}"); mv "{}" $(dirname "{}")/wav'
echo

echo "Converting:"
find "$extracted_path" -name "*.flac" | sed "s/\.flac$//" | sort | xargs -P`nproc` -I{} -n1 \
    sh -c 'basename "{}"; ffmpeg -v 16 -i "{}.flac" -ar 16000 -ac 1 -c:a pcm_s16le "{}.wav" -y && rm "{}.flac"'
echo

failed=$(find "$extracted_path" -name "*.flac")
if [[ -n "$failed" ]]; then
    echo "Conversion failed:"
    echo "$failed"
    echo
fi
