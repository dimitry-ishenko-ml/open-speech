#!/bin/bash

# Convert FLAC files into 16-bit 16kHz mono WAV format.

####################
vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"
extracted_path="$vox_path/extracted"

echo "Converting:"
find "$extracted_path" -name "*.flac" | sed "s/\.flac$//" | sort | \
    xargs -P`nproc` -I{} -n1 ffmpeg -hide_banner -i "{}.flac" -ar 16000 -ac 1 -c:a pcm_s16le "{}.wav"
echo
