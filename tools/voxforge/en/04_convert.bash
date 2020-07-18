#!/bin/bash

# Convert FLAC files into 16-bit 16kHz mono WAV format.

####################
vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"
extracted_path="$vox_path/extracted"

echo "Renaming:"
find "$extracted_path" -type d -name "flac" | sort | \
    xargs -P`nproc` -I{} -n1 sh -c 'basename $(dirname "{}"); mv "{}" $(dirname "{}")/wav'
echo

not_done() { find "$extracted_path" -name "*.flac" | sort; }

echo "Converting:"
not_done | sed "s/\.flac$//" | xargs -P`nproc` -I{} -n1 \
    sh -c "basename '{}'; ffmpeg -v 16 -i '{}.flac' -ar 16000 -ac 1 -c:a pcm_s16le '{}.wav' -y && rm '{}.flac'"
echo

failed=$(not_done)
if [[ -n "$failed" ]]; then
    echo "Conversion failed:"
    echo "$failed"
    echo
fi
