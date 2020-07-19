#!/bin/bash

# Extract [LibriSpeech](http://www.openslr.org/12) dataset.

####################
libri_path="$HOME/tensorflow_datasets/manual/librispeech/en"
download_path="$libri_path"

extracted_path="$libri_path/extracted"
mkdir -p "$extracted_path"

paths=$(find "$download_path" -maxdepth 1 -name '*.tar.gz')
n=$(echo "$paths" | wc -l)

echo "Extracting:"
echo "$paths" | xargs -P$n -I{} -n1 tar -xvzf "{}" --keep-newer-files -C "$extracted_path"
echo
