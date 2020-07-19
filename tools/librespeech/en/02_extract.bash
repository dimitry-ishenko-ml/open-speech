#!/bin/bash

# Extract [LibreSpeech](http://www.openslr.org/12) dataset.

####################
libre_path="$HOME/tensorflow_datasets/manual/librespeech/en"
download_path="$libre_path"

extracted_path="$libre_path/extracted"
mkdir -p "$extracted_path"

paths=$(find "$download_path" -maxdepth 1 -name '*.tar.gz')
n=$(echo "$paths" | wc -l)

echo "Extracting:"
echo "$paths" | xargs -P$n -I{} -n1 tar -xvzf "{}" --keep-newer-files -C "$extracted_path"
echo
