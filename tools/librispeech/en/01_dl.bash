#!/bin/bash

# Download [LibriSpeech](http://www.openslr.org/12) dataset.

####################
libri_path="$HOME/tensorflow_datasets/manual/librispeech/en"
mkdir -p "$libri_path"

download_urls=(
    "http://www.openslr.org/resources/12/dev-clean.tar.gz"
    "http://www.openslr.org/resources/12/test-clean.tar.gz"
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz"
)
download_path="$libri_path"

echo "Downloading:"
for download_url in ${download_urls[@]}; do
    name=$(basename "$download_url")
    wget -cO "$download_path/$name" "$download_url"
done
echo
