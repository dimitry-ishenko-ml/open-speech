#!/bin/bash

# Extract [Mozilla Common Voice](https://voice.mozilla.org) dataset.

####################
cv_path="$HOME/tensorflow_datasets/manual/common-voice/en"
download_path="$cv_path/en.tar.gz"

extracted_path="$cv_path/extracted"
mkdir -p "$extracted_path"

echo "Extracting:"
tar -xvzf "$download_path" --keep-newer-files -C "$extracted_path"
echo
