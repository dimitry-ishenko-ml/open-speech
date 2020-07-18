#!/bin/bash

# Download [Mozilla Common Voice](https://voice.mozilla.org) dataset.
#
# You have to manually get the download URL from
# https://voice.mozilla.org/en/datasets and enter it into the `download_url`
# variable below.

####################
cv_path="$HOME/tensorflow_datasets/manual/common-voice/en"
mkdir -p "$cv_path"

download_url="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz"
download_path="$cv_path/en.tar.gz"

echo "Downloading:"
wget -cO "$download_path" "$download_url"
echo
