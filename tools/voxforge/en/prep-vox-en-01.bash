#!/bin/bash

# This script will download [VoxForge](http://www.voxforge.org) dataset index
# and archive files into `vox_path` and `files_path` directories respectively.

####################
echo
echo "Downloading index:"

vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"
mkdir -p "$vox_path"

download_url="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit"
index_path="$vox_path/index"

echo "$download_url"
wget -qcO- "$download_url" | sed -nr 's/^.*href="([^"]*\.tgz)".*$/\1/p' > "$index_path"

echo
echo "Downloading files:"

files_path="$vox_path/files"
mkdir -p "$files_path"
max_dl=16

cat "$index_path" | xargs -P$max_dl -I{} -n1 \
    sh -c "echo '  {}'; wget -qcO '$files_path/{}' '$download_url/{}'"
