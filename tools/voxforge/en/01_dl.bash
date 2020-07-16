#!/bin/bash

# Download [VoxForge](http://www.voxforge.org) dataset index and archives.

####################
download_url="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit"

vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"

files_path="$vox_path/files"
mkdir -p "$files_path"

index_path="$files_path/index"

echo "Downloading index:"
echo "$download_url"

wget -qcO- "$download_url" | sed -nr 's/^.*href="([^"]*\.tgz)".*$/\1/p' > "$index_path"
echo

max_dl=16

echo "Downloading file:"
cat "$index_path" | xargs -P$max_dl -I{} -n1 \
    sh -c "echo '  {}'; wget -qcO '$files_path/{}' '$download_url/{}'"
echo