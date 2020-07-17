#!/bin/bash

# Download [VoxForge](http://www.voxforge.org) dataset index.

####################
download_url="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit"

vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"

files_path="$vox_path/files"
mkdir -p "$files_path"

index_path="$files_path/index"

echo "Downloading:"
wget -cO- "$download_url" | sed -nr 's/^.*href="([^"]*\.tgz)".*$/\1/p' | sort > "$index_path"
echo

