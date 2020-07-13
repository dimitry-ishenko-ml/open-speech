#!/bin/bash

# This script will extract all archive files located in `files_path` into the
# `extracted_path` directory.

####################
echo
echo "Extracting files:"

vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"
files_path="$vox_path/files"

extracted_path="$vox_path/extracted"
mkdir -p "$extracted_path"

find "$files_path" -name "*.tgz" | sed -nr "s|^.*/(.*)\.tgz$|\1|p" | sort | \
    xargs -P`nproc` -I{} -n1 sh -c "echo '  {}'; tar -xzf '$files_path/{}.tgz' -C '$extracted_path'"