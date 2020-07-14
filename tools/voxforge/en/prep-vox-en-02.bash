#!/bin/bash

# Extract all archives.

####################
vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"
files_path="$vox_path/files"

extracted_path="$vox_path/extracted"
mkdir -p "$extracted_path"

echo "Extracting:"
find "$files_path" -name "*.tgz" | grep -o "[^/]*$" | sort | \
    xargs -P`nproc` -I{} -n1 sh -c "echo '  {}'; tar -xzf '$files_path/{}' -C '$extracted_path'"
echo
