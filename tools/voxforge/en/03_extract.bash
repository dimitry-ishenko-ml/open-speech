#!/bin/bash

# Extract archive files.

####################
vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"
files_path="$vox_path/files"
index_path="$files_path/index"

extracted_path="$vox_path/extracted"
mkdir -p "$extracted_path"

done_path="$extracted_path/done"
touch "$done_path"

not_done() { cat "$index_path" "$done_path" "$done_path" | sort | uniq -u; }

echo "Extracting:"
not_done | xargs -P`nproc` -I{} -n1 \
    sh -c "echo '{}'; tar -xzf '$files_path/{}' -C '$extracted_path' && echo '{}' >> '$done_path'"
echo

failed=$(not_done)
if [[ -n "$failed" ]]; then
    echo "Extraction failed:"
    echo "$failed"
    echo
fi
