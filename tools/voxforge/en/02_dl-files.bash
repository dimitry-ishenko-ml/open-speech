#!/bin/bash

# Download [VoxForge](http://www.voxforge.org) dataset archive files.

####################
download_url="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit"

vox_path="$HOME/tensorflow_datasets/manual/voxforge/en"
files_path="$vox_path/files"
index_path="$files_path/index"

done_path="$files_path/done"
touch "$done_path"

not_done() { cat "$index_path" "$done_path" "$done_path" | sort | uniq -u; }

echo "Downloading:"
not_done | xargs -P16 -I{} -n1 \
    sh -c "echo '{}'; wget -qcO '$files_path/{}' '$download_url/{}' && echo '{}' >> '$done_path'"
echo

failed=$(not_done)
if [[ -n "$failed" ]]; then
    echo "Download failed:"
    echo "$failed"
    echo
fi

extra=$((
    find $files_path -name '*.tgz' -printf '%P\n'
    cat "$index_path" "$index_path"
) | sort | uniq -u)
if [[ -n "$extra" ]]; then
    echo "Extra files:"
    echo "$extra"
    echo
fi
