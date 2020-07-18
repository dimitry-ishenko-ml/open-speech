#!/bin/bash

# Convert MP3 files into 16-bit 16kHz mono WAV format.

####################
cv_path="$HOME/tensorflow_datasets/manual/common-voice/en"
extracted_path="$cv_path/extracted/cv-corpus-5.1-2020-06-22/en"

tsv_names=("dev.tsv" "test.tsv" "train.tsv")
mp3_path="$extracted_path/clips"

audio_path="$cv_path/audio"
mkdir -p "$audio_path"

wanted_mp3()
{
    for tsv_name in ${tsv_names[*]}; do
        cat "$extracted_path/$tsv_name" | cut -d$'\t' -f2 | tail -n+2
    done
}

not_done()
{
    (
        wanted_mp3
        find "$mp3_path" -name '*.mp3' -printf '%P\n'
    ) | sort | uniq -d
}

echo "Converting:"
not_done | sed "s/\.mp3$//" | xargs -P`nproc` -I{} -n1 \
    sh -c "echo '{}'; ffmpeg -v 16 -i '$mp3_path/{}.mp3' -ar 16000 -ac 1 -c:a pcm_s16le '$audio_path/{}.wav' -y && rm '$mp3_path/{}.mp3'"
echo

failed=$(not_done)
if [[ -n "$failed" ]]; then
    echo "Conversion failed:"
    echo "$failed"
    echo
fi

extra=$((
    find "$audio_path" -name '*.wav' -printf '%P\n' | sed "s/\.wav$/.mp3/"
    wanted_mp3
    wanted_mp3
) | sort | uniq -u)
if [[ -n "$extra" ]]; then
    echo "Extra .wav files:"
    echo "$extra"
    echo
fi
