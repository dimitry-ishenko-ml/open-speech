#!/usr/bin/python3

# This script will scan through and clean up all sentences contained in
# `dev.tsv`, `test.tsv` and `train.tsv` files located in the `extracted_path`
# directory.
#
# The clean-up consists of:
# - where appropriate, convert unicode characters to their ASCII equivalents;
# - convert all characters to lower case;
# - remove all punctuation except for `'` (apostrophe);
# - verify matching WAV file exists in the `audio_path` directory.
#
# Processed data will be save in JSON format (with "table" orient) in
# corresponding `dev.json`, `test.json` and `train.json` files in the
# `audio_path` directory.

import csv
import pandas as pd

from pathlib import Path
from unidecode import unidecode

cv_path = "~/tensorflow_datasets/manual/common-voice/en"
cv_path = Path(cv_path).expanduser()

extracted_path = cv_path / "extracted"
tsv_names = [ "dev.tsv", "test.tsv", "train.tsv" ]

audio_path = cv_path / "audio"

remove_table = str.maketrans("", "", "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~")

def clean(sentence):
    sentence = unidecode(sentence)
    sentence = sentence.lower()
    sentence = sentence.translate(remove_table)
    return sentence

def read_data(tsv_path):
    rows = []
    with open(tsv_path) as file:
        for row in csv.DictReader(file, delimiter="\t"):

            name = Path(row["path"]).with_suffix(".wav")
            original = row["sentence"]
            sentence = clean(original)

            path = audio_path / name
            if not path.exists():
                print("Missing file:", path)
            
            else: rows.append({
                "path"    : str(name),
                "sentence": sentence,
                "original": original,
                "size"    : path.stat().st_size
            })
    return pd.DataFrame(rows)

for tsv_name in tsv_names:
    tsv_path = extracted_path / tsv_name
    json_path = (audio_path / tsv_name).with_suffix(".json")
    
    print("\nProccessing:", tsv_path)
    data = read_data(tsv_path)
    print("Total examples:", len(data))
    
    print("Saving to:", json_path)
    data.to_json(json_path, orient="table")
