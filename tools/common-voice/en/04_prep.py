#!/usr/bin/python3

# Collect and clean up sentences contained in the `{dev,test,train}.tsv` files
# and save them in JSON format.
#
# The clean-up consists of:
# - where appropriate, convert unicode characters to their ASCII equivalents;
# - convert all characters to lower case;
# - remove all punctuation except for `'` (apostrophe);
# - verify matching WAV file exists in the `audio_path` directory.

import csv
import pandas as pd

from pathlib import Path
from unidecode import unidecode

cv_path = "~/tensorflow_datasets/manual/common-voice/en"
cv_path = Path(cv_path).expanduser()

extracted_path = cv_path / "extracted" / "cv-corpus-5.1-2020-06-22" / "en"
tsv_names = [ "dev.tsv", "test.tsv", "train.tsv" ]

audio_path = cv_path / "audio"

remove_table = str.maketrans("", "", "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~")

max_size = 680000 # 21.25s @ 16kHz, 16-bit

def clean(sentence):
    sentence = unidecode(sentence)
    sentence = sentence.strip()
    sentence = sentence.lower()
    sentence = sentence.translate(remove_table)
    return sentence

def read_data(tsv_path):
    rows = []
    errors = []
    with open(tsv_path) as file:
        for row in csv.DictReader(file, delimiter="\t"):

            name = Path(row["path"]).with_suffix(".wav")
            original = row["sentence"]
            sentence = clean(original)

            path = audio_path / name
            if not path.exists():
                errors.append("Missing file: " + str(path))

            else:
                size = path.stat().st_size
                if size > max_size:
                    errors.append("Long file: " + str(path))

                else: rows.append({
                    "path"    : str(name),
                    "size"    : size,
                    "sentence": sentence,
                    "original": original,
                })

    return rows, errors

for tsv_name in tsv_names:
    tsv_path = extracted_path / tsv_name
    json_path = (audio_path / tsv_name).with_suffix(".json")

    print("\nProccessing:", tsv_path)
    rows, errors = read_data(tsv_path)

    data = pd.DataFrame(rows)
    print("Total examples:", len(data))

    print("Saving to:", json_path)
    data.to_json(json_path, orient="table")

    if len(errors):
        print("\nErrors:")
        for error in errors: print(error)
