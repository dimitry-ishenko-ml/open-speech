#!/usr/bin/python3

# Collect sentences contained in the `{dev,test,train}.tsv` files, verify they
# have matching WAV files in the `audio_path` directory and save them in JSON
# format.

import csv
import pandas as pd

from pathlib import Path

cv_path = "~/tensorflow_datasets/manual/common-voice/en"
cv_path = Path(cv_path).expanduser()

extracted_path = cv_path / "extracted" / "cv-corpus-5.1-2020-06-22" / "en"
tsv_names = [ "dev.tsv", "test.tsv", "train.tsv" ]

audio_path = cv_path / "audio"

max_size = 680000 # 21.25s @ 16kHz, 16-bit

def read_data(tsv_path):
    rows = []
    errors = []
    with open(tsv_path) as file:
        for row in csv.DictReader(file, delimiter="\t"):

            name = Path(row["path"]).with_suffix(".wav")
            label = row["sentence"]

            path = audio_path / name
            if not path.exists():
                errors.append("Missing file: " + str(path))

            else:
                size = path.stat().st_size
                if size > max_size:
                    errors.append("Long file: " + str(path))

                else: rows.append({
                    "path" : str(name), "size" : size, "label": label,
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
