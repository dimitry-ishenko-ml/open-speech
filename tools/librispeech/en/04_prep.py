#!/usr/bin/python3

# Collect sentences found in extracted dev, test and train archives, verify they
# have matching WAV files in the `audio_path` directory and save them in JSON
# format.

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

libri_path = "~/tensorflow_datasets/manual/librispeech/en"
libri_path = Path(libri_path).expanduser()

extracted_path = libri_path / "extracted" / "LibriSpeech"
parts = {
    "train.json": [ "train-clean-100", "train-clean-360" ],
    "valid.json": [  "dev-clean" ],
    "test.json" : [ "test-clean" ],
}

max_size = 680000 # 21.25s @ 16kHz, 16-bit

def read_data(trans_path):
    audio_path = trans_path.parent.relative_to(extracted_path)

    rows = []
    errors = []
    with open(trans_path) as file:
        for row in file.read().splitlines():

            try: name, label = row.split(" ", 1)
            except: continue

            name = audio_path / (name + ".wav")
            path = extracted_path / name
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

for part_json, part_names in parts.items():
    rows_all = []
    errors_all = []

    print()
    for part_name in part_names:
        print("Processing:", part_name)

        trans_paths = list((extracted_path / part_name).glob("**/*.trans.txt"))
        for trans_path in tqdm(trans_paths):
            rows, errors = read_data(trans_path)
            rows_all += rows
            errors_all += errors

    data_all = pd.DataFrame(rows_all)
    print("Total examples:", len(data_all))

    if len(errors_all):
        print("\nErrors:")
        for error in errors_all: print(error)

    json_path = extracted_path / part_json
    print("Saving:", json_path)
    data_all.to_json(json_path, orient="table")
