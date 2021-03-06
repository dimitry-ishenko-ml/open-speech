#!/usr/bin/python3

# Collect sentences found in extracted archives, split them into training,
# validation and test sets, verify they have matching WAV files in the
# `audio_path` directory and save in JSON format.

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

vox_path = "~/tensorflow_datasets/manual/voxforge/en"
vox_path = Path(vox_path).expanduser()

extracted_path = vox_path / "extracted"

data_names = [ "train.json", "valid.json", "test.json" ]
data_probs = [ .8, .1, .1 ]

arch_paths = extracted_path.glob("*")
prompt_names = [
    "prompts-original",
    "prompt.txt", "prompts.txt", "cc.prompts", "therainbowpassage.prompt",
    "Transcriptions.txt", "a13.text", "rp.text",
    "PROMPTS",
]

max_size = 680000 # 21.25s @ 16kHz, 16-bit

def read_data(arch_path):
    audio_path = arch_path / "wav"
    if not audio_path.exists():
        return [], ["Missing audio dir: " + str(arch_path)]

    for name in prompt_names:
        prompt_path = arch_path / "etc" / name
        if prompt_path.exists(): break

    if not prompt_path.exists():
        return [], ["Missing prompts: " + str(arch_path)]

    rows = []
    errors = []
    with open(prompt_path) as file:
        for row in file.read().splitlines():

            try: name, label = row.split(" ", 1)
            except: continue

            name = Path(name).name + ".wav"
            name = audio_path.relative_to(extracted_path) / name
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

rows_all = []
errors_all = []
print("Processing:")
for arch_path in tqdm(list(arch_paths)):
    if arch_path.is_dir():
        rows, errors = read_data(arch_path)
        rows_all += rows
        errors_all += errors

data_all = pd.DataFrame(rows_all)
print("Total examples:", len(data_all))

if len(errors_all):
    print("\nErrors:")
    for error in errors_all: print(error)

indices = np.random.choice(len(data_probs), size=len(data_all), p=data_probs)

print()
for n in range(len(data_names)):
    json_path = extracted_path / data_names[n]
    data = data_all[indices == n]

    print("Saving:", json_path)
    data.to_json(json_path, orient="table")
