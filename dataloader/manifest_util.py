# Manifest Utils
import os

from tqdm.auto import tqdm
import json
from collections import defaultdict


def read_manifest(path):
    manifest = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    with open(filepath, 'w', encoding='utf-8') as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum, ensure_ascii=False)
            f.write(f"{datum}\n")
    print(f"Finished writing manifest: {filepath}")
    return filepath


def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

