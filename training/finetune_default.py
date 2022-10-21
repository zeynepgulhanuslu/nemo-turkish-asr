import argparse
import os

# Manifest Utils
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
    with open(filepath, 'w') as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--manifest', type=str, required=True, help='Manifest directory')

    args = parser.parse_args()

    manifest_dir = args.manifest

    train_manifest_file = os.path.join(manifest_dir, 'nemo-train-manifest.json')
    test_manifest_file = os.path.join(manifest_dir, 'nemo-test-manifest.json')
    dev_manifest_file = os.path.join(manifest_dir, 'nemo-dev-manifest.json')

    train_manifest_data = read_manifest(train_manifest_file)
    test_manifest_data = read_manifest(test_manifest_file)
    dev_manifest_data = read_manifest(dev_manifest_file)

    train_text = [data['text'] for data in train_manifest_data]
    dev_text = [data['text'] for data in dev_manifest_data]
    test_text = [data['text'] for data in test_manifest_data]

    train_charset = get_charset(train_manifest_data)
    dev_charset = get_charset(dev_manifest_data)
    test_charset = get_charset(test_manifest_data)

    train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))
    test_set = set(test_charset.keys())

    print(f"Number of tokens in train+dev set : {len(train_dev_set)}")
    print(f"Number of tokens in test set : {len(test_set)}")

    # OOV tokens in test set
    train_test_common = set.intersection(train_dev_set, test_set)
    test_oov = test_set - train_test_common
    print(f"Number of OOV tokens in test set : {len(test_oov)}")
    print()
    print(test_oov)

    # Populate dictionary mapping count: list[tokens]
    train_counts = defaultdict(list)
    for token, count in train_charset.items():
        train_counts[count].append(token)
    for token, count in dev_charset.items():
        train_counts[count].append(token)

    # Compute sorter order of the count keys
    count_keys = sorted(list(train_counts.keys()))
