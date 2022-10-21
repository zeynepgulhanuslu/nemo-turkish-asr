import argparse
import json
import os

# Manifest Utils
from tqdm.auto import tqdm
import sox


def convert_kaldi_to_manifest(kaldi_dir, manifest_file):
    text_file = os.path.join(kaldi_dir, 'text')
    wav_scp_file = os.path.join(kaldi_dir, 'wav.scp')
    utt_dur_file = os.path.join(kaldi_dir, 'utt2dur')
    f_m = open(manifest_file, mode='w', encoding='utf-8')
    f_w = open(wav_scp_file, 'r', encoding='utf-8')
    f_t = open(text_file, 'r', encoding='utf-8')

    dur_dict = {}
    if os.path.exists(utt_dur_file):
        f_u = open(utt_dur_file, 'r', encoding='utf-8')
        for line in f_u:
            tokens = line.strip().split()
            dur_dict[tokens[0]] = tokens[1]

    text_dict = {}
    audio_dict = {}

    for line in f_t:
        tokens = line.strip().split(' ', 1)
        text_dict[tokens[0]] = tokens[1]

    for line in f_w:
        tokens = line.strip().split(' ', 1)
        audio_dict[tokens[0]] = tokens[1]

    for key, value in tqdm(audio_dict.items()):
        audio_path = value.strip()
        text = text_dict[key].strip()
        if len(dur_dict) > 0:
            duration = dur_dict[key]
        else:
            duration = sox.file_info.duration(audio_path)
        f_m.write(
            json.dumps({'audio_filepath': os.path.abspath(audio_path), "duration": duration, 'text': text})
            + '\n'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--kaldi_data', type=str, required=True, help='Kaldi data directory.')
    parser.add_argument('--manifest', type=str, required=True, help='manifest file')

    args = parser.parse_args()

    kaldi_data = args.kaldi_data
    manifest_file = args.manifest

    convert_kaldi_to_manifest(kaldi_data, manifest_file)
