def convert_wav_scp_path(input, out):
    f = open(input, 'r', encoding='utf-8')
    fo = open(out, 'w', encoding='utf-8')
    for line in f:
        tokens = line.split(' ')
        id = tokens[0]
        path = tokens[5]
        path_formatted = path.replace('clips', 'clips-wav').replace('mp3', 'wav')
        fo.write(id + ' ' + path_formatted + '\n')


if __name__ == '__main__':
    input_path = '/Users/zeynep/Desktop/zeynep/data/commonvoice-tr/k2-data/train/wav.scp'
    out_path = '/Users/zeynep/Desktop/zeynep/data/commonvoice-tr/k2-data/train/wav_formatted.scp'

    convert_wav_scp_path(input_path, out_path)
