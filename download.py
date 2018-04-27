import pandas as pd
import os
from os.path import join, exists
import requests
import shutil
from multiprocessing import Pool
import multiprocessing as multi
import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image


def download_image(box):
    if not os.path.isfile(box[0]):
        try:
            r = requests.get(box[1], stream=True)
            if r.status_code == 200:
                with open(box[0], "wb") as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
                Image.open(f)
                return box[0], True
            else:
                return box[0], False
        except Exception as e:
            return box[0], False
    else:
        return box[0], True


def main():
    parser = argparse.ArgumentParser(
        description='画像をダウンロードする')
    parser.add_argument('--file', default='train.csv',
                        help='kaggleで配布されているtrain.csv or test.csvの場所')
    parser.add_argument('--out', default='data', help='ダウンロード先ディレクトリ')
    parser.add_argument('--txt', default='all.txt', help='ダウンロードに成功した画像の一覧')
    parser.add_argument('--retry', default=None, type=str,
                        help='これを指定した場合は、このリスト内から再ダウンロードする. check_image.pyと併せて使う')
    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.file)
    p = Pool(multi.cpu_count())

    box = []
    if args.retry is not None:
        with open(args.retry, 'r') as f:
            targets = list(map(lambda x: Path(x.strip()).stem, f.readlines()))
        df = df[df.id.isin(targets)]
        print('number of retry targets:{}'.format(len(df)))
    for i, row in tqdm(df.iterrows(), desc='read csv', total=len(df), unit='lines'):
        out_dir = join(args.out, '{0:03d}'.format(i // 10000))
        if not exists(out_dir):
            print('create new directory:{}'.format(out_dir))
            os.makedirs(out_dir)
        file_name = join(out_dir, '{}.jpg'.format(row['id']))
        box.append([file_name, row['url']])

    available_images = []
    for i, (path, status) in enumerate(tqdm(p.imap_unordered(download_image, box), desc='download', total=len(box), unit='files')):
        if status:
            available_images.append(path)

        if i % 10000 == 0:
            print('iter: {}, num_downloaded: {}'.format(
                i, len(available_images)))
        p.close()

    with open(args.txt, 'w') as f:
        f.write('\n'.join(available_images))


if __name__ == '__main__':
    main()
