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
from io import BytesIO
from PIL import Image


def download_image(box):
    file_path = join(box['base_dir'], box['file_path'])
    if not os.path.isfile(file_path):
        try:
            r = requests.get(box['url'], stream=True)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content))
                w, h = img.size
                s = 224 / min(w, h)
                img = img.resize((int(w*s), int(h*s)), Image.BILINEAR)
                data = list(img.getdata())
                img = Image.new(img.mode, img.size)
                img.putdata(data)
                img.save(file_path)
                return box, True
            else:
                return box, False
        except Exception as e:
            return box, False
    else:
        return box, True


def main():
    parser = argparse.ArgumentParser(
        description='画像をダウンロードする')
    parser.add_argument('--file', default='train.csv',
                        help='kaggleで配布されているtrain.csv or test.csvの場所')
    parser.add_argument('--out', default='data', help='ダウンロード先ディレクトリ')
    parser.add_argument('--txt', default='all.txt',
                        help='ダウンロードに成功した画像の一覧')
    parser.add_argument('--retry', default=None, type=str,
                        help='これを指定した場合は、このリスト内から再ダウンロードする. check_image.pyと併せて使う')
    parser.add_argument('--limit', default=-1,
                        type=int, help='最大ダウンロード数')
    parser.add_argument('--resolution', default=-1,
                        type=int, help='ダウンロードする解像度')
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
        d = '{0:03d}'.format(i // 10000)
        out_dir = join(args.out, d)
        if not exists(out_dir):
            print('create new directory:{}'.format(out_dir))
            os.makedirs(out_dir)
        row['base_dir'] = args.out
        row['file_path'] = join(d, '{}.jpg'.format(row['id']))
        box.append(row)
        if args.limit > 0 and i > args.limit:
            break

    available_images = []
    for i, (box, status) in enumerate(tqdm(p.imap_unordered(download_image, box), desc='download', total=len(box), unit='files')):
        if status:
            available_images.append('{} {}'.format(
                box['file_path'], box['landmark_id']))

        if i % 10000 == 0:
            print('iter: {}, num_downloaded: {}'.format(
                i, len(available_images)))
        p.close()

    print('iter: {}, num_downloaded: {}'.format(
        i, len(available_images)))
    with open(args.txt, 'w') as f:
        f.write('\n'.join(available_images))


if __name__ == '__main__':
    main()
