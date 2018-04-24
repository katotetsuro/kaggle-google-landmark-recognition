import pandas as pd
import os
from os.path import join, exists
import requests
import shutil
from multiprocessing import Pool
import multiprocessing as multi
import argparse


def download_image(box):
    if not os.path.isfile(box[0]):
        r = requests.get(box[1], stream=True)
        if r.status_code == 200:
            with open(box[0], "wb") as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            print('success: {}'.format(box[0]))
            return box[0], True
        else:
            print('fail: {}'.format(box[0]))
            return box[0], False
    else:
        print('already exists: {}'.format(box[0]))
        return box[0], True


def main():
    parser = argparse.ArgumentParser(
        description='画像をダウンロードする')
    parser.add_argument('--file', default='train.csv',
                        help='kaggleで配布されているtrain.csv or test.csvの場所')
    parser.add_argument('--out', default='data', help='ダウンロード先ディレクトリ')
    parser.add_argument('--txt', default='all.txt', help='ダウンロードに成功した画像の一覧')
    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.file)
    p = Pool(multi.cpu_count())

    box = []
    for i, row in df.iterrows():
        url = row['url']
        out_dir = join(args.out, '{0:03d}'.format(i // 10000))
        if not exists(out_dir):
            print('create new directory:{}'.format(out_dir))
            os.makedirs(out_dir)
        file_name = join(out_dir, '{}.jpg'.format(row['id']))
        box.append([file_name, url])

    available_images = []
    for path, status in p.imap_unordered(download_image, box):
        if status:
            available_images.append(path)
        p.close()

    with open(args.txt, 'w') as f:
        f.write('\n'.join(available_images))


if __name__ == '__main__':
    main()
