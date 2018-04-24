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
            return 'success: {}'.format(box[0]), True
        else:
            return 'fail: {}'.format(box[0]), False
    else:
        print('already exists: {}'.format(box[0]))


def main():
    parser = argparse.ArgumentParser(
        description='画像をダウンロードする')
    parser.add_argument('--file', default='train.csv',
                        help='kaggleで配布されているtrain.csv or test.csvの場所')
    parser.add_argument('--out', default='data', help='ダウンロード先ディレクトリ')
    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.file)
    p = Pool(4)

    box = []
    for i, row in df.iterrows():
        url = row['url']
        out_dir = join(args.out, '{0:03d}'.format(i // 10000))
        if not exists(out_dir):
            print('create new directory:{}'.format(out_dir))
            os.makedirs(out_dir)
        file_name = join(out_dir, '{}.jpg'.format(row['id']))
        box.append([file_name, url])

    print('総ジョブ数', len(box))

    for msg, status in p.imap_unordered(download_image, box):
        if status == False:
            print(result)
        p.close()


if __name__ == '__main__':
    main()
