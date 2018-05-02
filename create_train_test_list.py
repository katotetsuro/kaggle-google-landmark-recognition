import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser(
        description='本当に存在する画像だけからtrain/valのlistを作る')
    parser.add_argument('--file', default='all.txt',
                        help='download.pyの結果作成されたダウンロードに成功した画像の一覧')
    parser.add_argument('--out_train', default='train.txt',
                        help='出力されるリストのファイル名')
    parser.add_argument('--out_test', default='test.txt',
                        help='出力されるリストのファイル名')
    parser.add_argument('--rate', default=0.8, help='trainにまわす比率')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        all = list(map(lambda x: x.strip(), f.readlines()))

    random.shuffle(all)

    size = int(len(all) * args.rate)
    train = all[:size]
    test = all[size:]
    print('全画像数:{}, 学習用:{}, テスト用:{}'.format(len(all), len(train), len(test)))

    with open(args.out_train, 'w') as f:
        f.write('\n'.join(train))

    with open(args.out_test, 'w') as f:
        f.write('\n'.join(test))


if __name__ == '__main__':
    main()
