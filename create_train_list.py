import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser(
        description='本当に存在する画像だけからtrain/valのlistを作る')
    parser.add_argument('--file', default='train.csv',
                        help='kaggleで配布されているtrain.csv or test.csvの場所')
    parser.add_argument('--image_dir', default='data/train',
                        help='画像が保存されているディレクトリ')
    parser.add_argument('--out_train', default='train.txt',
                        help='出力されるリストのファイル名')
    parser.add_argument('--out_test', default='test.txt',
                        help='出力されるリストのファイル名')
    parser.add_argument('--rate', default=0.8, help='trainにまわす比率')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        files = map(lambda x: x.replace('"', '').split(','), f.readlines())

    available_files = []
    for i, f in enumerate(files):
        if i == 0:
            # skip head
            continue
        id, url, label = f
        image_file_name = '{}.png'.format(id)
        if os.path.isfile(os.path.join(args.image_dir, image_file_name)):
            available_files.append(' '.join([image_file_name, label]))
        else:
            print('image file doesnt exist. {}'.format(
                os.path.join(args.image_dir, image_file_name)))

        if i % 1000 == 0:
            print(i)

    print('number of available_images:{}'.format(len(available_files)))

    random.shuffle(available_files)

    train_size = int(len(available_files) * args.rate)
    train_files = available_files[:train_size]
    with open(args.out_train, 'w') as f:
        f.writelines(train_files)

    test_files = available_files[:-train_size]
    with open(args.out_test, 'w') as f:
        f.writelines(test_files)


if __name__ == '__main__':
    main()
