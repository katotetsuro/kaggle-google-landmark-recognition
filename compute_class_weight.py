import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='クラスごとの出現数から、softmax_cross_entropyで使うweightを計算する')
    parser.add_argument('--file', default='train.txt')
    args = parser.parse_args()

    if Path(args.file).suffix == '.txt':
        with open(args.file, 'r') as f:
            labels = list(
                map(lambda x: int(x.strip().split()[1]), f.readlines()))

        labels = np.array(labels, dtype=np.float32)

    elif Path(args.file).suffix == '.csv':
        df = pd.read_csv(args.file)
        labels = np.array(df['landmark_id'], dtype=np.float32)
    sorted_labels, counts = np.unique(labels, return_counts=True)
    weight = 1. / counts

    print(min(labels))
    print(weight, weight.shape)
    np.save('weight.npy', weight)


if __name__ == '__main__':
    main()
