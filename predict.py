import argparse
from os.path import join
import random

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links import ResNet50Layers

import shaked_pyramid_net
import augmentor_transformer
import skip_transform
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='predict for submit:')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--model', '-m', default='pyramid', choices=['resnet50', 'pyramid'], required=True,
                        help='model')
    parser.add_argument('--weight', '-w', default='', required=True
                        help='load pretrained model')
    parser.add_argument('--start_channel', default=16,
                        type=int, help='start channel')
    parser.add_argument('--depth', default=16, type=int,
                        help='depth')
    parser.add_argument('--alpha', default=90, type=int,
                        help='alpha')
    parser.add_argument('--class_weight', default=None, type=str)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--tta', default=False, help='test time argument')
    args = parser.parse_args()

    print(args)
    print('')

    class_labels = 14951
    if args.model == 'resnet50':
        predictor = ResNet('auto')
        predictor.fc6 = L.Linear(2048, class_labels)
    elif args.model == 'pyramid':
        predictor = shaked_pyramid_net.PyramidNet(
            skip=True, num_class=class_labels, depth=args.depth, alpha=args.alpha, start_channel=args.start_channel)

    if not args.weight == '':
        chainer.serializers.load_npz(args.weight, model)

    data_file = join(args.data_dir, 'test.txt')
    test = chainer.datasets.ImageDataset(
        data_file, root=args.data_dir, dtype=np.uint8)
    test = skip_trainsform.SkipTransform(
        test, augmentor_transformer.AugmentorTransform(train=False))
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    with open(test_file, 'r') as f:
        ids = list(map(lambda x: Path(x.strip()).trim, f.readlines()))

    labels = []
    scores = []
    for i, batch in enumerate(test_iter):
        x = chainer.dataset.concat_examples(batch)
        y = F.Softmax(model(batch), axis=1)
        y = chainer.backends.cuda.to_cpu(y.array)
        labels.append(np.argmax(y, axis=1))
        scores.append(np.max(y, axis=1))

    labels = np.concatenate(labels)
    scores = np.concatenate(scores)

    df = pd.DataFrame([ids, labels, scores], columns=['id', 'labels', 'score'])
