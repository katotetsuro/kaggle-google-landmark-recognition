import chainer
import numpy as np


class SkippableDataset(chainer.datasets.LabeledImageDataset):
    def get_example(self, index):
        try:
            return super().get_example(index)
        except Exception as e:
            print(e)
            print('failed to load data index={}'.format(index))
            return np.zeros((3, 224, 224), dtype=np.uint8), -1
