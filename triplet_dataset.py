import chainer
import numpy
import pandas
from PIL import Image
import numpy as np


class TripletDataset(chainer.dataset.DatasetMixin):
    """
    3つに固定する必要がない実装になったけど、まぁいいか
    """

    def __init__(self, file, root='.', dtype=numpy.float32):
        df = pandas.read_csv(file, delimiter=' ', names=[
            'anchor', 'positive', 'negative'])
        self.anchor = chainer.datasets.ImageDataset(
            df['anchor'], root=root, dtype=dtype)
        self.positive = chainer.datasets.ImageDataset(
            df['positive'], root=root, dtype=dtype)
        self.negative = chainer.datasets.ImageDataset(
            df['negative'], root=root, dtype=dtype)

    def __len__(self):
        return len(self.anchor)

    def _preprocess(self, x):
        if x.shape[0] == 1:
            x = np.concatenate([x, x, x])
        elif x.shape[0] > 3:
            x = x[:3]

        img = Image.fromarray(x.transpose((1, 2, 0)))
        img = img.resize((224, 224), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)
        img = img.transpose((2, 0, 1))
        img = img / 255
        return img

    def get_example(self, index):
        return tuple(map(self._preprocess, [self.anchor[index], self.positive[index], self.negative[index]]))
