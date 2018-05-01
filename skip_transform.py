import chainer
import numpy as np


class SkipTransform(chainer.datasets.transform_dataset.TransformDataset):
    """
    get_exampleで読もうとしたときに例外が発生するデータを(zeros, -1)で飛ばすTransform
    """
    failed_indices = []

    def __init__(self,  dataset, transform, with_label=True):
        super().__init__(dataset, transform)
        self.with_label = True

    def get_example(self, index):
        try:
            return super().get_example(index)
        except Exception as e:
            print(e)
            print('failed to load data index={}'.format(index))
            failed_indices.append(index)
            if with_label:
                return np.zeros((3, 224, 224), dtype=np.float32), np.array(-1, dtype=np.int32)
            else:
                return np.zeros((3, 224, 224), dtype=np.float32)
