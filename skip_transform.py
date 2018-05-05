import chainer
import numpy as np


class SkipTransform(chainer.datasets.transform_dataset.TransformDataset):
    """
    get_exampleで読もうとしたときに例外が発生するデータを(zeros, -1)で飛ばすTransform
    """
    failed_indices = []

    def __init__(self,  dataset, transform, default_value):
        super().__init__(dataset, transform)
        self.default_value = default_value

    def get_example(self, index):
        try:
            return super().get_example(index)
        except Exception as e:
            print(e)
            print('failed to load data index={}'.format(index))
            SkipTransform.failed_indices.append(index)
            return self.default_value
