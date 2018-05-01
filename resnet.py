import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import ResNet50Layers


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h


class GroupedResNet(chainer.Chain):
    """
    予測するクラスをいくつかのGroupにわけてみる
    """

    def __init__(self, num_class, num_groups, use_pretrained=True):
        super().__init__()
        print('num_groups: {}'.format(num_groups))
        res5_output_channels = 2048
        out_channels = int(num_class / num_groups + 0.5) * num_groups
        print('number of channels per group: {}'.format(
            res5_output_channels / num_groups))
        print('number of output per group: {}'.format(out_channels / num_groups))
        print('最後のグループで無駄に使ってしまうチャンネル数: {}'.format(out_channels - num_class))

        with self.init_scope():
            self.resnet = chainer.links.ResNet50Layers(
                'auto' if use_pretrained else None)
            self.resnet.fc6 = L.Linear(None, 1)
            self.conv = L.Convolution2D(
                in_channels=None, out_channels=out_channels, ksize=3, groups=num_groups)

    def __call__(self, x):
        h = self.resnet(x, layers=['res5'])['res5']
        h = self.conv(h)

        return _global_average_pooling_2d(h)
