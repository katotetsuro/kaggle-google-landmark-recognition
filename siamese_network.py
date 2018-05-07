import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNet50Layers


def create_model(weight='auto', activate=F.sigmoid):
    resnet = ResNet50Layers(pretrained_model=weight)
    model = chainer.Sequential(
        lambda x: resnet(x, layers=['res5'])['res5'],
        L.Linear(None, 128)
    )
    if activate:
        model.append(activate)
    return model


class SiameseNet(chainer.Chain):
    def __init__(self, activate, init_scale, weight='auto'):
        super().__init__()
        with self.init_scope():
            self.resnet = ResNet50Layers(weight)
            self.fc = L.Linear(
                None, 128, initialW=chainer.initializers.LeCunNormal(scale=init_scale))

        self.activate = activate

    def __call__(self, x):
        h = self.resnet(x, layers=['res5'])['res5']
        h = self.fc(h)
        if self.activate:
            h = self.activate(h)

        return h


class SiameseNetTrainChain(chainer.Chain):
    def __init__(self, model, margin=0.2):
        super().__init__()
        with self.init_scope():
            self.model = model
        self.margin = margin

    def __call__(self, anchor, positive, negative):
        anchor, positive, negative = map(
            self.model, (anchor, positive, negative))
        loss = F.triplet(anchor, positive, negative, self.margin)
        chainer.reporter.report({
            'loss': loss
        }, self)

        p = ((anchor.array - positive.array)**2).sum(axis=1)
        n = ((anchor.array - negative.array)**2).sum(axis=1)
        accuracy = self.xp.sum(p < n)
#        print(accuracy, anchor.shape[0])
        accuracy = accuracy / anchor.shape[0]
        chainer.reporter.report({'accuracy': accuracy}, self)
#        print(loss, accuracy)
        return loss
