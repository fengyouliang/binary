import time

import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torch import nn

import config
from register import Registry

registry_model = Registry('model')


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay, momentum):
        # opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        opt = torch.optim.AdamW(self.parameters())
        return opt


class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


@registry_model.register()
class mobilenet(BasicModule):
    def __init__(self, num_classes=2):
        super(mobilenet, self).__init__()
        self.model_name = 'mobilenet'

        net = torchvision.models.mobilenet_v2(pretrained=config.is_pretrained, num_classes=1000)

        self.features = nn.Sequential(
            *list(net.children())[:-1],
        )
        self.classifer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_max_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


@registry_model.register()
class efficientnet(BasicModule):
    def __init__(self, net_type='efficientnet-b0', num_classes=2):
        super(efficientnet, self).__init__()
        self.model_name = 'EfficientNet'

        model = EfficientNet.from_pretrained(net_type)
        self.features = model.extract_features
        self.classifer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


@registry_model.register()
class resnext101_32x8d(BasicModule):
    def __init__(self, num_classes=2):
        super(resnext101_32x8d, self).__init__()
        self.model_name = 'resnext101_32x8d'

        net = torchvision.models.resnext101_32x8d(pretrained=config.is_pretrained, num_classes=1000)

        self.features = nn.Sequential(
            *list(net.children())[:-1],
        )
        self.classifer = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_max_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    net = registry_model.get(config.model['name'])()
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(y.shape)
    # from pretrainedmodels.models import resnext101_64x4d
    # net = resnext101_64x4d(num_classes=1000, pretrained='imagenet')
    # print(net)
