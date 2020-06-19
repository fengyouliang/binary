import torchvision
from torch import nn
from torchvision import models

import config
from models.danet import DANetHead
from models.efficientnet import EfficientNet
from register import Registry


registry_model = Registry('model')


@registry_model.register()
class mobilenet(nn.Module):
    def __init__(self, num_classes=2):
        super(mobilenet, self).__init__()
        net = torchvision.models.mobilenet_v2(pretrained=config.is_pretrained, num_classes=1000)

        self.features = nn.Sequential(
            *list(net.children())[:-1],
        )
        self.classifer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


@registry_model.register()
class resnext101_32x8d(nn.Module):
    def __init__(self, num_classes=2):
        super(resnext101_32x8d, self).__init__()
        net = torchvision.models.resnext101_32x8d(pretrained=config.is_pretrained, num_classes=1000)
        self.features = nn.Sequential(
            *list(net.children())[:-1],
        )
        self.classifer = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


@registry_model.register()
class mobilenet_da(nn.Module):
    def __init__(self, num_classes=2, att=True):
        super(mobilenet_da, self).__init__()
        net = models.mobilenet_v2(pretrained=config.is_pretrained, num_classes=1000)
        self.features = nn.Sequential(
            *list(net.children())[:-1],
        )
        self.att = att
        self.feature_channels = 1280
        if self.att:
            self.dan = DANetHead(self.feature_channels, self.feature_channels)
        self.classifer = nn.Linear(self.feature_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        if self.att:
            x = self.dan(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


@registry_model.register()
class efficientnet(nn.Module):
    def __init__(self, net_type='efficientnet-b0', num_classes=2):
        super(efficientnet, self).__init__()
        self.net = EfficientNet.from_pretrained(net_type, num_classes=num_classes)

    def forward(self, x):
        x = self.net(x)
        return x


@registry_model.register()
class resnext101_32x8d_da(nn.Module):
    def __init__(self, num_classes=2, att=False):
        super(resnext101_32x8d_da, self).__init__()
        self.att = att
        net = models.resnext101_32x8d(pretrained=config.is_pretrained, num_classes=1000)
        self.features = nn.Sequential(
            *list(net.children())[:-2],
        )
        self.feature_channels = 2048
        if self.att:
            self.dan = DANetHead(self.feature_channels, self.feature_channels)
        self.classifer = nn.Linear(self.feature_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        if self.att:
            x = self.dan(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    pass
