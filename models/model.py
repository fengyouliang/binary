import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torch import nn

import config
from models.danet import DANetHead
from register import Registry

from torchvision import models

registry_model = Registry('model')


@registry_model.register()
class mobilenet(nn.Module):
    def __init__(self, num_classes=2):
        super(mobilenet, self).__init__()
        self.model_name = 'mobilenet'

        net = models.mobilenet_v2(pretrained=config.is_pretrained, num_classes=1000)

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
class efficientnet(nn.Module):
    def __init__(self, net_type='efficientnet-b7', num_classes=2):
        super(efficientnet, self).__init__()
        self.model_name = 'EfficientNet'

        linear_length = {
            'b1': 1280,
            'b3': 1536,
            'b7': 2560,
        }

        model = EfficientNet.from_pretrained(net_type)
        self.features = model.extract_features
        self.classifer = nn.Linear(linear_length[net_type[-2:]], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


@registry_model.register()
class resnext101_32x8d(nn.Module):
    def __init__(self, num_classes=2):
        super(resnext101_32x8d, self).__init__()
        net = models.resnext101_32x8d(pretrained=config.is_pretrained, num_classes=1000)
        self.features = nn.Sequential(
            *list(net.children())[:-2],
        )
        self.feature_channels = 2048
        self.dan = DANetHead(self.feature_channels, self.feature_channels)
        self.classifer = nn.Linear(self.feature_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


@registry_model.register()
class DANet(nn.Module):
    def __init__(self, num_classes=2):
        super(DANet, self).__init__()
        net = models.resnet18(pretrained=config.is_pretrained, num_classes=1000)
        self.features = nn.Sequential(
            *list(net.children())[:-2],
        )
        self.dan = DANetHead(512, 512)
        self.classifer = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.dan(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    model = DANet()

    bs = 4
    x = torch.randn(bs, 3, 112, 224)
    logits = model(x)
    print(logits.shape)


