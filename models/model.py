import torchvision
from efficientnet_pytorch import EfficientNet
from torch import nn

import config
from register import Registry

registry_model = Registry('model')


@registry_model.register()
class mobilenet(nn.Module):
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


if __name__ == '__main__':
    # net = registry_model.get('efficientnet')('efficientnet-b7')
    # x = torch.randn(1, 3, 224, 224)
    # y = net(x)
    # print(y.shape)
    pass
