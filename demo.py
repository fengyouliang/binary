import torchvision
from torch import nn

import config


class mobile(nn.Module):
    def __init__(self, num_classes=2):
        super(mobile, self).__init__()
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


from models.model import mobilenet

net = mobilenet()
m = mobile()
print(net.parameters())
print(m.parameters())
