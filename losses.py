import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from register import Registry

registry_loss = Registry('loss')


@registry_loss.register()
class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.5):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param  # gamma
        self.balance_param = balance_param  # alpha

    def forward(self, output, target):
        # cross_entropy = F.cross_entropy(output, target)
        # cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


@registry_loss.register()
def crossentropyloss():
    return nn.CrossEntropyLoss()


@registry_loss.register()
def focalloss():
    return FocalLoss()


def test_focal_loss():
    loss = FocalLoss()

    input = Variable(torch.randn(3, 5), requires_grad=True)
    target = Variable(torch.LongTensor(3).random_(5))

    print(input.shape)
    print(target.shape)

    output = loss(input, target)
    print(output)
    output.backward()


if __name__ == '__main__':
    test_focal_loss()
