import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from register import Registry

registry_loss = Registry('loss')


@registry_loss.register()
class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=-1):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param  # gamma
        self.balance_param = balance_param  # alpha

    def forward(self, output, target):
        # cross_entropy = F.cross_entropy(output, target)
        # cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt
        if self.balance_param == -1:
            balanced_focal_loss = focal_loss
        else:
            balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


class FocalLoss_(nn.Module):
    # https://blog.csdn.net/Code_Mart/article/details/89736187
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss_, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        # pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1 - pred, pred), dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.
        # For each value in src, its output index is specified by its index in src for dimension != dim
        # and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha

        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class FocalLoss_arcface(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss_arcface, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


@registry_loss.register()
class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.sigmoid_flag = False

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):

        if self.sigmoid_flag:
            prob = torch.sigmoid(output)
            prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)
        else:
            prob = F.softmax(output, dim=1)

        # pos_mask = (target == 1).float()
        # neg_mask = (target == 0).float()
        # pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        # neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * torch.log(torch.sub(1.0, prob)) * neg_mask

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob)
        pos_loss = pos_loss[target == 1]
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * torch.log(torch.sub(1.0, prob))
        neg_loss = neg_loss[target == 0]

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        # num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        # num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        num_pos = sum((target == 1).float())
        num_neg = sum((target == 0).float())

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss


class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=[1, 1], gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 0  # 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones(self.num_class)
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
        pt = (one_hot_key * logit).sum(1) + self.eps

        # ----------memory saving way--------
        # pt = logit.gather(1, target).view(-1) + self.eps  # avoid apply

        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0, target.view(-1))
            logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss_2(nn.Module):
    # https://blog.csdn.net/qq_36401512/article/details/91450292
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss_2, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        target = target.long()

        # loss = self.alpha * (1 - p) ** self.gamma * logp * target + \
        #     (1 - self.alpha) * p ** self.gamma * logp * (1 - target)

        # without alpha class weight
        loss = (1 - p) ** self.gamma * logp * target + p ** self.gamma * logp * (1 - target)

        return loss.mean()


class FocalLoss(nn.Module):
    """ cross entropy focal loss
    https://www.kaggle.com/hmendonca/kaggle-pytorch-utility-script
    """

    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            self.alpha = self.alpha.type(inputs.type(), non_blocking=True)  # fix type and device
            alpha = self.alpha[targets]
        else:
            alpha = 1.

        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()
        return F_loss


@registry_loss.register()
def crossentropyloss():
    return nn.CrossEntropyLoss()


@registry_loss.register()
def focalloss():
    return FocalLoss()


def test_focal_loss():
    N, C = 10, 2
    input = torch.rand(N, C)
    # input = F.softmax(input, dim=1)
    target = (torch.rand(N) > 0.5).long()
    print(input.shape)
    print(target.shape)
    # print(input, target)

    FL = FocalLoss_2()
    output = FL(input, target)
    print('FL: ', output)

    CE = nn.CrossEntropyLoss()
    ce = CE(input, target)
    print('CE: ', ce)


if __name__ == '__main__':
    test_focal_loss()
