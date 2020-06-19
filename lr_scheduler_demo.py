import math
from torchvision.models import AlexNet
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def pytorch_cos():
    model = AlexNet(num_classes=2)
    optimizer = optim.SGD(params=model.parameters(), lr=0.0001)

    epoch = 100
    len_loader = 100

    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=1e-6, last_epoch=-1)
    plt.figure()
    x = []
    y = []
    for e in range(epoch):
        for i in range(len_loader):
            step = e + i / len_loader
            scheduler.step(step)
            lr = scheduler.get_last_lr()[0]

            x.append(step)
            y.append(lr)

    plt.plot(x, y)
    plt.xticks(np.arange(0, epoch + 1, 4))
    plt.show()


def custom_cos():
    epoch = 100
    plt.figure()
    x = list(range(epoch))
    y = []
    original_lr = 1
    warm_up = 5
    for e in range(epoch):
        e += 1
        if e < warm_up:
            lr = original_lr * (e / warm_up)
        else:
            lr = original_lr * 0.5 * (1 + math.cos(e / epoch * math.pi))
        y.append(lr)

    plt.plot(x, y)
    plt.xticks(np.arange(0, epoch + 1, 5))
    plt.show()


def compute_eta_t(eta_min, eta_max, T_cur, Ti):
    """Equation (5).
    # Arguments
        eta_min,eta_max,T_cur,Ti are same as equation.
    # Returns
        eta_t
    """
    pi = np.pi
    # if T_cur / Ti == 1/2:
    #     eta_max *= 0.1
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (np.cos(pi * T_cur / Ti) + 1)
    return eta_t


def numpy_cos():
    # 每Ti个epoch进行一次restart。
    Ti = [50, 100]
    n_batches = 200
    eta_ts = []
    for ti in Ti:
        T_cur = np.arange(0, ti, 1 / n_batches)
        for t_cur in T_cur:
            eta_ts.append(compute_eta_t(0, 1, t_cur, ti))

    n_iterations = sum(Ti) * n_batches
    epoch = np.arange(0, n_iterations) / n_batches

    plt.plot(epoch, eta_ts)
    plt.show()


def main():
    pytorch_cos()
    # numpy_cos()


if __name__ == '__main__':
    main()
