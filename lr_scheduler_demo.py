import math

import matplotlib.pyplot as plt

# model = AlexNet(num_classes=2)
# optimizer = optim.SGD(params=model.parameters(), lr=1)
#
# # lr_scheduler.StepLR()
# # Assuming optimizer uses lr = 0.05 for all groups
# # lr = 0.05     if epoch < 30
# # lr = 0.005    if 30 <= epoch < 60
# # lr = 0.0005   if 60 <= epoch < 90
#
# epoch = 100
#
# scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=0, last_epoch=-1)
# plt.figure()
# x = list(range(epoch))
# y = []
# for epoch in range(epoch):
#     scheduler.step()
#     lr = scheduler.get_last_lr()
#     print(epoch, scheduler.get_last_lr()[0])
#     y.append(scheduler.get_last_lr()[0])
#
# plt.plot(x, y)
# plt.show()
#
#
# def adjust_lr(optimizer, iter, gamma=0.5, warm_up=3):
#     original_lr = config.lr
#     if iter < warm_up:
#         new_lr = original_lr * (iter + 1) / warm_up
#     else:
#         new_lr = original_lr * gamma * (1 + math.cos(iter / config.max_epoch * math.pi))
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = new_lr

epoch = 100
plt.figure()
x = list(range(epoch))
y = []
original_lr = 0.001
warm_up = 5
for e in range(epoch):
    e += 1
    if e < warm_up:
        lr = original_lr * (e / warm_up)
    else:
        lr = original_lr * 0.5 * (1 + math.cos(e / epoch * math.pi))
    y.append(lr)

plt.plot(x, y)
plt.show()
