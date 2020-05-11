import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import AlexNet

model = AlexNet(num_classes=2)
optimizer = optim.SGD(params=model.parameters(), lr=1)

# lr_scheduler.StepLR()
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90

epoch = 50

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=epoch, eta_min=0, last_epoch=-1)
plt.figure()
x = list(range(epoch))
y = []
for epoch in range(epoch):
    scheduler.step()
    lr = scheduler.get_last_lr()
    print(epoch, scheduler.get_last_lr()[0])
    y.append(scheduler.get_last_lr()[0])

plt.plot(x, y)
plt.show()
