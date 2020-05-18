import torch
import torch.nn as nn
import torch.nn.functional as F
model = nn.Sequential(
    nn.Conv2d(1, 3, 3, 3, 0, bias=False)
)
# print(model)

x = torch.randn(1, 1, 3, 3)
y = model(x)
y = y.reshape(y.size(1))
print(y)
prob = F.softmax(y, dim=0)  # .detach().data
print(prob, sum(prob))

target = [0, 1, 0]

criterion = nn.NLLLoss2d()

loss = criterion(prob, target)
print(loss)
