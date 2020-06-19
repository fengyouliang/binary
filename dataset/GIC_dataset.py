from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


tf = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
])
target_tf = transforms.Compose([
    transforms.ToTensor(),
])

GIC_dataset = ImageFolder(root='/home/youliang/datasets/GIC/train', transform=tf)
GIC_loader = DataLoader(GIC_dataset, batch_size=4, pin_memory=True)

for batch_data in GIC_loader:
    image, targe = batch_data
    print(image.shape)
    print(targe)

print()
