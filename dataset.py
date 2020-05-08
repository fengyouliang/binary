import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


class MyDataset(Dataset):
    def __init__(self, type_mode, reszie=config.resize, transform=None):
        super(MyDataset, self).__init__()

        self.resize = reszie
        self.transform = transform
        self.type_mode = type_mode

        self.root = config.data_path

        self.pos = f'{self.root}/{type_mode}/ok/'
        self.neg = f'{self.root}/{type_mode}/ng/'

        pos_images = [self.pos + file for file in os.listdir(self.pos)]
        pos_label = [1 for i in range(len(pos_images))]
        neg_images = [self.neg + file for file in os.listdir(self.neg)]
        neg_label = [0 for i in range(len(neg_images))]

        if self.type_mode == 'train':
            random.seed(42)
            random.shuffle(pos_images)
            random.seed(42)
            random.shuffle(pos_label)
            pos_images = pos_images[:config.train_keep]
            pos_label = pos_label[:config.train_keep]
        elif self.type_mode == 'val':
            random.seed(42)
            random.shuffle(pos_images)
            random.seed(42)
            random.shuffle(pos_label)
            pos_images = pos_images[:config.val_keep]
            pos_label = pos_label[:config.val_keep]

        self.all_image = pos_images + neg_images
        self.all_label = pos_label + neg_label

    def __len__(self):
        return len(self.all_image)

    def __getitem__(self, index):
        img, label = self.all_image[index], self.all_label[index]
        if self.type_mode == 'train':
            tf = transforms.Compose([
                lambda x: Image.open(x),
                transforms.Resize((int(self.resize[0]), int(self.resize[1]))),
                # transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                # transforms.RandomErasing(),
            ])
        else:
            tf = transforms.Compose([
                lambda x: Image.open(x),
                transforms.Resize((int(self.resize[0]), int(self.resize[1]))),
                transforms.ToTensor(),
            ])

        img = tf(img)

        label = torch.tensor(label)

        return img, label
