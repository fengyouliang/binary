#!/usr/bin/env python
# coding: utf-8
import math
import os

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config
from losses import registry_loss
from models.model import registry_model
from train import trainer

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.device_ids))


def train_loop():
    model = registry_model.get(config.model['name'])(num_classes=config.num_class)
    model = torch.nn.DataParallel(model)
    model = model.to(torch.device('cuda'))

    optimizer = Adam(model.parameters(), lr=config.lr)

    # step_scheduler = lr_scheduler.StepLR(optimizer, 3, gamma=0.7, last_epoch=-1)
    cos_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=1e-6, last_epoch=-1)
    scheduler = cos_scheduler

    criterion = registry_loss.get(config.criterion)()

    print(config.criterion)
    print(config.lr)

    train_tf = transforms.Compose([
        transforms.Resize(config.resize),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(config.resize),
        transforms.ToTensor(),
    ])
    GIC_train_dataset = ImageFolder(root='/home/youliang/datasets/GIC/train', transform=train_tf)
    GIC_train_loader = DataLoader(GIC_train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    GIC_val_dataset = ImageFolder(root='/home/youliang/datasets/GIC/val', transform=test_tf)
    GIC_val_loader = DataLoader(GIC_val_dataset, batch_size=config.test_batch_size, pin_memory=True)

    tqdm_length = math.ceil(len(GIC_train_dataset) / config.batch_size)

    trainer(model, optimizer, criterion, scheduler, GIC_train_loader, GIC_val_loader, tqdm_length)


if __name__ == '__main__':
    train_loop()
