#!/usr/bin/env python
# coding: utf-8
import math
import os

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import config
from dataset import MyDataset
from losses import registry_loss
from models.model import registry_model
from train import trainer


def train_loop():
    model = registry_model.get(config.model['name'])()
    model = torch.nn.DataParallel(model)
    model = model.to(torch.device('cuda'))
    print(model)

    # optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)  #
    optimizer = Adam(model.parameters(), lr=config.lr)
    # optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    criterion = registry_loss.get(config.criterion)()
    step_scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    train_data = MyDataset('train')
    val_data = MyDataset('val')

    tqdm_length = math.ceil(len(train_data) / config.batch_size)

    if config.train_keep == -1:
        ok_count = len(os.listdir(f"{config.data_path}/train/ok/")) - 1
    else:
        ok_count = config.train_keep
    ng_count = len(os.listdir(f"{config.data_path}/train/ng/"))
    weights = [1 / ok_count for i in range(ok_count)] + [1 / ng_count for i in range(ng_count)]
    sampler = WeightedRandomSampler(weights, len(train_data), True)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    trainer(model, optimizer, criterion, step_scheduler, train_loader, val_loader, tqdm_length)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.device_ids))
    train_loop()
