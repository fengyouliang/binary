#!/usr/bin/env python
# coding: utf-8
import math
import os

import torch
from torch import nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import config
from dataset import MyDataset
from losses import FocalLoss
from models.model import registry_model
from train import trainer


def train_loop():
    model = registry_model.get(config.model['name'])()

    model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    model = model.cuda(device=config.device_ids[0])

    optimizer = model.module.get_optimizer(config.lr, config.weight_decay, config.momentum)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=config.max_epoch)

    # criterion = registry_loss.get(config.criterion)()
    # criterion = FocalLoss()
    criterion = nn.CrossEntropyLoss()

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
    train_loader = DataLoader(train_data, batch_size=config.batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    trainer(model, optimizer, criterion, None, train_loader, val_loader, tqdm_length)


if __name__ == '__main__':
    train_loop()
