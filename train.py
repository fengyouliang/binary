import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import config
import utils
from config import *


def trainer(model, optimizer, criterion, train_loader, val_loader, tqdm_length):
    best_acc = best_val_acc
    best_ok_ap = 0
    best_ng_ap = 0
    best_ap = 0
    best_FNR = 0

    for epoch in range(config.max_epoch):

        # utils.adjust_lr(optimizer, epoch)

        bar = tqdm(enumerate(train_loader), total=tqdm_length)
        for ii, (data, label) in bar:
            input = data.cuda()
            target = label.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            bar.set_description(
                f'{epoch}-{ii} loss:{loss.item():.4f} lr:{optimizer.state_dict()["param_groups"][0]["lr"]:.10f}')

        model.module.save()

        val_accuracy, y_true, y_score = val(model, val_loader)

        # AP
        ok_val_ap, ng_val_ap, mAP = utils.get_AP_metric(y_true, y_score)

        # FNR
        final_metric_dict = utils.get_FRN_metric(y_true, y_score)

        print(f'Acc: {val_accuracy:.2f}\t OK_AP：{ok_val_ap:.2f}\t NG_AP: {ng_val_ap:.2f}\t mAP: {mAP:.2f}')
        print(f'BEST Acc: {best_acc:.2f}\t OK_AP: {best_ok_ap:.2f}\t NG_AP: {best_ng_ap:.2f}\t mAP: {best_ap:.2f}')
        print(final_metric_dict)

        if final_metric_dict['FNR'] > best_FNR:
            best_FNR = final_metric_dict['FNR']
            if config.save_flag:
                model.module.save(
                    f'./best_FNR_model/{epoch}_acc_{val_accuracy:.4f}_mAP_{mAP}_FNR{final_metric_dict["FNR"]:.4F}.pth')
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            if config.save_flag:
                model.module.save(
                    f'./best_acc_model/{epoch}_acc_{val_accuracy:.4f}_ok_ap_{ok_val_ap:.4f}_ng_ap_{ng_val_ap}_mAP_{mAP}.pth')
        if mAP > best_ap:
            best_ap = mAP
            if config.save_flag:
                model.module.save(
                    f'./best_ap_model/{epoch}_acc_{val_accuracy:.4f}_ok_ap_{ok_val_ap:.4f}_ng_ap_{ng_val_ap}_mAP_{mAP}.pth')
        best_ok_ap = max(ok_val_ap, best_ok_ap)
        best_ng_ap = max(ng_val_ap, best_ng_ap)


@torch.no_grad()
def val(model, dataloader):
    model.eval()

    correct = 0
    total = 0

    y_true = np.array([])
    y_score = np.zeros(shape=(1, 2))

    for x, y in tqdm(dataloader):
        x = x.cuda()
        y = y.cuda()

        output = model(x)
        _, predicted = torch.max(output.data, 1)

        softmax = nn.functional.softmax
        s_pred = softmax(output, dim=1)

        y_true = np.append(y_true, y.data.cpu().numpy())
        y_score = np.concatenate((y_score, s_pred.data.cpu().numpy()), axis=0)

        total += y.size(0)
        correct += (predicted == y).sum().item()

    assert y_true.shape[0] == y_score.shape[0] - 1

    return correct / total, y_true, y_score[1:]
