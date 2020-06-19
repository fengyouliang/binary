import os
import time

import numpy as np
import torch
from sklearn import metrics
from torch import nn
from tqdm import tqdm

import config
import utils
from MetricEval import ClassifierEvalMulticlass, ClassifierEvalBinary


def trainer(model, optimizer, criterion, scheduler, train_loader, val_loader, tqdm_length, log_flag=False):
    best_acc = 0.5
    best_ap = 0
    best_FOR = 0
    best_ok_ap = 0
    best_ng_ap = 0

    best_ap_epoch = []
    best_acc_epoch = []
    best_FOR_epoch = []
    save_names = []

    for epoch in range(config.max_epoch):
        batch_avg_loss = 0

        bar = tqdm(enumerate(train_loader), total=tqdm_length)
        for ii, (data, label) in bar:
            image = data.cuda()
            target = label.cuda()

            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            cur_loss = loss.item()
            batch_avg_loss += cur_loss
            cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
            batch_loss = batch_avg_loss / (ii + 1)
            bar.set_description(f'{epoch} loss:{cur_loss:.2e} lr:{cur_lr:.2e}')

        if scheduler is None:
            utils.adjust_lr(optimizer, epoch)
        else:
            scheduler.step()

        val_accuracy, y_true, y_score = val(model, val_loader)
        if config.num_class == 2:
            # confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_true, np.argmax(y_score, 1))
            # AP
            # ok_val_ap, ng_val_ap, mAP = utils.get_AP_metric(y_true, y_score)
            mulit_class_ap = ClassifierEvalMulticlass.compute_ap(y_true, y_score)
            ng_val_ap = mulit_class_ap[0]
            ok_val_ap = mulit_class_ap[1]
            mAP = (ng_val_ap + ok_val_ap) / 2
            # FOR
            final_metric_dict = utils.get_FOR_metric(y_true, y_score)

            ok_y_score = y_score[:, 1]
            ok_p_at_r = ClassifierEvalBinary.compute_p_at_r(y_true, ok_y_score, 1)

            ng_y_true = np.array(y_true).astype("bool")
            ng_y_true = (1 - ng_y_true).astype(np.int)
            ng_y_score = y_score[:, 0]
            ng_p_at_r = ClassifierEvalBinary.compute_p_at_r(ng_y_true, ng_y_score, 1)

            print(f'Acc: {val_accuracy:.2f}\t OK_AP：{ok_val_ap:.2f}\t NG_AP: {ng_val_ap:.2f}\t mAP: {mAP:.2f}')
            print(f'BEST Acc: {best_acc:.2f}\t OK_AP: {best_ok_ap:.2f}\t NG_AP: {best_ng_ap:.2f}\t mAP: {best_ap:.2f}')
            print(confusion_matrix)
            print(mulit_class_ap)
            print(f'ok_p_at_r: {ok_p_at_r}, ng_p_at_r: {ng_p_at_r}')
            print(final_metric_dict)

            save_path = f'./checkpoints/{config.model["name"]}'
            save_name = f'{epoch}_acc_{val_accuracy:.4f}_p@r_{ng_p_at_r}_FOR_{final_metric_dict["FOR"]:.4F}.pth'
            save_names.append(save_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model, f'{save_path}/{save_name}')

            if final_metric_dict['FOR'] > best_FOR:
                best_FOR = final_metric_dict['FOR']
                best_FOR_epoch.append(epoch)

            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_acc_epoch.append(epoch)

            if mAP > best_ap:
                best_ap = mAP
                best_ap_epoch.append(epoch)

            best_ok_ap = max(ok_val_ap, best_ok_ap)
            best_ng_ap = max(ng_val_ap, best_ng_ap)
        else:
            mulit_class_ap = ClassifierEvalMulticlass.compute_ap(y_true, y_score)
            confusion_matrix = metrics.confusion_matrix(y_true, np.argmax(y_score, 1))

            save_path = f'./checkpoints/{config.model["name"]}'
            epoch_index = epoch + 1
            save_name = f'{epoch_index:03d}_acc_{val_accuracy:.4f}.pth'
            save_names.append(save_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model, f'{save_path}/{save_name}')

            print(val_accuracy)
            print(mulit_class_ap)
            print(confusion_matrix)

    if log_flag:
        cur_time = time.strftime('%m%d_%H_%M')
        log_file_name = f"{config.model['name']}_{cur_time}.txt"
        utils.write_log(log_file_name, best_FOR_epoch, best_acc_epoch, best_ap_epoch, save_names)


@torch.no_grad()
def val(model, dataloader):
    model.eval()

    correct = 0
    total = 0

    y_true = np.array([])
    y_score = np.zeros(shape=(1, config.num_class))

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
