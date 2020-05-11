import math
import os
import shutil

import numpy as np

import config
from MetricEval import ClassifierEvalBinary


def adjust_lr(optimizer, iter, gamma=0.5, warm_up=3):
    print(iter)
    original_lr = config.lr
    if iter < warm_up:
        new_lr = original_lr * (iter + 1) / warm_up
    else:
        new_lr = original_lr * gamma * (1 + math.cos(iter / config.max_epoch * math.pi))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def movefile(index, mode='ok'):
    in_file_path = f'/mnt/tmp/feng/second_final_data/final_data_fold_{index}/train/{mode}'
    out_file_path = f'/mnt/tmp/feng/second_final_data/final_data_fold_{index}/val/{mode}'
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)
    for file in os.listdir(in_file_path):
        file_path = f'{in_file_path}/{file}'
        p = np.random.rand(1)[0]
        if p > 0.8:
            shutil.move(file_path, f'{out_file_path}/{file}')


def check_move_file():
    for index in [1, 2, 3]:
        for mode in ['ok', 'ng']:
            in_file_path = f'/mnt/tmp/feng/second_final_data/final_data_fold_{index}/train/{mode}'
            out_file_path = f'/mnt/tmp/feng/second_final_data/final_data_fold_{index}/val/{mode}'
            in_length = len(os.listdir(in_file_path))
            out_length = len(os.listdir(out_file_path))
            print(in_length / (in_length + out_length))


def get_FRN_metric(y_true, y_score):
    threshold = y_score[y_true == 0].min(axis=0)[0]

    true_ok_ng_score = y_score[y_true == 1][:, 0]
    not_ok = true_ok_ng_score > threshold

    true_ok_length = len(y_true[y_true == 1])
    not_ok_length = len(not_ok[not_ok == True])

    final_metric_dict = {
        'threshold': threshold,
        'true=ok@pred=not_ok': not_ok_length,
        'true=ok@pred=ok': true_ok_length - not_ok_length,
        'all_ok': true_ok_length,
        'FNR': (true_ok_length - not_ok_length) / true_ok_length,
    }
    return final_metric_dict


def get_AP_metric(y_true, y_score):
    ok_y_score = y_score[:, 1]
    ok_val_ap = ClassifierEvalBinary.compute_ap(y_true, ok_y_score)

    ng_y_true = np.array(y_true).astype("bool")
    ng_y_true = (1 - ng_y_true).astype(np.int)
    ng_y_score = y_score[:, 0]
    ng_val_ap = ClassifierEvalBinary.compute_ap(ng_y_true, ng_y_score)

    mAP = (ok_val_ap + ng_val_ap) / 2

    return ok_val_ap, ng_val_ap, mAP


if __name__ == '__main__':
    pass
