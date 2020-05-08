#!/usr/bin/env python
# coding: utf-8

import math
import os
import random
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as t
import torchvision.models
from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn import metrics
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

data_path = '/mnt/tmp/feng/final_data_1'

device_ids = [0, 2]
# device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

resize = (112, 224)

train_keep = -1
val_keep = 150

batch_size = 512

is_pretrained = True

lr = 2e-4
weight_decay = 0
momentum = 0

max_epoch = 100
best_val_acc = 0.5


class BasicModule(t.nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay, momentum):
        return t.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


class Flat(t.nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class mobilenet(BasicModule):
    def __init__(self, num_classes=2):
        super(mobilenet, self).__init__()
        self.model_name = 'mobilenet'

        net = torchvision.models.mobilenet_v2(pretrained=is_pretrained, num_classes=1000)

        self.features = nn.Sequential(
            *list(net.children())[:-1],
        )
        self.classifer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_max_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


class efficientnet(BasicModule):
    def __init__(self, net_type='efficientnet-b0', num_classes=2):
        super(efficientnet, self).__init__()
        self.model_name = 'EfficientNet'

        model = EfficientNet.from_pretrained(net_type)
        self.features = model.extract_features
        self.classifer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifer(x)
        return x


class MyDataset(Dataset):
    def __init__(self, type_mode, reszie=resize, transform=None):
        super(MyDataset, self).__init__()

        self.resize = reszie
        self.transform = transform
        self.type_mode = type_mode

        self.root = data_path

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
            pos_images = pos_images[:train_keep]
            pos_label = pos_label[:train_keep]
        elif self.type_mode == 'val':
            random.seed(42)
            random.shuffle(pos_images)
            random.seed(42)
            random.shuffle(pos_label)
            pos_images = pos_images[:val_keep]
            pos_label = pos_label[:val_keep]

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

        label = t.tensor(label)

        return img, label


def adjust_lr(op, iter, gamma=0.5, warm_up=3):
    original_lr = lr
    if iter < warm_up:
        new_lr = original_lr * (iter + 1) / warm_up
    else:
        new_lr = original_lr * gamma * (1 + math.cos(iter / max_epoch * math.pi))
    for para in op.param_groups:
        para['lr'] = new_lr


if not os.path.exists('./best_model'):
    os.makedirs('./best_model')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

if not os.path.exists('./best_acc_model'):
    os.makedirs('./best_acc_model')
if not os.path.exists('./best_ap_model'):
    os.makedirs('./best_ap_model')


class ClassifierEval(object):
    """A class to evaluate the performance of a classifier."""

    @classmethod
    def compute_ap(cls):
        """Compute AP of each class."""
        pass

    @classmethod
    def compute_p_at_r(cls):
        """Compute precision at (Recall >= referred_recall)."""
        pass

    @classmethod
    def draw_pr_curve(cls):
        """Draw precision-recall curve."""
        pass

    @classmethod
    def compute_fnr_and_fpr(cls):
        """Refer to https://en.wikipedia.org/wiki/Receiver_operating_characteristic for detailed calculation
        fnr = 1 - tpr = fn / (tp + fn), miss rate （漏报率）
        fpr = fp / (tn + fp), false alarm rate （误报率）
        """
        pass


class ClassifierEvalBinary(ClassifierEval):
    """二分类评估器。"""

    @classmethod
    def compute_ap(cls, y_true, y_score):
        """计算二分类的AP 
        Args:
            y_true: 1维nparray，所有样本的真值列表[nSamples], e.g., [0, 1, 1, 0], 1为ng，0为ok
            y_score: 1维nparray，所有样本在目标类别上的分数, 注意该分数是最后一层（softmax层）的输出结果。e.g., [0.2, 0.9, 0.4, 0.5]
        Return:
            一个数字：AP值。
        """
        ap = metrics.average_precision_score(y_true=y_true, y_score=y_score)
        return round(ap, 4)

    @classmethod
    def compute_p_at_r(cls, y_true, y_score, recall_thresh=0.995):
        """计算二分类中recall >= 0.995时的precision。
        Args:
            y_true: 1维nparray，所有样本的真值列表[nSamples], 以0和1形式给出， e.g., [0, 0, 1, 0]
            y_score: 1维nparray，所有样本在正类上的分数, e.g., [0.2, 0.9, 0.7, 0.1]
        Return:
            Precision@(Recall>=recall_thresh)
        """
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        ind_all = np.where(recall < recall_thresh)
        ind = ind_all[0][0] - 1
        return round(precision[ind], 4)

    @classmethod
    def draw_pr_curve(cls, y_true, y_score, cls_id=0, output_path='./pr_curve.png'):
        """画二分类的pr曲线。
        Args:
            y_true: 1维nparray，所有样本的真值列表[nSamples], 以0和1形式给出， e.g., [0, 0, 1, 0]
            y_score: 1维nparray，所有样本在正类上的分数, e.g., [0.2, 0.9, 0.7, 0.1]
        Return:
            无
        """
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)

        # draw pr curve
        plt.figure()
        plt.step(recall, precision, where='pre')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('PR Curve of Cls {:d}'.format(cls_id))
        plt.savefig(output_path)
        plt.close()

    @classmethod
    def compute_fnr_and_fpr(cls, y_true, y_score, fnr_thresh=0.005, fail_study=False):
        """计算二分类中的漏报率和误报率，要求漏报率必须小于等于fnr_thresh。
        Args:
            y_true: 1维nparray，所有样本的真值列表[nSamples], 以0和1形式给出， e.g., [0, 0, 1, 0]
            y_score: 1维nparray，所有样本在正类上的分数, e.g., [0.2, 0.9, 0.7, 0.1]
            fnr_thresh: 能容忍的最大的漏报率
            fail_study: True时返回失败案例的索引，含漏报和误报的。False时返回空list。
        Return:
            min_score: 判断为ng的最小score，ng类别的阈值大于等于该score则判断为ng，否则判断为ok 
            res: 含fnr和fpr
            fn_index_list: fail_study为False时为[]，否则为漏报的图片的index list
            fp_index_list: fail_study为False时为[]，否则为误报的图片的index list
        """
        fn_index_list, fp_index_list = [], []
        _, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
        ind_all = np.where(recall < (1 - fnr_thresh))
        ind = ind_all[0][0] - 1
        min_score = thresholds[ind]
        y_pred = [int(per_score >= min_score) for per_score in y_score]
        tp, fp, tn, fn = 0, 0, 0, 0
        if fail_study:
            for i, (per_true, per_pred) in enumerate(zip(y_true, y_pred)):
                if per_true == 1 and per_pred == 1:
                    tp += 1
                elif per_true == 1 and per_pred == 0:
                    fn += 1
                    fn_index_list.append(i)
                elif per_true == 0 and per_pred == 1:
                    fp += 1
                    fp_index_list.append(i)
                elif per_true == 0 and per_pred == 0:
                    tn += 1
        else:
            for per_true, per_pred in zip(y_true, y_pred):
                if per_true == 1 and per_pred == 1:
                    tp += 1
                elif per_true == 1 and per_pred == 0:
                    fn += 1
                elif per_true == 0 and per_pred == 1:
                    fp += 1
                elif per_true == 0 and per_pred == 0:
                    tn += 1
        fnr = fn / (tp + fn)
        fpr = fp / (tn + fp)
        res = dict()
        res['fnr'] = round(fnr, 4)
        res['fpr'] = round(fpr, 4)
        return min_score, res, fn_index_list, fp_index_list

    @classmethod
    def draw_failure_cases(cls, img_path_list, y_true, y_score, min_score, fn_index_list, fp_index_list, cls_dict,
                           res_dir=None):
        """保存failure cases，含漏报和误报的。
        Args:
            img_path_list: 图片路径
            y_true: 1维nparray，所有样本的真值列表[nSamples], 以0和1形式给出， e.g., [0, 0, 1, 0]
            y_score: 1维nparray，所有样本在正类上的分数, e.g., [0.2, 0.9, 0.7, 0.1]
            min_score: 判断为ng的最小score值
            fn_index_list: 漏报的样本索引
            fp_index_list: 误报的样本索引
            cls_dict: 类别字典，用于做failure case的显示用，如{0:'OK', 1:'NG'}
            res_dir: failure case存放路径
        Return:
            无
        """
        # Step 1. Make dirs.
        if not res_dir:
            raise ValueError('Result directory error!')
        res_dir_fn = os.path.join(res_dir, 'loubao')
        res_dir_fp = os.path.join(res_dir, 'wubao')
        if not os.path.exists(res_dir_fn):
            os.makedirs(res_dir_fn)
        if not os.path.exists(res_dir_fp):
            os.makedirs(res_dir_fp)

        # Step 2. Get dt_label_list.
        dt_label_list = []
        for per_score in y_score:
            if per_score >= min_score:
                dt_label_list.append(1)
            else:
                dt_label_list.append(0)

        # Step 3. Draw failures.
        if not fn_index_list:
            print('fn_index_list is empty!')
        else:
            for ind in fn_index_list:
                gt_label = y_true[ind]
                dt_label = dt_label_list[ind]
                img = mpimg.imread(img_path_list[ind])
                plt.imshow(img)
                plt.title('cls: {} -> cls: {}'.format(cls_dict[gt_label], cls_dict[dt_label]))
                plt.axis('off')
                output_path = os.path.join(res_dir_fn, os.path.basename(img_path_list[ind]))
                output_path_new = output_path.replace('.jpg', '.png')
                plt.savefig(output_path_new)
                plt.close()

        if not fp_index_list:
            print('fp_index_list is empty!')
        else:
            for ind in fp_index_list:
                gt_label = y_true[ind]
                dt_label = dt_label_list[ind]
                img = mpimg.imread(img_path_list[ind])
                plt.imshow(img)
                plt.title('cls: {} -> cls: {}'.format(cls_dict[gt_label], cls_dict[dt_label]))
                plt.axis('off')
                output_path = os.path.join(res_dir_fp, os.path.basename(img_path_list[ind]))
                output_path_new = output_path.replace('.jpg', '.png')
                plt.savefig(output_path_new)
                plt.close()


def label_to_onehot(label_list, nb_classes):
    """将一个2维的label index list转为2维的onehot array 
    Args:
        label_list: 一个2维的label list, e.g., [[2], [0, 1], [1]]
        nb_classes: 类别数量, e.g., 3
    Return:
        一个二维的array, e.g., np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    """
    res_arr = np.zeros([len(label_list), nb_classes])
    for i, label in enumerate(label_list):
        res_arr[i][int(label)] = 1
    return res_arr


def onehot_to_label(onehot_array):
    """将一个2维的onehot array 转为1维的label list， 支持多标签形式
    Args:
        一个二维的array, e.g., [[0, 0, 1], [1, 0, 0], [0, 1, 1]]
    Return:
        label_list: 一个2维的label list, e.g., [[2], [0], [1, 2]]
    """
    label_list = []
    for arr in onehot_array:
        label_list.append(list(np.where(arr == 1)[0]))
    return label_list


class ClassifierEvalMulticlass(ClassifierEval):
    """多类别分类评估器。"""

    @classmethod
    def compute_ap(cls, y_true, y_score):
        """计算多类别分类的AP 
        Args:
            y_true: 2维nparray，所有样本的真值列表[nSamples, nClasses], e.g., [2, 0]
            y_score: 2维nparray，所有样本在各个类别上的分数, 注意每一个样本在各个类别上的分数之和须为1， e.g., [[0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.3, 0.5], ...]
        Return:
            ap_list: 各个类别的AP值组成的list
        """
        # 1. 将类别转为one_hot的形式
        nb_classes = y_score.shape[1]
        y_true = label_to_onehot(y_true, nb_classes)
        # 2. 计算各个类别的ap
        ap_dict = dict()
        for i, (true_per_cls, score_per_cls) in enumerate(zip(y_true.T, y_score.T)):
            ap = metrics.average_precision_score(
                y_true=true_per_cls, y_score=score_per_cls)
            ap_dict[i] = round(ap, 4)
        return ap_dict

    @classmethod
    def compute_p_at_r(cls, y_true, y_score, recall_thresh=0.995):
        """计算多分类中recall >= 0.995时各个类别的precision。
        Args:
            y_true: 1维nparray，所有样本的真值列表[nSamples], e.g., [2, 0, 3]
            y_score: 1维nparray，所有样本在各个类上的分数, e.g., [[0.2, 0.7, 0.1, 0], [0.1, 0.3, 0.5, 0.1], [0.4, 0.2, 0.1, 0.3]]
        Return:
            每个类别的Precision@(Recall>=recall_thresh)组成的dict， 如{0: 0.2345, 1： 0.2344， 2： 0.7623， 3： 0.3334}
        """
        label_list = sorted(list(set(y_true)))  # label从小到大排序
        prec_dict = dict()
        for i, label in enumerate(label_list):
            y_true_per_cls = np.array([int(lab == label) for lab in y_true])
            y_score_per_cls = y_score[:, i]
            precision = ClassifierEvalBinary.compute_p_at_r(
                y_true_per_cls, y_score_per_cls, recall_thresh=recall_thresh)
            prec_dict[label] = round(precision, 4)  # 保留4位小数
        return prec_dict

    @classmethod
    def draw_pr_curve(cls, y_true, y_score, output_dir='hehe'):
        """画多类别分类的pr曲线，每个类别一个pr曲线。
        Args:
            y_true: 1维nparray，所有样本的真值列表[nSamples], e.g., [2, 0, 3]
            y_score: 1维nparray，所有样本在各个类上的分数, e.g., [[0.2, 0.7, 0.1, 0], [0.1, 0.3, 0.5, 0.1], [0.4, 0.2, 0.1, 0.3]]
        Return:
            无
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        label_list = sorted(list(set(y_true)))  # label从小到大排序
        for i, label in enumerate(label_list):
            y_true_per_cls = np.array([int(lab == label) for lab in y_true])
            y_score_per_cls = y_score[:, i]
            output_path = os.path.join(output_dir, 'cls_' + str(label))
            ClassifierEvalBinary.draw_pr_curve(
                y_true_per_cls, y_score_per_cls, cls_id=int(i), output_path=output_path)

    @classmethod
    def compute_fnr_and_fpr(cls, y_true, y_score, ok_ind=0, fnr_thresh=0.005, fail_study=False):
        """计算多类别分类中的漏报率和误报率，要求漏报率必须小于等于fnr_thresh。
        Args:
            y_true: 1维nparray，所有样本的真值列表[nSamples], e.g., [2, 3, 1, 0]
            y_score: 2维nparray，所有样本在各个类上的分数, e.g., [[0.2, 0.7, 0.1, 0], [0.1, 0.3, 0.5, 0.1], [0.4, 0.2, 0.1, 0.3], [0.3, 0.3, 0.4, 0]]
            ok_ind: ok类别的index，默认0为ok，其余为ng 
            fnr_thresh: 能容忍的最大的漏报率
            fail_study: True时返回失败案例的索引，含漏报和误报的。False时返回空list。
        Return:
            min_score: 判断为ng的最小score，ng类别的阈值大于等于该score则判断为ng，否则判断为ok 
            res: 含fnr和fpr
            fn_index_list: fail_study为False时为[]，否则为漏报的图片的index list
            fp_index_list: fail_study为False时为[]，否则为误报的图片的index list
        """
        y_true_new = []
        y_score_new = []
        for per_true, per_score in zip(y_true, y_score):
            label = 0 if per_true == ok_ind else 1
            per_score_new = per_score.copy()
            # 将ok类score置为0,再取最大score，该score为ng类别的最大score。
            per_score_new[ok_ind] = 0
            score = np.max(per_score_new)
            y_true_new.append(label)
            y_score_new.append(score)
        min_score, res, fn_index_list, fp_index_list = ClassifierEvalBinary.compute_fnr_and_fpr(
            np.array(y_true_new), np.array(y_score_new), fnr_thresh, fail_study)
        return min_score, res, fn_index_list, fp_index_list

    @classmethod
    def draw_failure_cases(cls, ok_ind, img_path_list, y_true, y_score, min_score, fn_index_list, fp_index_list,
                           cls_dict, res_dir=None):
        """保存failure cases，含漏报和误报的。
        Args:
            ok_ind: ok类别的index
            img_path_list: 图片路径
            y_true: 1维nparray，所有样本的真值列表[nSamples], e.g., [2, 3, 1, 0]
            y_score: 2维nparray，所有样本在各个类上的分数, e.g., [[0.2, 0.7, 0.1, 0], [0.1, 0.3, 0.5, 0.1], [0.4, 0.2, 0.1, 0.3], [0.3, 0.3, 0.4, 0]]
            min_score: 判为ng的最小score
            fn_index_list: 漏报的样本索引
            fp_index_list: 误报的样本索引
            cls_dict: 类别字典，用于显示failure case，如cls_dict={0:'ok', 1:'0', 2:'1', 3:'2', 4:'3', 5:'8'}
            res_dir: failure case存放路径
        Return:
            无
        """
        # Step 1. Make dirs.
        if not res_dir:
            raise ValueError('Result directory error!')
        res_dir_fn = os.path.join(res_dir, 'loubao')
        res_dir_fp = os.path.join(res_dir, 'wubao')
        if not os.path.exists(res_dir_fn):
            os.makedirs(res_dir_fn)
        if not os.path.exists(res_dir_fp):
            os.makedirs(res_dir_fp)

        # Step 2. Get dt_label_list.
        dt_label_list = []
        for per_score in y_score:
            per_score_new = per_score.copy()
            # 将ok类score置为0,再取最大score，该score为ng类别的最大score。
            per_score_new[ok_ind] = 0
            score = np.max(per_score_new)
            if score >= min_score:
                ind = np.argmax(per_score_new)
            else:
                ind = ok_ind
            dt_label_list.append(ind)

        # Step 3. Draw failures.
        if not fn_index_list:
            print('fn_index_list is empty!')
        else:
            for ind in fn_index_list:
                gt_label = y_true[ind]
                dt_label = dt_label_list[ind]
                img = mpimg.imread(img_path_list[ind])
                plt.imshow(img)
                plt.title('cls: {} -> cls: {}'.format(cls_dict[gt_label], cls_dict[dt_label]))
                plt.axis('off')
                output_path = os.path.join(res_dir_fn, os.path.basename(img_path_list[ind]))
                output_path_new = output_path.replace('.jpg', '.png')
                plt.savefig(output_path_new)
                plt.close()

        if not fp_index_list:
            print('fp_index_list is empty!')
        else:
            for ind in fp_index_list:
                gt_label = y_true[ind]
                dt_label = dt_label_list[ind]
                img = mpimg.imread(img_path_list[ind])
                plt.imshow(img)
                plt.title('cls: {} -> cls: {}'.format(cls_dict[gt_label], cls_dict[dt_label]))
                plt.axis('off')
                output_path = os.path.join(res_dir_fp, os.path.basename(img_path_list[ind]))
                output_path_new = output_path.replace('.jpg', '.png')
                plt.savefig(output_path_new)
                plt.close()


def train(load_model_path=None):
    model = mobilenet()
    if load_model_path is not None:
        model.load(load_model_path)

    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])
    # model.to(device)

    train_data = MyDataset('train')
    val_data = MyDataset('val')

    if train_keep == -1:
        ok_count = len(os.listdir(f"{data_path}/train/ok/")) - 1
    else:
        ok_count = train_keep

    ng_count = len(os.listdir(f"{data_path}/train/ng/"))

    weights = [1 / ok_count for i in range(ok_count)] + [1 / ng_count for i in range(ng_count)]
    sampler = WeightedRandomSampler(weights, len(train_data), True)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    criterion = t.nn.CrossEntropyLoss()

    optimizer = model.module.get_optimizer(lr, weight_decay, momentum)

    best_acc = best_val_acc
    best_ok_ap = 0
    best_ng_ap = 0
    best_ap = 0

    for epoch in range(max_epoch):

        adjust_lr(optimizer, epoch)

        bar = tqdm(enumerate(train_loader), total=math.ceil(len(train_data) / batch_size))
        for ii, (data, label) in bar:
            input = data.cuda()
            target = label.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            bar.set_description(f'{epoch}-{ii} loss:{loss.item():.4f}')

        model.module.save()

        val_accuracy, y_true, y_score = val(model, val_loader)

        ok_y_score = y_score[:, 1]
        ok_val_ap = ClassifierEvalBinary.compute_ap(y_true, ok_y_score)

        ng_y_true = np.array(y_true).astype("bool")
        ng_y_true = (1 - ng_y_true).astype(np.int)
        ng_y_score = y_score[:, 0]
        ng_val_ap = ClassifierEvalBinary.compute_ap(ng_y_true, ng_y_score)

        mAP = (ok_val_ap + ng_val_ap) / 2

        # mulit_class_ap = ClassifierEvalMulticlass.compute_ap(y_true, y_score)

        threshold = y_score[y_true == 0].min(axis=0)[0]
        true_ok_ng_score = y_score[y_true == 1][:, 0]
        not_ok = true_ok_ng_score > threshold

        true_ok_length = len(y_true[y_true == 1])
        not_ok_length = len(not_ok[not_ok == False])

        final_metric_dict = {
            'threshold': threshold,
            'true=ok@pred=not_ok': not_ok_length,
            'true=ok@pred=ok': true_ok_length - not_ok_length,
            'all_ok': true_ok_length,
            'FNR': (true_ok_length - not_ok_length) / true_ok_length,
        }

        print(f'Acc: {val_accuracy:.2f}\t OK_AP：{ok_val_ap:.2f}\t NG_AP: {ng_val_ap:.2f}\t mAP: {mAP:.2f}')
        print(f'BEST Acc: {best_acc:.2f}\t OK_AP: {best_ok_ap:.2f}\t NG_AP: {best_ng_ap:.2f}\t mAP: {best_ap:.2f}')
        print(final_metric_dict)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            model.module.save(
                f'best_acc_model/{epoch}_acc_{val_accuracy:.4f}_ok_ap_{ok_val_ap:.4f}_ng_ap_{ng_val_ap}_mAP_{mAP}.pth')
        if mAP > best_ap:
            best_ap = mAP
            model.module.save(
                f'best_ap_model/{epoch}_acc_{val_accuracy:.4f}_ok_ap_{ok_val_ap:.4f}_ng_ap_{ng_val_ap}_mAP_{mAP}.pth')
        best_ok_ap = max(ok_val_ap, best_ok_ap)
        best_ng_ap = max(ng_val_ap, best_ng_ap)


@t.no_grad()
def val(model, dataloader):
    model.eval()

    correct = 0
    total = 0

    y_true = np.array([])
    y_score = np.zeros(shape=(1, 2))

    for x, y in tqdm(dataloader):
        # x, y = x.to(device), y.to(device)
        x = x.cuda()
        y = y.cuda()

        output = model(x)
        _, predicted = t.max(output.data, 1)

        softmax = t.nn.functional.softmax
        s_pred = softmax(output, dim=1)
        # print('s_pred: ', s_pred)

        y_true = np.append(y_true, y.data.cpu().numpy())
        y_score = np.concatenate((y_score, s_pred.data.cpu().numpy()), axis=0)

        total += y.size(0)
        correct += (predicted == y).sum().item()

    assert y_true.shape[0] == y_score.shape[0] - 1

    return correct / total, y_true, y_score[1:]


class Test_Dataset(Dataset):
    def __init__(self, type_mode, reszie=resize, transform=None):
        super(Test_Dataset, self).__init__()

        self.resize = reszie
        self.transform = transform
        self.type_mode = type_mode

        self.root = data_path

        self.pos = f'{self.root}/{type_mode}/ok/'
        self.neg = f'{self.root}/{type_mode}/ng/'

        self.pos_images = [self.pos + file for file in os.listdir(self.pos)]
        pos_label = [1 for i in range(len(self.pos_images))]
        self.neg_images = [self.neg + file for file in os.listdir(self.neg)]
        neg_label = [0 for i in range(len(self.neg_images))]

        self.all_image = self.pos_images + self.neg_images
        self.all_label = pos_label + neg_label

    def __len__(self):
        return len(self.all_image)

    def __getitem__(self, index):
        img, label = self.all_image[index], self.all_label[index]
        tf = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Resize((int(self.resize[0]), int(self.resize[1]))),
            transforms.ToTensor(),
        ])
        img = tf(img)

        label = t.tensor(label)

        return img, label


def load_model(pth):
    # model = resnet_18()
    model = mobilenet()
    model.load(pth)
    model = model.cuda(device=device_ids[0])
    return model


@t.no_grad()
def final_test(model):
    model.eval()

    correct = 0
    total = 0

    y_true = np.array([])
    y_score = np.zeros(shape=(1, 2))
    min_y_ng_score = 1

    all_y = []
    all_pred = []

    test_data = Test_Dataset('test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    ng_image_name = np.array(test_data.neg_images, dtype=str)

    for x, y in tqdm(test_loader):
        x, y = x.cuda(), y.cuda()

        output = model(x)

        pred = output.argmax(dim=1)
        all_pred.extend(pred.data.cpu().numpy().tolist())
        _, predicted = t.max(output.data, 1)

        softmax = t.nn.functional.softmax
        s_pred = softmax(output, dim=1)

        all_y.extend(y.data.cpu().numpy().tolist())
        # all_ng_p.extend(s_pred[:, 0].data.cpu().numpy().tolist())

        cur_min_y_ng_score = min(s_pred[:, 0].data.cpu()).item()
        min_y_ng_score = min(min_y_ng_score, cur_min_y_ng_score)

        y_true = np.append(y_true, y.data.cpu().numpy())
        y_score = np.concatenate((y_score, s_pred.data.cpu().numpy()), axis=0)

        total += y.size(0)
        correct += (predicted == y).sum().item()

    assert y_true.shape[0] == y_score.shape[0] - 1
    y_score = y_score[1:]

    from sklearn import metrics
    confusion_matrix = metrics.confusion_matrix(all_y, all_pred)

    true_ok_length = sum(all_y)
    all_y = np.array(all_y)
    all_pred = np.array(all_pred)

    threshold = y_score[true_ok_length:].min(axis=0)[0]
    true_ok_ng_score = y_score[:true_ok_length, 0]
    not_ok = true_ok_ng_score > threshold

    # 求真实类别为NG最低K个的score
    true_ng_ng_score = y_score[true_ok_length:, 0]
    idx = true_ng_ng_score.argsort()
    true_ng_ng_score = true_ng_ng_score[idx]
    ng_image_name = ng_image_name[idx]

    vis_ng_score = {
        'true_ng_ng_score': true_ng_ng_score,
        'idx': idx,
        'true_ng_ng_score': true_ng_ng_score,
        'ng_image_name': ng_image_name,
    }

    # import pdb; pdb.set_trace()

    final_metric_dict = {
        'threshold': threshold,
        'true=ok@pred=not_ok': len(not_ok),
        'true=ok@pred=ok': true_ok_length - sum(not_ok),
        'FNR': (true_ok_length - sum(not_ok)) / true_ok_length,
    }

    threshold = y_score.min(axis=0)[0]
    ok_gt_th = y_score[:true_ok_length, 1] > threshold

    mulit_class_ap = ClassifierEvalMulticlass.compute_ap(y_true, y_score)

    ok_y_score = y_score[:, 1]
    ok_val_ap = ClassifierEvalBinary.compute_ap(y_true, ok_y_score)
    ok_p_at_r = ClassifierEvalBinary.compute_p_at_r(y_true, ok_y_score, 1)
    ClassifierEvalBinary.draw_pr_curve(y_true, ok_y_score, cls_id=1, output_path='./ok_pr_curve.png')

    ng_y_true = np.array(y_true).astype("bool")
    ng_y_true = (1 - ng_y_true).astype(np.int)
    ng_y_score = y_score[:, 0]
    ng_val_ap = ClassifierEvalBinary.compute_ap(ng_y_true, ng_y_score)
    ng_p_at_r = ClassifierEvalBinary.compute_p_at_r(ng_y_true, ng_y_score, 1)
    ClassifierEvalBinary.draw_pr_curve(ng_y_true, ng_y_score, cls_id=0, output_path='./ng_pr_curve.png')

    mAP = (ok_val_ap + ng_val_ap) / 2

    ret_dict = {
        'acc': correct / total,
        'mulit_class_ap': mulit_class_ap,
        'ok_val_ap': ok_val_ap,
        'ng_val_ap': ng_val_ap,
        'mAP': mAP,
        'ok_p_at_r': ok_p_at_r,
        'ng_p_at_r': ng_p_at_r,
        'confusion_matrix': confusion_matrix,
        'final_metric_dict': final_metric_dict,
        'vis_ng_score': vis_ng_score
    }

    return ret_dict


train()

data_path = '/home/feng/final_data_2'
pth = './best_ap_model/64_acc_0.9042_ok_ap_0.9573_ng_ap_0.9719_mAP_0.9646.pth'
model = load_model(pth)

# In[52]:


ret_dict = final_test(model)

# In[53]:


ret_dict

# In[54]:


vis = ret_dict['vis_ng_score']

# In[ ]:


# In[55]:


score = vis['true_ng_ng_score']
name = vis['ng_image_name']

length = len(score[score < 0.5])

for i in range(length):
    print(name[i], score[i])

# In[ ]:


# In[ ]:


# In[30]:


import cv2 as cv
import shutil

score = vis['true_ng_ng_score']
name = vis['ng_image_name']

length = len(score[score < 0.5])

for i in range(length):
    image = cv.imread(name[i])

    output_name = name[i].split('/')[-1].split('.')[0]
    output_path = f'{data_path}/vis_ng'
    shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(f'{output_path}/{score[i]:6f}_{output_name}.jpg', image)

# In[ ]:


# In[ ]:


# In[31]:


log_path = './log/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
with open(f'{log_path}/test_log.txt', 'a+', encoding='utf-8') as f:
    lines = []
    cur_time = time.strftime('%m%d_%H:%M:%S')
    lines.append(f'time: {cur_time}')
    lines.append(f'fold: {data_path[-1]}')
    lines.append(f'pth: {pth}')
    l = [f'{k}: {v}' for k, v in ret_dict.items()]
    lines.append('\t'.join(l))
    lines.append('\n')
    f.writelines('\n'.join(lines))

# In[ ]:


# In[ ]:


# In[56]:


p = 0.53125
ng = 409
ok = 224

# In[57]:


not_ok = (1 - p) / p * ng
not_ok

# In[58]:


ok - not_ok, ok, (ok - not_ok) / ok

# In[ ]:


# In[ ]:


# In[ ]:
