import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


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
