import os

import numpy as np
import torch
from PIL import Image
from sklearn import metrics
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
import utils
from MetricEval import ClassifierEvalBinary, ClassifierEvalMulticlass
from grad_cam import grad_cam_bad_case
from my_cam import cam_bad_case


class Test_Dataset(Dataset):
    def __init__(self, type_mode='test', reszie=config.resize, transform=None):
        super(Test_Dataset, self).__init__()

        self.resize = reszie
        self.transform = transform
        self.type_mode = type_mode

        self.root = config.test_path

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

        label = torch.tensor(label)

        return img, label


@torch.no_grad()
def final_test(model):
    model.eval()

    correct = 0
    total = 0

    y_true = np.array([])
    y_score = np.zeros(shape=(1, 2))

    all_pred = []  # y_score

    test_data = Test_Dataset()
    test_batch_size = len(test_data) if config.test_batch_size == -1 else config.test_batch_size
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    image_name = test_data.pos_images + test_data.neg_images
    all_image_name = np.array(image_name, dtype=str)

    for x, y in tqdm(test_loader):
        x, y = x.cuda(), y.cuda()

        output = model(x)
        pred = output.argmax(dim=1)
        all_pred.extend(pred.data.cpu().numpy().tolist())
        _, predicted = torch.max(output.data, 1)

        softmax = nn.functional.softmax
        s_pred = softmax(output, dim=1)

        y_true = np.append(y_true, y.data.cpu().numpy())
        y_score = np.concatenate((y_score, s_pred.data.cpu().numpy()), axis=0)

        total += y.size(0)
        correct += (predicted == y).sum().item()

    assert y_true.shape[0] == y_score.shape[0] - 1
    y_score = y_score[1:]

    confusion_matrix = metrics.confusion_matrix(y_true, all_pred)

    final_metric_dict = utils.get_FOR_metric(y_true, y_score)

    # 按score 排序，分NG， OK
    ng_score = y_score[:, 0]
    ok_score = y_score[:, 1]
    ng_score = ng_score[y_true == 0]
    ok_score = ok_score[y_true == 1]
    ng_name = all_image_name[y_true == 0]
    ok_name = all_image_name[y_true == 1]

    ng_idx = ng_score.argsort()
    ok_idx = ok_score.argsort()
    ng_score = ng_score[ng_idx]
    ok_score = ok_score[ok_idx]
    ng_name = ng_name[ng_idx]
    ok_name = ok_name[ok_idx]

    vis_bad_case = {
        'ng_score': ng_score,
        'ok_score': ok_score,
        'ng_name': ng_name,
        'ok_name': ok_name,
    }

    mulit_class_ap = ClassifierEvalMulticlass.compute_ap(y_true, y_score)

    ok_y_score = y_score[:, 1]
    ok_val_ap = ClassifierEvalBinary.compute_ap(y_true, ok_y_score)
    ok_p_at_r = ClassifierEvalBinary.compute_p_at_r(y_true, ok_y_score, 1)
    ClassifierEvalBinary.draw_pr_curve(y_true, ok_y_score, cls_id=1, output_path='./pic/ok_pr_curve.png')

    ng_y_true = np.array(y_true).astype("bool")
    ng_y_true = (1 - ng_y_true).astype(np.int)
    ng_y_score = y_score[:, 0]
    ng_val_ap = ClassifierEvalBinary.compute_ap(ng_y_true, ng_y_score)
    ng_p_at_r = ClassifierEvalBinary.compute_p_at_r(ng_y_true, ng_y_score, 1)
    ClassifierEvalBinary.draw_pr_curve(ng_y_true, ng_y_score, cls_id=0, output_path='./pic/ng_pr_curve.png')

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
        # 'vis_ng_score': vis_ng_score,
        'vis_bad_case': vis_bad_case
    }

    return ret_dict


def vis_bad_case(vis_bad_case, fold_idx):
    import cv2 as cv
    import shutil

    # ng_score = vis_bad_case['ng_score']
    # ok_score = vis_bad_case['ok_score']
    # ng_name = vis_bad_case['ng_name']
    # ok_name = vis_bad_case['ok_name']

    v = {
        'ok': {
            'score': vis_bad_case['ok_score'],
            'name': vis_bad_case['ok_name'],
        },
        'ng': {
            'score': vis_bad_case['ng_score'],
            'name': vis_bad_case['ng_name'],
        }
    }
    for mode in v.keys():
        output_path = f'./vis_bad_case/fold_{fold_idx}/{mode}'
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

    for mode in v.keys():
        score, name = v[mode]['score'], v[mode]['name']
        length = len(score[score < 0.5])

        for i in range(length):
            image = cv.imread(name[i])
            output_name = name[i].split('/')[-1].split('.')[0]
            cv.imwrite(f'./vis_bad_case/fold_{fold_idx}/{mode}/{score[i]:6f}_{output_name}.jpg', image)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.device_ids))

    model = utils.load_model(config.test_pth, config.cuda_available_index)
    ret_dict = final_test(model)
    torch.cuda.empty_cache()

    print(ret_dict)
    vis_bad_case(ret_dict["vis_bad_case"], config.test_path[-1])
    grad_cam_bad_case(config.data_fold_index)
    # cam_bad_case(output_type=config.data_fold_index)
