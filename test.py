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
from models.model import registry_model


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


def load_model(pth):
    model = registry_model.get(config.model)()
    model.load(pth)
    model = model.cuda(device=config.device_ids[0])
    return model


@torch.no_grad()
def final_test(model):
    model.eval()

    correct = 0
    total = 0

    y_true = np.array([])
    y_score = np.zeros(shape=(1, 2))

    all_pred = []  # y_score

    test_data = Test_Dataset()
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    ng_image_name = np.array(test_data.neg_images, dtype=str)

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

    final_metric_dict = utils.get_FRN_metric(y_true, y_score)

    # 求真实类别为NG最低K个的score
    true_ok_length = len(y_true[y_true == 1])
    true_ng_ng_score = y_score[true_ok_length:, 0]
    idx = true_ng_ng_score.argsort()
    true_ng_ng_score = true_ng_ng_score[idx]
    ng_image_name = ng_image_name[idx]

    vis_ng_score = {
        'idx': idx,
        'true_ng_ng_score': true_ng_ng_score,
        'ng_image_name': ng_image_name,
    }

    mulit_class_ap = ClassifierEvalMulticlass.compute_ap(y_true, y_score)

    ok_y_score = y_score[:, 1]
    ok_val_ap = ClassifierEvalBinary.compute_ap(y_true, ok_y_score)
    ok_p_at_r = ClassifierEvalBinary.compute_p_at_r(y_true, ok_y_score, 1)
    # ClassifierEvalBinary.draw_pr_curve(y_true, ok_y_score, cls_id=1, output_path='./ok_pr_curve.png')

    ng_y_true = np.array(y_true).astype("bool")
    ng_y_true = (1 - ng_y_true).astype(np.int)
    ng_y_score = y_score[:, 0]
    ng_val_ap = ClassifierEvalBinary.compute_ap(ng_y_true, ng_y_score)
    ng_p_at_r = ClassifierEvalBinary.compute_p_at_r(ng_y_true, ng_y_score, 1)
    # ClassifierEvalBinary.draw_pr_curve(ng_y_true, ng_y_score, cls_id=0, output_path='./ng_pr_curve.png')

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


if __name__ == '__main__':
    for pth in os.listdir('best_ap_model'):
        # pth = '26_acc_0.9051_ok_ap_0.9595_ng_ap_0.9976_mAP_0.97855.pth'
        model = load_model(f'best_ap_model/{pth}')
        ret_dict = final_test(model)
        print(ret_dict)
        print(pth)
        print(ret_dict['confusion_matrix'])
        print(ret_dict['final_metric_dict'])
