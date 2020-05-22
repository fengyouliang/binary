import os

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm


def emsemble_pred(model_list, image):
    tf = transforms.Compose([
        lambda x: Image.open(x),
        transforms.Resize((112, 224)),
        transforms.ToTensor(),
    ])
    roi = tf(image)
    roi = roi.to('cuda')
    roi = roi.unsqueeze(0)

    scores = []
    for model in model_list:
        logits = model(roi)
        torch.cuda.empty_cache()
        scores.append(logits.data.cpu().numpy())

    scores = np.array(scores)
    mean_scores = scores.mean(axis=0)
    return mean_scores


def ng_process(model_list):
    image_path = '/mnt/tmp/feng/kuozhankuang/fold_1/test/ng'
    ng_ensemble_scores = []
    pbar = tqdm(os.listdir(image_path))
    for image in pbar:
        score = emsemble_pred(model_list, f'{image_path}/{image}')
        ng_ensemble_scores.append(score)
    ensemble_scores = np.array(ng_ensemble_scores)
    ensemble_scores = ensemble_scores.squeeze()
    ensemble_scores = torch.tensor(ensemble_scores)
    pred = F.softmax(ensemble_scores, dim=1)
    ng_score = pred[:, 0]
    sort_res = ng_score.sort()
    thres_at_1 = sort_res[0][0]
    thres_at_995 = sort_res[0][2]

    return [thres_at_1, thres_at_995]


def ok_process(model_list):
    image_path = '/mnt/tmp/feng/kuozhankuang/fold_1/test/ok'
    ok_ensemble_scores = []
    pbar = tqdm(os.listdir(image_path))
    for image in pbar:
        score = emsemble_pred(model_list, f'{image_path}/{image}')
        ok_ensemble_scores.append(score)
    ensemble_scores = np.array(ok_ensemble_scores)
    ensemble_scores = ensemble_scores.squeeze()
    ensemble_scores = torch.tensor(ensemble_scores)
    pred = F.softmax(ensemble_scores, dim=1)
    ng_score = pred[:, 0]

    return ng_score


def demo():
    device_ids = [3]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))

    model_list = []
    pths = [
        './checkpoints/EfficientNet/28_acc_0.9769_mAP_0.978_FOR_0.2415.pth',  # 0.5656565656565656 0.5909090909090909
        './checkpoints/mobilenet/20_acc_0.9806_mAP_0.9823999999999999_FOR_0.5975.pth',  # 0.17676767676767677 0.6111111111111112
        './checkpoints/resnext101_32x8d/28_acc_0.9891_mAP_0.9919_FOR_0.1761.pth',  # 0.4696969696969697 0.5505050505050505
        './checkpoints/mobilenet/21_acc_0.9037_mAP_0.8948499999999999_FOR_0.3932.pth',  # 0.4292929292929293 0.48484848484848486
        # './checkpoints/mobilenet/12_acc_0.9461_mAP_0.951_FOR_0.4693.pth',  # 0.06565656565656566 0.35353535353535354
        # './checkpoints/mobilenet/19_acc_0.9806_mAP_0.98035_FOR_0.5821.pth',  # 0.03535353535353535 0.2828282828282828

    ]
    for pth in pths:
        model = torch.load(pth)
        model = model.module
        model_list.append(model)

    thres_at_1, thres_at_995 = ng_process(model_list)
    ng_score = ok_process(model_list)

    sum_1 = float(sum(ng_score < thres_at_1))
    sum_995 = float(sum(ng_score < thres_at_995))
    print(sum_1 / len(ng_score))
    print(sum_995 / len(ng_score))


if __name__ == '__main__':
    # thres = 1.4583e-07
    demo()
