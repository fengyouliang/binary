import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

import config


cpu = torch.device('cpu')
cuda_index = 2
available_cuda = torch.device(f'cuda:{cuda_index}')


def result_test(model):
    pred_res = []
    for mode in os.listdir(f'{config.test_path}/test'):
        y_true = mode
        cur_path = f'{config.test_path}/test/{mode}'
        for file in os.listdir(cur_path):
            image_path = f'{cur_path}/{file}'
            print(image_path)
            tf = transforms.Compose([
                lambda x: Image.open(x),
                transforms.Resize((112, 224)),
                transforms.ToTensor(),
            ])
            image = tf(image_path)
            image = image.to(available_cuda)
            image = image.unsqueeze(0)
            logits = model(image)
            softmax = nn.functional.softmax
            pred = softmax(logits, dim=1)
            pred = pred.data.cpu().numpy()[0]
            pred_cls = logits.argmax(dim=1).data.cpu().numpy()[0]
            pred_cls = 'ok' if pred_cls == 1 else 'ng'
            ng_score, ok_score = pred
            # print(y_true, pred_cls, ng_score, ok_score)
            d = {
                'image_path': image_path,
                'y_true': y_true,
                'ng_score': ng_score,
                'ok_score': ok_score,
                'pred_cls': pred_cls
            }
            pred_res.append(d)
    return pred_res


def load_model():
    """
    一机多卡load
    :return:
    """
    # load to cpu
    # checkpoint = torch.load(config.test_pth, map_location=cpu)
    # model = checkpoint.module

    # load to cuda
    checkpoint = torch.load(config.test_pth, map_location=lambda storage, loc: storage.cuda(cuda_index))
    model = checkpoint.module

    return model


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def to_json(result_pred):
    final_json = {}

    class_dict = [
        {
            "class_name": "OK",
            "class_id": 1
        },
        {
            "class_name": "NG",
            "class_id": 2
        }]

    record = []

    for item in result_pred:
        image_path = item['image_path']
        image_name = os.path.basename(image_path).split('.')[0]
        y_true = item['y_true']
        class_dict_idx = 0 if y_true == 'ok' else 1
        ng_score = item['ng_score']
        # ng_score = f'{ng_score}'
        ok_score = item['ok_score']
        # ok_score = f'{ok_score}'

        sub_record = {
            "pred_inst": [
                {
                    "class_name": "OK",
                    "class_id": 1,
                    "score": ok_score
                },
                {
                    "class_name": "NG",
                    "class_id": 2,
                    "score": ng_score
                },
            ],
            "info": {
                "image_path": image_name
            },
            "gt_inst": [
                class_dict[class_dict_idx]
            ]
        }

        record.append(sub_record)

    final_json['class_dict'] = class_dict
    final_json['record'] = record

    with open('./json/resnext.json', 'w') as fp:
        json.dump(final_json, fp, indent=3, cls=NpEncoder)


if __name__ == '__main__':
    model = load_model()
    res = result_test(model)
    to_json(res)
