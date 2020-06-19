import torch
import pandas as pd
import config
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.nn.functional import softmax
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.device_ids))

# pth = '/home/youliang/code/binary/checkpoints/mobilenet/030_acc_0.9995.pth'  # mobillenet
pth = '/home/youliang/code/binary/checkpoints/efficientnet/030_acc_0.9990.pth'  # efficientnet-b0

classes = ['NG1', 'NG2', 'NG3', 'NG4', 'NG5', 'NG6', 'NG7', 'OK']


def load_model(pth_path):
    # model = registry_model.get(config.model['name'])(num_classes=config.num_class)
    checkpoint = torch.load(pth_path, map_location=lambda storage, loc: storage.cuda())
    model = checkpoint.module
    return model


@torch.no_grad()
def main():
    model = load_model(pth)
    print(model)

    for name, param in model.named_parameters():
        print(name, param.shape)
        print(param)

    test_tf = transforms.Compose([
        transforms.Resize(config.resize),
        transforms.ToTensor(),
    ])

    GIC_val_dataset = ImageFolder(root='/home/youliang/datasets/GIC/val', transform=test_tf)
    GIC_val_loader = DataLoader(GIC_val_dataset, batch_size=config.test_batch_size, pin_memory=True)

    bar = tqdm(enumerate(GIC_val_loader))
    for ii, (data, label) in bar:
        image = data.cuda()
        logits = model(image)
        prob = softmax(logits, dim=1)
        max_pred_score, max_pred_class = prob.max(dim=1)
        print(label, max_pred_class)
        break


@torch.no_grad()
def test(test_path='/home/youliang/datasets/GIC/test'):
    model = load_model(pth)
    print('model load done!')

    res_dict = []
    bar = tqdm(os.listdir(test_path))
    for image in bar:
        image_path = f'{test_path}/{image}'
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize(config.resize),
            transforms.ToTensor(),
        ])
        image = tf(image_path)
        image = image.unsqueeze(0)
        image = image.cuda()
        logits = model(image)
        prob = softmax(logits, dim=1)
        max_pred_score, max_pred_class = prob.max(dim=1)
        pred_class = max_pred_class.data.cpu().numpy().item()

        pd_row = {
            'image_path': image_path,
            'pred': classes[pred_class]
        }
        res_dict.append(pd_row)
    return res_dict


if __name__ == '__main__':
    results = test()
    print(results)
    test_df = pd.DataFrame(results, columns=['image_path', 'pred'])
    print(test_df)
