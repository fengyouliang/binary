import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import config
import utils
from models.model import registry_model


def cam(class_name, image_path, pth, device=torch.device('cuda:2'), num_classes=2, output_dir='./CAM/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # model
    # model = registry_model.get(config.model['name'])()
    # checkpoint = torch.load(pth, map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint)
    # model.to(device)
    model = utils.load_model(pth, config.cuda_available_index)

    # image
    image = Image.open(image_path).convert('RGB')
    W, H = image.size
    tf = transforms.Compose([
        transforms.Resize((112, 224)),
        transforms.ToTensor(),
    ])
    x = tf(image)
    x = x.to(config.device).unsqueeze(0)

    x = model.features(x)
    weight = model.classifer.weight.T
    n, c, h, w = x.size()
    x = x.permute(0, 2, 3, 1).contiguous().view(-1, c)
    logits = torch.mm(x, weight)
    logits = logits.view(h, w, num_classes).contiguous()
    logits = logits.detach().cpu().numpy()

    class_index = 0 if class_name == 'ng' else 1
    cam = logits[:, :, class_index]
    cam = cv.resize(cam, (W, H))
    cam = np.maximum(cam, 0)
    x_min, x_max = np.min(cam), np.max(cam)
    cam = (cam - x_min) / (x_max - x_min) * 255
    cam = cam.astype(dtype=np.int8)

    img = cv.imread(image_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    cam_img = cam
    plt.imshow(cam_img, cmap='jet', alpha=0.3)  # 显示activation map
    plt.subplot(2, 1, 2)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'class: {class_name}')  # 打印类别信息
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    output_path = output_path.replace('jpg', 'png')  # savefig不支持jpg，要转为png
    plt.savefig(output_path)
    plt.close()


def cam_bad_case(output_type=''):
    root_path = './vis_bad_case/fold_1/'
    modes = ['ok', 'ng']
    for mode in modes:
        test_path = f'{root_path}/{mode}'
        bar = tqdm(os.listdir(test_path))
        for file in bar:
            bar.set_description(file)
            image_path = f'{test_path}/{file}'
            cam(mode, image_path, config.test_pth, output_dir=f'./CAM/test{output_type}/{mode}')


if __name__ == '__main__':
    # cam_bad_case()
    pass
