import os

import cv2 as cv
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

import config
import utils


def get_cam(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (200, 500)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv.resize(cam_img, size_upsample))
    return output_cam


def grad_cam(model, image_path, finalconv_name='features'):
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    params = list(model.parameters())
    weight_cls = np.squeeze(params[-2].data.cpu().numpy())

    # image
    image = Image.open(image_path).convert('RGB')
    W, H = image.size
    tf = transforms.Compose([
        transforms.Resize((112, 224)),
        transforms.ToTensor(),
    ])
    x = tf(image)
    x = x.to(config.device).unsqueeze(0)

    logit = model(x)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()

    classes = {
        0: 'ng',
        1: 'ok'
    }
    # output the prediction
    # for i in range(2):
    #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    CAMs = get_cam(features_blobs[0], weight_cls, [idx[0]])

    # print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
    img = cv.imread(image_path)
    height, width, _ = img.shape
    heatmap = cv.applyColorMap(cv.resize(CAMs[0], (width, height)), cv.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5

    return result


def draw_grad_cam(image_path, result, mode='merge', output_path='./CAM'):
    basename = os.path.basename(image_path)[:-4]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    imwrite_name = f'{output_path}/{basename}_grad_cam.jpg'
    if mode == 'cam':
        cv.imwrite(imwrite_name, result)
    elif mode == 'merge':
        img = cv.imread(image_path)
        image = np.vstack((img, result))
        cv.imwrite(imwrite_name, image)


def grad_cam_bad_case(fold_index):
    model = utils.load_model(config.test_pth, config.cuda_available_index)
    root_path = f'./vis_bad_case/fold_{fold_index}/'
    modes = ['ok', 'ng']
    for mode in modes:
        test_path = f'{root_path}/{mode}'
        bar = tqdm(os.listdir(test_path))
        for file in bar:
            bar.set_description(file)
            image_path = f'{test_path}/{file}'
            result = grad_cam(model, image_path)
            draw_grad_cam(image_path, result, output_path=f'./Grad_CAM/test_{fold_index}/{mode}')


if __name__ == '__main__':
    image_path = './examples/demo.jpg'
    pth = '/home/youliang/code/binary/best_FNR_model/31_acc_0.9443_mAP_0.9620500000000001_FNR0.5503.pth'
    model = utils.load_model(pth, config.cuda_available_index)
    result = grad_cam(model, image_path)
    draw_grad_cam(image_path, result, output_path='./examples')
