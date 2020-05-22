import torch

model = {
    # 'name': 'resnext101_32x8d',
    # 'name': 'mobilenet',
    # 'name': 'EfficientNet',
    # 'name': 'efficientnet_da'
    'name': 'mobilenet_da'
}
batch_size = 512
test_batch_size = 256

lr = 1e-5
weight_decay = 1e-5
momentum = 0.9

resize = (112, 224)
# resize = (200, 500)

# cuda_available_index = 0
# device = torch.device('cpu') if cuda_available_index == -1 else torch.device(f'cuda:{cuda_available_index}')
device_ids = [0, 1]

train_keep = -1
val_keep = -1

is_pretrained = True

# criterion = 'focalloss'
criterion = 'crossentropyloss'

# optimizer = 'adam'

max_epoch = 30

save_flag = True

vis_bad_case_flag = True

# first data path
# data_path = '/mnt/tmp/feng/final_data_1'
# test_path = '/mnt/tmp/feng/final_data_1'
# second data path
# data_path = '/mnt/tmp/feng/second_final_data/final_data_fold_1'
# test_path = '/mnt/tmp/feng/second_final_data/final_data_fold_1'

# kuozhankuang data path
data_fold_index = 1
data_path = f'/mnt/tmp/feng/kuozhankuang/fold_{data_fold_index}'
test_path = f'/mnt/tmp/feng/kuozhankuang/fold_{data_fold_index}'

# resnext101_32x8d
# test_pth = '0_acc_0.9717_ok_ap_0.9993_ng_ap_0.9526_mAP_0.97595.pth'  # 2个
# test_pth = '0_acc_0.9766_ok_ap_0.9991_ng_ap_0.9318_mAP_0.9654499999999999.pth'
# resnext101_32x8d_0508_03_07_37.pth 0.6171171171171171


# mobile net
# test_pth = '0_acc_0.5353_ok_ap_0.5151_ng_ap_0.591_mAP_0.55305.pth'

# Crossentropy v.s. FocalLoss
# test_pth = './best_FNR_model/30_acc_0.9376_mAP_0.9614_FNR0.4972.pth'  # Crossentropy best FNR model
# test_pth = './best_FNR_model/43_acc_0.9014_mAP_0.8955500000000001_FNR0.3427.pth'  # FocalLoss best FNR model

# test_pth = './checkpoints/resnext101_32x8d/28_acc_0.9891_mAP_0.9919_FOR_0.1761.pth'
test_pth = './checkpoints/EfficientNet/28_acc_0.9769_mAP_0.978_FOR_0.2415.pth'
# test_pth = './checkpoints/mobilenet/20_acc_0.9806_mAP_0.9823999999999999_FOR_0.5975.pth'
