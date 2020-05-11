model = {
    'name': 'resnext101_32x8d',
}

# data_path = '/mnt/tmp/feng/final_data_1'
# test_path = '/mnt/tmp/feng/final_data_1'

data_path = '/mnt/tmp/feng/second_final_data/final_data_fold_1'
test_path = '/mnt/tmp/feng/second_final_data/final_data_fold_1'

# test_pth = '0_acc_0.9717_ok_ap_0.9993_ng_ap_0.9526_mAP_0.97595.pth'  # 2个
test_pth = '0_acc_0.9766_ok_ap_0.9991_ng_ap_0.9318_mAP_0.9654499999999999.pth'

# resnext101_32x8d_0508_03_07_37.pth 0.6171171171171171

device_ids = [0, 1, 2]
# device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

resize = (112, 224)

train_keep = -1
val_keep = -1

batch_size = 128

is_pretrained = True

lr = 2e-4
weight_decay = 1e-5
momentum = 0.9

criterion = 'focalloss'
# criterion = 'crossentropyloss'


max_epoch = 50
best_val_acc = 0.5

save_flag = True

vis_bad_case_flag = True
