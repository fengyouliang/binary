model = {
    'name': 'resnext101_32x8d',
}

# data_path = '/mnt/tmp/feng/final_data_1'
# test_path = '/mnt/tmp/feng/final_data_1'

data_path = '/mnt/tmp/feng/second_final_data/final_data_fold_1'
test_path = '/mnt/tmp/feng/second_final_data/final_data_fold_1'

test_pth = '0_acc_0.9717_ok_ap_0.9993_ng_ap_0.9526_mAP_0.97595.pth'  # 2个
# test_pth = '0_acc_0.9745_ok_ap_0.9988_ng_ap_0.9293_mAP_0.9640500000000001.pth'
# test_pth = '0_acc_0.9760_ok_ap_0.9990_ng_ap_0.9337_mAP_0.96635.pth'
# test_pth = '38_acc_0.9311_ok_ap_0.9874_ng_ap_0.9994_mAP_0.9934000000000001.pth'

# resnext101_32x8d_0508_03_07_37.pth 0.6171171171171171

device_ids = [0, 2]
# device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

resize = (112, 224)

train_keep = -1
val_keep = -1

batch_size = 64

is_pretrained = True

lr = 2e-4
weight_decay = 1e-5
momentum = 0.9

max_epoch = 10
best_val_acc = 0.5

save_flag = True

vis_bad_case_flag = True
