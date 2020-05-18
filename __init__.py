import os

if not os.path.exists('./best_model'):
    os.makedirs('./best_model')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

if not os.path.exists('./best_acc_model'):
    os.makedirs('./best_acc_model')
if not os.path.exists('./best_ap_model'):
    os.makedirs('./best_ap_model')
if not os.path.exists('./best_FNR_model'):
    os.makedirs('./best_FNR_model')
if not os.path.exists('./pic'):
    os.makedirs('./pic')
if not os.path.exists('./log'):
    os.makedirs('./log')
if not os.path.exists('./json'):
    os.makedirs('./json')
if not os.path.exists('./Grad_CAM'):
    os.makedirs('./Grad_CAM')

