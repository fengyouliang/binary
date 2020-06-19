import os
import os.path as osp
import random
import shutil


def split_train_val(p=0.8):
    src_root_path = '/home/youliang/datasets/GIC/train'
    dst_root_path = '/home/youliang/datasets/GIC/val'

    for class_name in os.listdir(src_root_path):
        src_path = f'{src_root_path}/{class_name}'
        dst_path = f'{dst_root_path}/{class_name}'
        os.makedirs(dst_path, exist_ok=True)
        for filename in os.listdir(src_path):
            if random.random() > p:
                src_file = f'{src_path}/{filename}'
                dst_file = f'{dst_path}/{filename}'
                shutil.move(src_file, dst_file)


def leaky_label():
    root_path = '/home/youliang/datasets/GIC_original'
    dirs = ['OK_train', 'NG4_train', 'NG2_train', 'NG5_train', 'NG1_train', 'NG6_ train', 'NG3_train', 'NG7_train']
    result = dict()
    for dir in dirs:
        images = os.listdir(osp.join(root_path, dir))
        label_ids = [image.split('_')[-2] for image in images]
        label_ids = list(set(label_ids))
        label_ids.sort()
        result[dir] = label_ids
    return result


def check_unique_label(retult_d):
    for k, v in retult_d.items():
        for item in v:
            for _k, _v in retult_d.items():
                if _k != k and item in _v:
                    print(item, k, _k)


def main():
    split_train_val()


if __name__ == '__main__':
    main()