import os
import cv2
import numpy as np
import shutil
import albumentations as A
from tqdm import tqdm

SRC_DIR = "./dataset_split/train"

AUG_DIR = "./augmented_dataset/train"

AUG_MULTIPLIER = 9


def find_mask(ori_name, mask_dir):
    """Find the mask with the same stem regardless of file extension."""
    base_name = os.path.splitext(ori_name)[0]
    for ext in ['.png', '.jpg', '.bmp', '.jpeg', '.PNG', '.JPG', '.BMP']:
        mask_path = os.path.join(mask_dir, base_name + ext)
        if os.path.exists(mask_path):
            return mask_path
    return None


def get_aug_pipeline(num_additional_targets):
    """Build an augmentation pipeline with synchronized image/mask geometry."""
    additional_targets = {}

    for i in range(1, num_additional_targets):
        additional_targets[f'image{i}'] = 'image'
        additional_targets[f'mask{i}'] = 'mask'

    return A.Compose([

        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.8,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        ),

        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3)
    ], additional_targets=additional_targets)


def augment_dataset():
    if os.path.exists(AUG_DIR):
        print(f"清理旧的增强数据集目录: {AUG_DIR}")
        shutil.rmtree(AUG_DIR)

    classes = ["Benign", "Malignant"]

    for cls in classes:
        cls_src_dir = os.path.join(SRC_DIR, cls)
        if not os.path.exists(cls_src_dir):
            continue

        patients = os.listdir(cls_src_dir)
        print(f"\n正在增强 [{cls}] 类别，共 {len(patients)} 名患者...")

        for patient in tqdm(patients, desc=f"{cls} Augmentation"):
            patient_src_dir = os.path.join(cls_src_dir, patient)
            ori_src_dir = os.path.join(patient_src_dir, "Original")
            mask_src_dir = os.path.join(patient_src_dir, "Mask")

            if not os.path.isdir(ori_src_dir) or not os.path.isdir(mask_src_dir):
                continue

            valid_files = []
            patient_imgs = []
            patient_masks = []

            for file_name in os.listdir(ori_src_dir):
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue

                ori_path = os.path.join(ori_src_dir, file_name)
                mask_path = find_mask(file_name, mask_src_dir)

                if mask_path:

                    img = cv2.imread(ori_path, cv2.IMREAD_UNCHANGED)

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    if img is not None and mask is not None:
                        valid_files.append(file_name)
                        patient_imgs.append(img)
                        patient_masks.append(mask)

            num_pairs = len(valid_files)
            if num_pairs == 0: continue

            patient_dst_dir = os.path.join(AUG_DIR, cls, patient)
            ori_dst_dir = os.path.join(patient_dst_dir, "Original")
            mask_dst_dir = os.path.join(patient_dst_dir, "Mask")
            os.makedirs(ori_dst_dir, exist_ok=True)
            os.makedirs(mask_dst_dir, exist_ok=True)

            for idx, file_name in enumerate(valid_files):
                cv2.imwrite(os.path.join(ori_dst_dir, file_name), patient_imgs[idx])

                clean_mask = (patient_masks[idx] > 127).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(mask_dst_dir, f"{os.path.splitext(file_name)[0]}.png"), clean_mask)

            aug_pipeline = get_aug_pipeline(num_pairs)

            for aug_idx in range(1, AUG_MULTIPLIER + 1):

                kwargs = {
                    'image': patient_imgs[0],
                    'mask': patient_masks[0]
                }
                for i in range(1, num_pairs):
                    kwargs[f'image{i}'] = patient_imgs[i]
                    kwargs[f'mask{i}']  = patient_masks[i]

                augmented = aug_pipeline(**kwargs)

                aug_imgs = [augmented['image']]
                aug_masks = [augmented['mask']]
                for i in range(1, num_pairs):
                    aug_imgs.append(augmented[f'image{i}'])
                    aug_masks.append(augmented[f'mask{i}'])

                for idx, file_name in enumerate(valid_files):
                    base_name, ext = os.path.splitext(file_name)
                    save_name = f"{base_name}_aug{aug_idx}{ext}"
                    mask_save_name = f"{base_name}_aug{aug_idx}.png"

                    a_img = aug_imgs[idx]
                    a_mask = aug_masks[idx]

                    a_mask = (a_mask > 127).astype(np.uint8) * 255

                    cv2.imwrite(os.path.join(ori_dst_dir, save_name), a_img)
                    cv2.imwrite(os.path.join(mask_dst_dir, mask_save_name), a_mask)

if __name__ == "__main__":
    print("启动患者级别的跨模态同步数据增强...")
    augment_dataset()
    print(f"\n增强完成！包含原图和 {AUG_MULTIPLIER} 倍增强的数据已保存至: {AUG_DIR}")
    print("可以直接将 train_vl.py 里的训练集路径指向这个新目录了！")
