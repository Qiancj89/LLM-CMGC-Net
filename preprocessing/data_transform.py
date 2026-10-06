import os
import cv2
import numpy as np
import nibabel as nib
from pathlib import Path


def resize_and_pad(image, target_size=512, is_mask=False):
    """Resize proportionally and zero-pad to a square target size."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    if len(image.shape) == 3:
        new_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        new_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = resized
    else:
        new_image = np.zeros((target_size, target_size), dtype=np.uint8)
        new_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        if is_mask:

            _, new_image = cv2.threshold(new_image, 127, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            new_image = cv2.morphologyEx(new_image, cv2.MORPH_CLOSE, kernel)
            new_image = cv2.morphologyEx(new_image, cv2.MORPH_OPEN, kernel)

    return new_image


def process_mask_data(mask_path):
    """Load a mask from a supported format as a binary NumPy array."""
    if mask_path.endswith('.nii.gz'):
        nii_img = nib.load(mask_path)
        data = nii_img.get_fdata()

        mask_2d = data[:, :, 0] if len(data.shape) > 2 else data
        mask_2d = mask_2d.T.astype(np.float32)

        # In the source mask convention, 1 marks the lesion and 2 the background.
        binary = np.zeros_like(mask_2d, dtype=np.uint8)
        binary[mask_2d == 1] = 255
        return binary
    else:

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None: return None
        binary = np.zeros_like(mask_img, dtype=np.uint8)

        binary[mask_img == 1] = 255
        return binary


def run_preprocessing(src_root, dst_root):

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    for root, dirs, files in os.walk(src_root):

        rel_path = os.path.relpath(root, src_root)
        target_path = os.path.join(dst_root, rel_path)
        os.makedirs(target_path, exist_ok=True)

        for file in files:
            src_file_path = os.path.join(root, file)

            folder_type = os.path.basename(root)

            if folder_type == 'Original':

                img = cv2.imread(src_file_path)
                if img is not None:
                    processed_img = resize_and_pad(img, target_size=512, is_mask=False)
                    cv2.imwrite(os.path.join(target_path, file), processed_img)

            elif folder_type == 'Mask':

                file_name_no_ext = file.split('.')[0]
                save_name = f"{file_name_no_ext}.png"

                mask_data = process_mask_data(src_file_path)
                if mask_data is not None:

                    processed_mask = resize_and_pad(mask_data, target_size=512, is_mask=True)
                    cv2.imwrite(os.path.join(target_path, save_name), processed_mask)

if __name__ == "__main__":
    SOURCE_DIR = 'dataset/Malignant'
    DEST_DIR = 'PreprocessedDataSet/Malignant'

    print(f"正在开始预处理，目标目录：{DEST_DIR}...")
    run_preprocessing(SOURCE_DIR, DEST_DIR)
    print("预处理完成！所有图像已调整为 512x512 并完成二值化转换。")
