import os
import random
import shutil
from tqdm import tqdm

# Legacy examination-folder split; repeated visits can cross train/validation.
SRC_DIR = "./PreprocessedDataSet"
DST_DIR = "./dataset_split"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def split_dataset():

    random.seed(RANDOM_SEED)

    classes = ["Benign", "Malignant"]

    for cls in classes:
        cls_src_path = os.path.join(SRC_DIR, cls)
        if not os.path.exists(cls_src_path):
            print(f"警告：找不到目录 {cls_src_path}，已跳过。")
            continue

        patients = [p for p in os.listdir(cls_src_path) if os.path.isdir(os.path.join(cls_src_path, p))]

        random.shuffle(patients)

        split_idx = int(len(patients) * TRAIN_RATIO)
        train_patients = patients[:split_idx]
        val_patients = patients[split_idx:]

        print(f"\n================ 类别: {cls} ================")
        print(f"总患者数: {len(patients)}")
        print(f"分配到训练集 (Train): {len(train_patients)} 名患者")
        print(f"分配到验证集 (Val)  : {len(val_patients)} 名患者")

        for phase, phase_patients in [("train", train_patients), ("val", val_patients)]:
            print(f"\n正在复制 {cls} 类别下的 {phase} 数据...")

            for patient in tqdm(phase_patients, desc=f"{cls}-{phase}"):
                src_patient_dir = os.path.join(cls_src_path, patient)
                dst_patient_dir = os.path.join(DST_DIR, phase, cls, patient)

                if os.path.exists(dst_patient_dir):
                    shutil.rmtree(dst_patient_dir)

                shutil.copytree(src_patient_dir, dst_patient_dir)

if __name__ == "__main__":
    print("开始基于患者级别的 Train/Val 划分...")
    split_dataset()
    print(f"\n划分完成！拆分后的安全数据集已保存在: {DST_DIR}")
