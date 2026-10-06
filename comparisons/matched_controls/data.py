"""Shared examination loader for the four CMPB control conditions."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from manifest import ROOT

SIGN_COLUMNS = [
    "Internal_Echo", "Morphology", "Boundary", "Solid",
    "Separation", "Nipple", "Blood_Flow",
]
SIGN_CLASSES = [4, 2, 2, 2, 2, 2, 2]
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _lookup(path, column):
    frame = pd.read_csv(path)
    frame[column] = frame[column].astype(str).str.casefold()
    if frame[column].duplicated().any():
        raise ValueError(f"Duplicate record key in {path}")
    return frame.set_index(column)


class ExaminationDataset(Dataset):
    def __init__(self, manifest, partition, crop_mode, text_model_name, repeats=1, seed=42):
        if crop_mode not in {"none", "mask"}:
            raise ValueError("crop_mode must be none or mask")
        self.rows = manifest.loc[manifest.partition == partition].reset_index(drop=True)
        self.partition = partition
        self.crop_mode = crop_mode
        self.repeats = repeats if partition == "train" else 1
        self.seed = seed
        self.epoch = 0
        self.max_bag = 3
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        clinical = {
            "development": _lookup(ROOT / "clinical_data.csv", "English_Name"),
            "test": _lookup(ROOT / "test_clinical_data.csv", "English_Name"),
        }
        narratives = {
            "development": _lookup(ROOT / "clinical_summary_Huatuo-7B.csv", "姓名"),
            "test": _lookup(ROOT / "clinical_summary_Huatuo-7B_testdata.csv", "姓名"),
        }
        self.records = []
        for row in self.rows.itertuples(index=False):
            source = row.record_id.split("/", 1)[0]
            name = Path(row.source_path).name.casefold()
            if name not in clinical[source].index or name not in narratives[source].index:
                raise ValueError(f"Missing clinical row or narrative for {row.record_id}")
            clin = clinical[source].loc[name]
            text = str(narratives[source].loc[name, "Clinical_Summary"])
            if not text.strip() or text.lower() == "nan":
                raise ValueError(f"Empty narrative for {row.record_id}")
            labels = []
            for column, classes in zip(SIGN_COLUMNS, SIGN_CLASSES):
                value = clin[column]
                label = -100 if pd.isna(value) else int(value)
                if label != -100 and not 0 <= label < classes:
                    raise ValueError(f"Invalid {column} label in {row.record_id}")
                labels.append(label)
            malignancy = int(clin["Malignant"])
            if malignancy not in {0, 1}:
                raise ValueError(f"Invalid malignancy label in {row.record_id}")
            tokens = self.tokenizer(
                text, padding="max_length", truncation=True, max_length=512,
                return_tensors="pt",
            )
            folder = ROOT / row.source_path
            files = sorted(
                p for p in (folder / "Original").iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            ceus = [p for p in files if "left" in p.stem.lower()]
            paired_b = [p for p in files if "right" in p.stem.lower()]
            supplementary = [p for p in files if p not in ceus and p not in paired_b]
            right_by_stem = {p.stem.lower(): p for p in paired_b}
            pairs = []
            for left in ceus:
                right_stem = left.stem.lower().replace("left", "right", 1)
                if right_stem not in right_by_stem:
                    raise ValueError(f"Unmatched B-mode/CEUS pair in {row.record_id}")
                pairs.append((left, right_by_stem[right_stem]))
            if not pairs or len(pairs) != len(paired_b):
                raise ValueError(f"Incomplete B-mode/CEUS pairing in {row.record_id}")
            selected_ceus, selected_bmode = sorted(pairs, key=lambda pair: pair[0].name.casefold())[-1]
            if not 1 <= len(supplementary) <= self.max_bag:
                raise ValueError(f"Expected 1--3 supplementary views in {row.record_id}")
            self.records.append({
                "record_id": row.record_id,
                "patient_id": row.patient_id,
                "folder": folder,
                "ceus": selected_ceus,
                "paired_b": selected_bmode,
                "archived_pair_count": len(pairs),
                "supplementary": supplementary,
                "labels": labels,
                "malignancy": malignancy,
                "tokens": tokens,
            })

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records) * self.repeats

    def labels(self):
        return [r["malignancy"] for r in self.records for _ in range(self.repeats)]

    @staticmethod
    def _read_image(path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(folder, image_path):
        mask_path = folder / "Mask" / (image_path.stem + ".png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unreadable mask: {mask_path}")
        return mask

    @staticmethod
    def _augment(images, masks, rng):
        flip = rng.random() < 0.5
        affine = rng.random() < 0.8
        angle = float(rng.uniform(-15, 15)) if affine else 0.0
        scale = float(rng.uniform(0.9, 1.1)) if affine else 1.0
        shift_x, shift_y = rng.uniform(-0.06, 0.06, size=2) if affine else (0, 0)
        brightness = float(rng.uniform(-0.15, 0.15)) if rng.random() < 0.5 else 0.0
        contrast = float(rng.uniform(0.85, 1.15)) if brightness else 1.0
        transformed_images, transformed_masks = [], []
        for image, mask in zip(images, masks):
            if flip:
                image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
            height, width = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            matrix[0, 2] += shift_x * width
            matrix[1, 2] += shift_y * height
            image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
            if brightness:
                image = np.clip(image.astype(np.float32) * contrast + brightness * 255, 0, 255).astype(np.uint8)
            transformed_images.append(image)
            transformed_masks.append(mask)
        return transformed_images, transformed_masks

    def _prepare(self, image, mask, supplementary=False):
        if supplementary and self.crop_mode == "mask" and np.any(mask > 127):
            contours, _ = cv2.findContours(
                (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            x, y, width, height = cv2.boundingRect(np.concatenate(contours))
            side = int(max(width, height) * 1.5)
            cx, cy = x + width // 2, y + height // 2
            x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
            x2, y2 = min(image.shape[1], cx + side // 2), min(image.shape[0], cy + side // 2)
            image, mask = image[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        image = (image.astype(np.float32) / 255 - MEAN) / STD
        return image.transpose(2, 0, 1), (mask[None] > 127).astype(np.float32)

    def __getitem__(self, index):
        record_index, repeat = divmod(index, self.repeats)
        record = self.records[record_index]
        paths = [record["ceus"], record["paired_b"], *record["supplementary"]]
        images = [self._read_image(path) for path in paths]
        masks = [self._read_mask(record["folder"], path) for path in paths]
        if self.partition == "train" and repeat:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            images, masks = self._augment(images, masks, rng)
        pc, main_mask = self._prepare(images[0], masks[0])
        pb, _ = self._prepare(images[1], masks[0])
        bag, bag_masks, valid = [], [], []
        for image, mask in zip(images[2:], masks[2:]):
            prepared_image, prepared_mask = self._prepare(image, mask, supplementary=True)
            bag.append(prepared_image)
            bag_masks.append(prepared_mask)
            valid.append(1.0)
        while len(bag) < self.max_bag:
            bag.append(np.zeros((3, 256, 256), dtype=np.float32))
            bag_masks.append(np.zeros((1, 256, 256), dtype=np.float32))
            valid.append(0.0)
        return {
            "pc": torch.from_numpy(pc.copy()),
            "pb": torch.from_numpy(pb.copy()),
            "mask_main": torch.from_numpy(main_mask.copy()),
            "ubs": torch.from_numpy(np.stack(bag)),
            "ubs_masks": torch.from_numpy(np.stack(bag_masks)),
            "uv": torch.tensor(valid, dtype=torch.float32),
            "clin": torch.tensor(record["labels"], dtype=torch.long),
            "mal": torch.tensor(record["malignancy"], dtype=torch.float32),
            "input_ids": record["tokens"]["input_ids"][0],
            "attention_mask": record["tokens"]["attention_mask"][0],
            "record_id": record["record_id"],
            "patient_id": record["patient_id"],
        }
