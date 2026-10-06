import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(os.environ.get("LLM_CMGC_DATA_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "image_baselines"
CLIN_COLS = ["Internal_Echo", "Morphology", "Boundary", "Solid", "Separation", "Nipple", "Blood_Flow"]
NUM_CLASSES = [4, 2, 2, 2, 2, 2, 2]


def seed_everything(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_patient_row(df, patient_name):
    if df is None:
        return None
    for col in ["姓名", "濮撳悕", "Patient_Name", "patient", "ID", "Name", "English_Name"]:
        if col in df.columns:
            row = df[df[col].astype(str) == patient_name]
            if len(row) > 0:
                return row.iloc[0]
    row = df[df.eq(patient_name).any(axis=1)]
    if len(row) > 0:
        return row.iloc[0]
    return None


def parse_label_from_row(row, root):
    if row is not None:
        for col in ["病理诊断", "鐥呯悊璇婃柇", "良恶性", "鑹伓鎬?", "Label", "label", "Malignant", "病理", "鐥呯悊"]:
            if col in row:
                val = str(row[col]).strip()
                if val in ["1", "1.0", "恶性", "鎭舵€?", "Malignant", "是", "鏄?"]:
                    return 1.0
                if val in ["0", "0.0", "良性", "鑹€?", "Benign", "否", "鍚?"]:
                    return 0.0
    root_s = str(root)
    if "Malignant" in root_s:
        return 1.0
    if "Benign" in root_s:
        return 0.0
    return np.nan


class PairedUltrasoundDataset(Dataset):
    def __init__(self, data_dir, text_csv, clin_csv, mode="early", img_size=256, max_train_samples=None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.img_size = (img_size, img_size)
        self.df_text = pd.read_csv(text_csv) if text_csv and Path(text_csv).exists() else None
        self.df_clin = pd.read_csv(clin_csv) if clin_csv and Path(clin_csv).exists() else None
        self.samples = self._parse()
        if max_train_samples and len(self.samples) > max_train_samples:
            random.Random(2026).shuffle(self.samples)
            self.samples = self.samples[:max_train_samples]

    def _parse(self):
        samples = []
        valid_exts = (".bmp", ".png", ".jpg", ".jpeg", ".BMP", ".PNG", ".JPG", ".JPEG")
        for root, dirs, _ in os.walk(self.data_dir):
            if "Original" not in dirs or "Mask" not in dirs:
                continue
            root = Path(root)
            patient = root.name
            text_row = find_patient_row(self.df_text, patient)
            mal = parse_label_from_row(text_row, root)
            if np.isnan(mal):
                continue
            clin = np.full(len(CLIN_COLS), -100, dtype=np.int64)
            clin_row = find_patient_row(self.df_clin, patient)
            if clin_row is not None:
                for i, col in enumerate(CLIN_COLS):
                    if col in clin_row and not pd.isna(clin_row[col]):
                        try:
                            clin[i] = int(float(clin_row[col]))
                        except Exception:
                            pass

            orig = root / "Original"
            mask = root / "Mask"
            files = [f for f in os.listdir(orig) if f.endswith(valid_exts)]
            groups = {}
            for f in files:
                base = Path(f).stem
                suffix = "_aug_" + (base.rsplit("_aug_", 1)[1] if "_aug_" in base else "original")
                groups.setdefault(suffix, []).append(f)

            for _, group_files in groups.items():
                pc_f, pb_f = None, None
                for f in group_files:
                    lf = f.lower()
                    if "left" in lf:
                        pc_f = f
                    elif "right" in lf:
                        pb_f = f
                if pc_f and pb_f:
                    samples.append(
                        {
                            "orig": orig,
                            "mask": mask,
                            "pc": pc_f,
                            "pb": pb_f,
                            "mal": np.array([mal], dtype=np.float32),
                            "clin": clin,
                            "patient": patient,
                        }
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    def _read_img(self, folder, filename):
        img = cv2.imread(str(folder / filename), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(folder / filename)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return np.transpose(img, (2, 0, 1))

    def _read_mask(self, folder, filename):
        p = folder / (Path(filename).stem + ".png")
        mask = cv2.imread(str(p), 0)
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)
        else:
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        return (mask.reshape(1, *self.img_size) > 127).astype(np.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pc = self._read_img(s["orig"], s["pc"])
        pb = self._read_img(s["orig"], s["pb"])
        if self.mode == "bmode":
            x = pb
        elif self.mode == "ceus":
            x = pc
        else:
            x = np.concatenate([pb, pc], axis=0)
        mask = self._read_mask(s["mask"], s["pc"])
        return {
            "image": torch.from_numpy(x).float(),
            "mask": torch.from_numpy(mask).float(),
            "mal": torch.from_numpy(s["mal"]).float(),
            "clin": torch.from_numpy(s["clin"]).long(),
            "patient": s["patient"],
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ImageOnlyMTL(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(64, 32)
        self.seg = nn.Conv2d(32, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mal = nn.Sequential(nn.Dropout(0.25), nn.Linear(256, 1))
        self.clin_heads = nn.ModuleList([nn.Linear(256, n) for n in NUM_CLASSES])

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        z = self.gap(e4).flatten(1)
        return self.seg(d1), self.mal(z), [h(z) for h in self.clin_heads]


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(-2, -1))
    union = p.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - ((2 * inter + 1e-5) / (union + 1e-5)).mean()


def seg_metrics(pred, gt):
    p = (pred > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    tp = np.sum(p * g)
    fp = np.sum(p * (1 - g))
    fn = np.sum((1 - p) * g)
    tn = np.sum((1 - p) * (1 - g))
    dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    pa = (tp + tn) / (tp + tn + fp + fn)
    pts_p = np.argwhere(p > 0)
    pts_g = np.argwhere(g > 0)
    if len(pts_p) == 0 or len(pts_g) == 0:
        hd = np.nan
    else:
        hd = max(directed_hausdorff(pts_p, pts_g)[0], directed_hausdorff(pts_g, pts_p)[0])
    return dsc, iou, hd, 1.0 - iou, (np.sum(p) - np.sum(g)) / np.sum(g) if np.sum(g) > 0 else np.nan, pa


def train_one(model, loader, optimizer, device):
    model.train()
    total = 0.0
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device)
        mask = batch["mask"].to(device)
        mal = batch["mal"].to(device)
        clin = batch["clin"].to(device)
        optimizer.zero_grad(set_to_none=True)
        seg, mal_logit, clin_logits = model(x)
        loss = dice_loss(seg, mask) + F.binary_cross_entropy_with_logits(seg, mask)
        loss = loss + 2.0 * bce(mal_logit, mal)
        sign_loss = 0.0
        for i, logits in enumerate(clin_logits):
            sign_loss = sign_loss + ce(logits, clin[:, i])
        loss = loss + sign_loss / len(clin_logits)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, out_dir, mode):
    model.eval()
    rows, y_true, y_prob = [], [], []
    clin_true = [[] for _ in NUM_CLASSES]
    clin_pred = [[] for _ in NUM_CLASSES]
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device)
        mask = batch["mask"].numpy()
        seg, mal_logit, clin_logits = model(x)
        prob = torch.sigmoid(mal_logit).cpu().numpy().reshape(-1)
        pred_mask = torch.sigmoid(seg).cpu().numpy()
        for b in range(x.size(0)):
            dsc, iou, hd, voe, rvd, pa = seg_metrics(pred_mask[b, 0], mask[b, 0])
            rows.append([batch["patient"][b], dsc, iou, hd, voe, rvd, pa, float(batch["mal"][b].item()), float(prob[b])])
            y_true.append(int(batch["mal"][b].item()))
            y_prob.append(float(prob[b]))
        clin = batch["clin"].numpy()
        for i, logits in enumerate(clin_logits):
            pred = logits.argmax(dim=1).cpu().numpy()
            valid = clin[:, i] != -100
            clin_true[i].extend(clin[valid, i].tolist())
            clin_pred[i].extend(pred[valid].tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Patient", "DSC", "IoU", "HD", "VOE", "RVD", "PA", "Label", "Prob"]).to_csv(
        out_dir / f"{mode}_patient_results.csv", index=False
    )
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    metrics = {
        "Model": mode,
        "DSC": float(np.nanmean([r[1] for r in rows])),
        "IoU": float(np.nanmean([r[2] for r in rows])),
        "HD": float(np.nanmean([r[3] for r in rows])),
        "VOE": float(np.nanmean([r[4] for r in rows])),
        "RVD": float(np.nanmean([r[5] for r in rows])),
        "PA": float(np.nanmean([r[6] for r in rows])),
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else np.nan,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Hamming Loss": float(hamming_loss(y_true, y_pred)),
    }
    clin_rows = []
    for i, name in enumerate(CLIN_COLS):
        if len(clin_true[i]) == 0:
            continue
        clin_rows.append(
            {
                "Indicator": name,
                "Accuracy": accuracy_score(clin_true[i], clin_pred[i]),
                "Macro-F1": f1_score(clin_true[i], clin_pred[i], average="macro", zero_division=0),
                "Weighted-F1": f1_score(clin_true[i], clin_pred[i], average="weighted", zero_division=0),
            }
        )
    pd.DataFrame(clin_rows).to_csv(out_dir / f"{mode}_clinical_indicator_results.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=900)
    parser.add_argument("--modes", nargs="+", default=["bmode", "ceus", "early"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for mode in args.modes:
        mode_dir = OUT_DIR / mode
        train_ds = PairedUltrasoundDataset(
            ROOT / "augmented_dataset" / "train",
            ROOT / "clinical_summary_Huatuo-7B.csv",
            ROOT / "clinical_data.csv",
            mode=mode,
            max_train_samples=args.max_train_samples,
        )
        val_ds = PairedUltrasoundDataset(
            ROOT / "dataset_split" / "val",
            ROOT / "clinical_summary_Huatuo-7B.csv",
            ROOT / "clinical_data.csv",
            mode=mode,
        )
        test_ds = PairedUltrasoundDataset(
            ROOT / "PreprocessedTestData",
            ROOT / "clinical_summary_Huatuo-7B_testdata.csv",
            ROOT / "test_clinical_data.csv",
            mode=mode,
        )
        in_ch = 6 if mode == "early" else 3
        model = ImageOnlyMTL(in_ch).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
        log_rows = []
        for epoch in range(1, args.epochs + 1):
            loss = train_one(model, train_loader, optimizer, args.device)
            val_metrics = evaluate(model, val_loader, args.device, mode_dir, f"{mode}_val_epoch{epoch}")
            log_rows.append({"epoch": epoch, "loss": loss, **val_metrics})
            torch.save(model.state_dict(), mode_dir / f"epoch_{epoch}.pth")
        pd.DataFrame(log_rows).to_csv(mode_dir / "training_log.csv", index=False)
        torch.save(model.state_dict(), mode_dir / "final_model.pth")
        metrics = evaluate(model, test_loader, args.device, mode_dir, mode)
        summary.append(metrics)

    keys = list(summary[0].keys())
    with open(OUT_DIR / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    main()
