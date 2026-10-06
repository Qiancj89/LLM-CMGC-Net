import argparse
import csv
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "sota_text_100epoch_best"
CLIN_COLS = ["Internal_Echo", "Morphology", "Boundary", "Solid", "Separation", "Nipple", "Blood_Flow"]
NUM_CLASSES = [4, 2, 2, 2, 2, 2, 2]


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def find_patient_row(df, patient_name):
    if df is None:
        return None
    for col in ["姓名", "Patient_Name", "patient", "ID", "Name", "English_Name", "濮撳悕"]:
        if col in df.columns:
            row = df[df[col].astype(str) == patient_name]
            if len(row) > 0:
                return row.iloc[0]
    row = df[df.eq(patient_name).any(axis=1)]
    if len(row) > 0:
        return row.iloc[0]
    return None


def parse_label(row, root):
    if row is not None:
        for col in ["良恶性", "病理诊断", "Label", "label", "Malignant", "鐥呯悊璇婃柇", "鑹伓鎬?"]:
            if col in row:
                val = str(row[col]).strip()
                if val in ["1", "1.0", "恶性", "Malignant", "yes", "Yes", "鎭舵€?"]:
                    return 1.0
                if val in ["0", "0.0", "良性", "Benign", "no", "No", "鑹€?"]:
                    return 0.0
    root_s = str(root)
    if "Malignant" in root_s:
        return 1.0
    if "Benign" in root_s:
        return 0.0
    return np.nan


def parse_label_from_clinical_row(row):
    if row is None:
        return np.nan
    for col in ["Malignant", "Label", "label", "良恶性", "病理", "病理诊断"]:
        if col in row and not pd.isna(row[col]):
            val = str(row[col]).strip()
            if val in ["1", "1.0", "恶性", "Malignant", "yes", "Yes"]:
                return 1.0
            if val in ["0", "0.0", "良性", "Benign", "no", "No"]:
                return 0.0
            try:
                return float(val)
            except Exception:
                pass
    return np.nan


class PairedUltrasoundDataset(Dataset):
    def __init__(self, data_dir, text_csv, clin_csv, img_size=256):
        self.data_dir = Path(data_dir)
        self.img_size = (img_size, img_size)
        self.df_text = pd.read_csv(text_csv) if text_csv and Path(text_csv).exists() else None
        self.df_clin = pd.read_csv(clin_csv) if clin_csv and Path(clin_csv).exists() else None
        self.samples = self._parse()

    def _parse(self):
        samples = []
        valid_exts = (".bmp", ".png", ".jpg", ".jpeg", ".BMP", ".PNG", ".JPG", ".JPEG")
        for root, dirs, _ in os.walk(self.data_dir):
            if "Original" not in dirs or "Mask" not in dirs:
                continue
            root = Path(root)
            patient = root.name
            row = find_patient_row(self.df_text, patient)
            text = "" if row is None else str(row.get("Clinical_Summary", ""))
            clin = np.full(len(CLIN_COLS), -100, dtype=np.int64)
            clin_row = find_patient_row(self.df_clin, patient)
            mal = parse_label_from_clinical_row(clin_row)
            if np.isnan(mal):
                mal = parse_label(row, root)
            if np.isnan(mal):
                continue
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

            for group_files in groups.values():
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
                            "text": text,
                            "text_emb": np.zeros(768, dtype=np.float32),
                            "patient": patient,
                            "main_slice": pc_f,
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
        mask_path = folder / (Path(filename).stem + ".png")
        mask = cv2.imread(str(mask_path), 0)
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)
        else:
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        return (mask.reshape(1, *self.img_size) > 127).astype(np.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pb = self._read_img(s["orig"], s["pb"])
        pc = self._read_img(s["orig"], s["pc"])
        mask = self._read_mask(s["mask"], s["pc"])
        return {
            "pb": torch.from_numpy(pb).float(),
            "pc": torch.from_numpy(pc).float(),
            "image": torch.from_numpy(np.concatenate([pb, pc], axis=0)).float(),
            "mask": torch.from_numpy(mask).float(),
            "mal": torch.from_numpy(s["mal"]).float(),
            "clin": torch.from_numpy(s["clin"]).long(),
            "text_emb": torch.from_numpy(s["text_emb"]).float(),
            "patient": s["patient"],
            "main_slice": s["main_slice"],
        }


@torch.no_grad()
def attach_text_embeddings(dataset, tokenizer, text_encoder, device, batch_size=16):
    text_encoder.eval()
    texts = [s.get("text", "") for s in dataset.samples]
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_text = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        pooled = text_encoder(**encoded).pooler_output.detach().cpu().numpy().astype(np.float32)
        embeddings.append(pooled)
    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
        for sample, emb in zip(dataset.samples, embeddings):
            sample["text_emb"] = emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="bn"):
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm == "bn" else nn.InstanceNorm2d
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class MultiTaskMixin:
    def _make_heads(self, feat_dim):
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.text_proj = nn.Sequential(nn.Linear(768, feat_dim), nn.GELU(), nn.Dropout(0.1))
        self.fuse_proj = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.GELU(), nn.Dropout(0.1))
        self.mal = nn.Sequential(nn.Dropout(0.25), nn.Linear(feat_dim, 1))
        self.clin_heads = nn.ModuleList([nn.Linear(feat_dim, n) for n in NUM_CLASSES])

    def _heads(self, feat, text_emb=None):
        z = self.gap(feat).flatten(1)
        if text_emb is not None:
            z = self.fuse_proj(torch.cat([z, self.text_proj(text_emb)], dim=1))
        return self.mal(z), [head(z) for head in self.clin_heads]


class UNetMTL(nn.Module, MultiTaskMixin):
    def __init__(self, in_ch=6, base=32, block=ConvBlock):
        super().__init__()
        self.enc1 = block(in_ch, base)
        self.enc2 = block(base, base * 2)
        self.enc3 = block(base * 2, base * 4)
        self.enc4 = block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = block(base * 2, base)
        self.seg = nn.Conv2d(base, 1, 1)
        self._make_heads(base * 8)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def decode(self, e1, e2, e3, e4):
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.seg(d1)

    def forward(self, x, text_emb=None):
        e1, e2, e3, e4 = self.encode(x)
        mal, clin = self._heads(e4, text_emb)
        return self.decode(e1, e2, e3, e4), mal, clin


class ResUNetMTL(UNetMTL):
    def __init__(self, in_ch=6):
        super().__init__(in_ch=in_ch, base=32, block=ResidualBlock)


class UNetPlusPlusMTL(nn.Module, MultiTaskMixin):
    def __init__(self, in_ch=6, base=24):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = lambda x, s: F.interpolate(x, size=s, mode="bilinear", align_corners=False)
        self.x00 = ConvBlock(in_ch, base)
        self.x10 = ConvBlock(base, base * 2)
        self.x20 = ConvBlock(base * 2, base * 4)
        self.x30 = ConvBlock(base * 4, base * 8)
        self.x01 = ConvBlock(base + base * 2, base)
        self.x11 = ConvBlock(base * 2 + base * 4, base * 2)
        self.x21 = ConvBlock(base * 4 + base * 8, base * 4)
        self.x02 = ConvBlock(base * 2 + base * 2, base)
        self.x12 = ConvBlock(base * 4 + base * 4, base * 2)
        self.x03 = ConvBlock(base * 3 + base * 2, base)
        self.seg = nn.Conv2d(base, 1, 1)
        self._make_heads(base * 8)

    def forward(self, x, text_emb=None):
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x01 = self.x01(torch.cat([x00, self.up(x10, x00.shape[-2:])], 1))
        x11 = self.x11(torch.cat([x10, self.up(x20, x10.shape[-2:])], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30, x20.shape[-2:])], 1))
        x02 = self.x02(torch.cat([x00, x01, self.up(x11, x00.shape[-2:])], 1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21, x10.shape[-2:])], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12, x00.shape[-2:])], 1))
        mal, clin = self._heads(x30, text_emb)
        return self.seg(x03), mal, clin


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.g = nn.Conv2d(g_ch, inter_ch, 1, bias=False)
        self.x = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * self.psi(self.g(g) + self.x(x))


class AttentionUNetMTL(nn.Module, MultiTaskMixin):
    def __init__(self, in_ch=6, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.att3 = AttentionGate(base * 8, base * 4, base * 2)
        self.att2 = AttentionGate(base * 4, base * 2, base)
        self.att1 = AttentionGate(base * 2, base, base // 2)
        self.dec3 = ConvBlock(base * 12, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.seg = nn.Conv2d(base, 1, 1)
        self._make_heads(base * 8)

    def forward(self, x, text_emb=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), self.att3(e4, e3)], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), self.att2(d3, e2)], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), self.att1(d2, e1)], 1))
        mal, clin = self._heads(e4, text_emb)
        return self.seg(d1), mal, clin


class ASPP(nn.Module):
    def __init__(self, ch, out_ch):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(ch, out_ch, 1),
                nn.Conv2d(ch, out_ch, 3, padding=2, dilation=2),
                nn.Conv2d(ch, out_ch, 3, padding=4, dilation=4),
                nn.Conv2d(ch, out_ch, 3, padding=6, dilation=6),
            ]
        )
        self.proj = nn.Sequential(nn.Conv2d(out_ch * 4, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.proj(torch.cat([b(x) for b in self.branches], 1))


class DeepLabV3PlusMTL(nn.Module, MultiTaskMixin):
    def __init__(self, in_ch=6, base=32):
        super().__init__()
        self.stem = ConvBlock(in_ch, base)
        self.enc2 = ResidualBlock(base, base * 2)
        self.enc3 = ResidualBlock(base * 2, base * 4)
        self.enc4 = ResidualBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(base * 8, base * 4)
        self.low = nn.Conv2d(base, base, 1)
        self.dec = ConvBlock(base * 5, base * 2)
        self.seg = nn.Conv2d(base * 2, 1, 1)
        self._make_heads(base * 8)

    def forward(self, x, text_emb=None):
        low = self.stem(x)
        e2 = self.enc2(self.pool(low))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        z = self.aspp(e4)
        z = F.interpolate(z, size=low.shape[-2:], mode="bilinear", align_corners=False)
        seg = self.seg(self.dec(torch.cat([z, self.low(low)], 1)))
        mal, clin = self._heads(e4, text_emb)
        return seg, mal, clin


class TransformerBottleneck(nn.Module):
    def __init__(self, ch, heads=4, layers=1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=ch, nhead=heads, dim_feedforward=ch * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.pos = nn.Parameter(torch.zeros(1, 256, ch))

    def forward(self, x):
        b, c, h, w = x.shape
        pooled = F.adaptive_avg_pool2d(x, (16, 16))
        tokens = pooled.flatten(2).transpose(1, 2) + self.pos
        tokens = self.encoder(tokens)
        y = tokens.transpose(1, 2).reshape(b, c, 16, 16)
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


class TransUNetLiteMTL(UNetMTL):
    def __init__(self, in_ch=6):
        super().__init__(in_ch=in_ch, base=32, block=ConvBlock)
        self.trans = TransformerBottleneck(256, heads=4, layers=1)

    def forward(self, x, text_emb=None):
        e1, e2, e3, e4 = self.encode(x)
        e4 = e4 + self.trans(e4)
        mal, clin = self._heads(e4, text_emb)
        return self.decode(e1, e2, e3, e4), mal, clin


class WindowAttentionBlock(nn.Module):
    def __init__(self, ch, window=4, heads=4):
        super().__init__()
        self.window = window
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(nn.LayerNorm(ch), nn.Linear(ch, ch * 2), nn.GELU(), nn.Linear(ch * 2, ch))

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = (self.window - h % self.window) % self.window
        pad_w = (self.window - w % self.window) % self.window
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x_pad.shape[-2:]
        y = x_pad.permute(0, 2, 3, 1).reshape(b, hp // self.window, self.window, wp // self.window, self.window, c)
        y = y.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window * self.window, c)
        y_norm = self.norm1(y)
        y = y + self.attn(y_norm, y_norm, y_norm, need_weights=False)[0]
        y = y + self.ffn(y)
        y = y.reshape(b, hp // self.window, wp // self.window, self.window, self.window, c)
        y = y.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c).permute(0, 3, 1, 2)
        return y[:, :, :h, :w]


class SwinUNetLiteMTL(UNetMTL):
    def __init__(self, in_ch=6):
        super().__init__(in_ch=in_ch, base=32, block=ConvBlock)
        self.win = WindowAttentionBlock(256, window=4, heads=4)

    def forward(self, x, text_emb=None):
        e1, e2, e3, e4 = self.encode(x)
        e4 = e4 + self.win(e4)
        mal, clin = self._heads(e4, text_emb)
        return self.decode(e1, e2, e3, e4), mal, clin


class LateFusionUNetMTL(nn.Module, MultiTaskMixin):
    def __init__(self, base=24):
        super().__init__()
        self.b_encoder = UNetMTL(in_ch=3, base=base)
        self.c_encoder = UNetMTL(in_ch=3, base=base)
        self.fuse = nn.Conv2d(base * 16, base * 8, 1)
        self.dec3 = ConvBlock(base * 12, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.seg = nn.Conv2d(base, 1, 1)
        self._make_heads(base * 8)

    def forward(self, x, text_emb=None):
        pb, pc = x[:, :3], x[:, 3:]
        b1, b2, b3, b4 = self.b_encoder.encode(pb)
        c1, c2, c3, c4 = self.c_encoder.encode(pc)
        e1 = (b1 + c1) / 2
        e2 = (b2 + c2) / 2
        e3 = (b3 + c3) / 2
        e4 = self.fuse(torch.cat([b4, c4], 1))
        d3 = self.dec3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        mal, clin = self._heads(e4, text_emb)
        return self.seg(d1), mal, clin


def build_model(name):
    builders = {
        "unet": lambda: UNetMTL(),
        "unetpp": lambda: UNetPlusPlusMTL(),
        "attunet": lambda: AttentionUNetMTL(),
        "resunet": lambda: ResUNetMTL(),
        "deeplabv3p": lambda: DeepLabV3PlusMTL(),
        "transunet": lambda: TransUNetLiteMTL(),
        "swinunet": lambda: SwinUNetLiteMTL(),
        "latefusion": lambda: LateFusionUNetMTL(),
    }
    if name not in builders:
        raise ValueError(f"Unknown model: {name}")
    return builders[name]()


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(-2, -1))
    union = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - ((2 * inter + 1e-5) / (union + 1e-5)).mean()


def calculate_seg_metrics(pred_mask, gt_mask):
    p = (pred_mask > 0.5).astype(np.float32)
    g = (gt_mask > 0.5).astype(np.float32)
    tp = np.sum(p * g)
    fp = np.sum(p * (1 - g))
    fn = np.sum((1 - p) * g)
    tn = np.sum((1 - p) * (1 - g))
    dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if np.sum(g) == 0 else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if np.sum(g) == 0 else 0.0)
    pa = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    voe = 1.0 - iou
    rvd = (np.sum(p) - np.sum(g)) / np.sum(g) if np.sum(g) > 0 else np.nan
    pts_p = np.argwhere(p > 0)
    pts_g = np.argwhere(g > 0)
    if len(pts_p) == 0 and len(pts_g) == 0:
        hd = 0.0
    elif len(pts_p) == 0 or len(pts_g) == 0:
        hd = np.nan
    else:
        hd = max(directed_hausdorff(pts_p, pts_g)[0], directed_hausdorff(pts_g, pts_p)[0])
    return {"DSC": dsc, "IoU": iou, "HD": hd, "VOE": voe, "RVD": rvd, "PA": pa}


def classification_metrics(y_true, y_prob):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else np.nan,
        "AP": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) == 2 else np.nan,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Hamming Loss": float(hamming_loss(y_true, y_pred)),
    }


def selection_score(metrics, task_type):
    if task_type == "segmentation":
        return float(metrics["DSC"]), "DSC"
    if task_type == "classification":
        score = metrics["AUC"]
        if np.isnan(score):
            score = metrics["Accuracy"]
            return float(score), "Accuracy"
        return float(score), "AUC"
    auc_or_acc = metrics["AUC"] if not np.isnan(metrics["AUC"]) else metrics["Accuracy"]
    return float((metrics["DSC"] + auc_or_acc + metrics["Accuracy"]) / 3.0), "mean_DSC_AUC_Accuracy"


def train_one_epoch(model, loader, optimizer, scaler, device, use_text=True):
    model.train()
    bce_mal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False, disable=True):
        x = batch["image"].to(device)
        text_emb = batch["text_emb"].to(device) if use_text else None
        mask = batch["mask"].to(device)
        mal = batch["mal"].to(device)
        clin = batch["clin"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            seg, mal_logit, clin_logits = model(x, text_emb)
            loss = dice_loss(seg, mask) + F.binary_cross_entropy_with_logits(seg, mask)
            loss = loss + 2.0 * bce_mal(mal_logit, mal)
            sign_loss = 0.0
            for i, logits in enumerate(clin_logits):
                sign_loss = sign_loss + ce(logits, clin[:, i])
            loss = loss + sign_loss / len(clin_logits)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, out_dir, prefix, save_rows=True, use_text=True):
    model.eval()
    seg_rows = []
    y_true, y_prob = [], []
    clin_true = [[] for _ in NUM_CLASSES]
    clin_pred = [[] for _ in NUM_CLASSES]
    for batch in tqdm(loader, desc="eval", leave=False, disable=True):
        x = batch["image"].to(device)
        text_emb = batch["text_emb"].to(device) if use_text else None
        seg, mal_logit, clin_logits = model(x, text_emb)
        pred_masks = torch.sigmoid(seg).cpu().numpy()
        masks = batch["mask"].numpy()
        probs = torch.sigmoid(mal_logit).cpu().numpy().reshape(-1)
        for i in range(x.size(0)):
            metrics = calculate_seg_metrics(pred_masks[i, 0], masks[i, 0])
            row = {
                "Patient_Name": batch["patient"][i],
                "Main_Slice": batch["main_slice"][i],
                "Mal_GT": float(batch["mal"][i].item()),
                "Mal_Prob": float(probs[i]),
                "Mal_Pred": int(probs[i] > 0.5),
                **metrics,
            }
            seg_rows.append(row)
            y_true.append(int(batch["mal"][i].item()))
            y_prob.append(float(probs[i]))
        clin = batch["clin"].numpy()
        for j, logits in enumerate(clin_logits):
            pred = logits.argmax(1).cpu().numpy()
            valid = clin[:, j] != -100
            clin_true[j].extend(clin[valid, j].tolist())
            clin_pred[j].extend(pred[valid].tolist())

    cls = classification_metrics(y_true, y_prob)
    seg_keys = ["DSC", "IoU", "HD", "VOE", "RVD", "PA"]
    metrics = {
        "Model": prefix,
        **{k: float(np.nanmean([r[k] for r in seg_rows])) for k in seg_keys},
        **cls,
    }

    clin_rows = []
    clin_wf1 = []
    for j, name in enumerate(CLIN_COLS):
        if len(clin_true[j]) == 0:
            continue
        acc = accuracy_score(clin_true[j], clin_pred[j])
        mf1 = f1_score(clin_true[j], clin_pred[j], average="macro", zero_division=0)
        wf1 = f1_score(clin_true[j], clin_pred[j], average="weighted", zero_division=0)
        clin_wf1.append(wf1)
        clin_rows.append({"Indicator": name, "Accuracy": acc, "Macro-F1": mf1, "Weighted-F1": wf1})
    metrics["Clinical-Weighted-F1"] = float(np.mean(clin_wf1)) if clin_wf1 else np.nan

    out_dir.mkdir(parents=True, exist_ok=True)
    if save_rows:
        pd.DataFrame(seg_rows).to_csv(out_dir / f"{prefix}_test_patient_results.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(clin_rows).to_csv(out_dir / f"{prefix}_test_clinical_indicator_results.csv", index=False, encoding="utf-8-sig")
    return metrics


def train_model(name, args, train_ds, val_ds, test_ds):
    model_dir = OUT_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    model = build_model(name).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    best_val = -math.inf
    best_metric_name = ""
    best_model_path = model_dir / f"best_{args.task_type}_model.pth"
    logs = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_text=args.use_text)
        scheduler.step()
        val_metrics = None
        score = np.nan
        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, device, model_dir, f"{name}_val_epoch{epoch}", save_rows=False, use_text=args.use_text)
            score, metric_name = selection_score(val_metrics, args.task_type)
            if score > best_val:
                best_val = score
                best_metric_name = metric_name
                torch.save(model.state_dict(), best_model_path)
        log_row = {"epoch": epoch, "loss": loss, "lr": scheduler.get_last_lr()[0], "selection_score": score, "selection_metric": best_metric_name}
        if val_metrics is not None:
            log_row.update({f"val_{k}": v for k, v in val_metrics.items() if k != "Model"})
        logs.append(log_row)
        pd.DataFrame(logs).to_csv(model_dir / "training_log.csv", index=False)
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"{name} epoch {epoch:03d}/{args.epochs} loss={loss:.4f} selection_score={score:.4f}", flush=True)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, model_dir, name, save_rows=True, use_text=args.use_text)
    test_metrics["SelectionMetric"] = best_metric_name
    test_metrics["BestValidationScore"] = best_val
    test_metrics["BestModelPath"] = str(best_model_path)
    with open(model_dir / "test_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_metrics.keys()))
        writer.writeheader()
        writer.writerow(test_metrics)
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--task-type", choices=["segmentation", "classification", "multitask"], default="multitask")
    parser.add_argument("--text-model-name", default="hfl/chinese-macbert-base")
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["unet", "unetpp", "attunet", "resunet", "deeplabv3p", "transunet", "swinunet", "latefusion"],
    )
    args = parser.parse_args()
    args.use_text = not args.no_text
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = PairedUltrasoundDataset(ROOT / "augmented_dataset" / "train", ROOT / "clinical_summary_Huatuo-7B.csv", ROOT / "clinical_data.csv")
    val_ds = PairedUltrasoundDataset(ROOT / "dataset_split" / "val", ROOT / "clinical_summary_Huatuo-7B.csv", ROOT / "clinical_data.csv")
    test_ds = PairedUltrasoundDataset(ROOT / "PreprocessedTestData", ROOT / "clinical_summary_Huatuo-7B_testdata.csv", ROOT / "test_clinical_data.csv")
    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    if args.use_text:
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, local_files_only=True)
        text_encoder = AutoModel.from_pretrained(args.text_model_name, local_files_only=True).to(args.device)
        for p in text_encoder.parameters():
            p.requires_grad_(False)
        attach_text_embeddings(train_ds, tokenizer, text_encoder, args.device)
        attach_text_embeddings(val_ds, tokenizer, text_encoder, args.device)
        attach_text_embeddings(test_ds, tokenizer, text_encoder, args.device)
        del text_encoder
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("LLM clinical narrative embeddings attached with local MacBERT encoder.")

    summary = []
    for name in args.models:
        summary.append(train_model(name, args, train_ds, val_ds, test_ds))
        pd.DataFrame(summary).to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    main()
