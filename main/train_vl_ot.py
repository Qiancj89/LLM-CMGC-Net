import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import random
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, hamming_loss
)
from transformers import AutoTokenizer

from vl_cmgc_ot_net import VL_CMGC_Net

CONFIG = {
    "train_dir": "./augmented_dataset/train",
    "val_dir": "./dataset_split/val",

    "csv_path": "./clinical_summary_Huatuo-7B.csv",
    "clin_csv_path": "./clinical_data.csv",
    "text_model_name": "hfl/chinese-macbert-base",

    "clin_col_names": [
        "Internal_Echo", "Morphology", "Boundary", "Solid",
        "Separation", "Nipple", "Blood_Flow"
    ],

    "batch_size": 4,
    "lr_base": 1e-4,
    "lr_text": 1e-5,
    "epochs": 100,
    "max_bag_size": 10,
    "img_size": (256, 256),
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "save_dir": "./checkpoints_ot2/Huatuo-7B",
    "vis_dir": "./checkpoints_ot2/visualizations_Huatuo-7B",
    "log_path": "./checkpoints_ot2/Huatuo-7B/train.log",
    "resume_checkpoint": None,
}

NUM_CLASSES_LIST = [4, 2, 2, 2, 2, 2, 2]


def log(msg):
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    with open(CONFIG["log_path"], "a", encoding='utf-8') as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")
    print(msg)


def dice_loss(pred, target):
    p = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(-2, -1))
    union = p.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1. - ((2. * inter + 1e-5) / (union + 1e-5)).mean()


def calculate_segmentation_metrics(pred, gt):
    pred, gt = (pred > 0.5).astype(np.uint8), (gt > 0.5).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    p_sum, g_sum = float(pred.sum()), float(gt.sum())

    dsc = (2 * inter) / (p_sum + g_sum) if (p_sum + g_sum) > 0 else 1.0
    iou = inter / union if union > 0 else 1.0

    p_c, g_c = np.argwhere(pred), np.argwhere(gt)
    if len(p_c) > 0 and len(g_c) > 0:
        hd = max(directed_hausdorff(p_c, g_c)[0], directed_hausdorff(g_c, p_c)[0])
    else:
        hd = 0.0 if len(p_c) == 0 and len(g_c) == 0 else 362.0

    return {"DSC": dsc, "IoU": iou, "HD": hd, "VOE": 100*(1-iou), "RVD": 100*(p_sum-g_sum)/g_sum if g_sum>0 else 0, "PA": (pred==gt).mean()}


class TruePatientLevelVLDataset(Dataset):
    def __init__(self, data_dir, text_csv, clin_csv, tokenizer_name):
        self.data_dir = data_dir
        self.df_text = pd.read_csv(text_csv)
        self.df_clin = pd.read_csv(clin_csv) if os.path.exists(clin_csv) else None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.target_size = CONFIG["img_size"]
        self.max_bag = CONFIG["max_bag_size"]
        self.samples = self._parse_dataset()

    def _get_patient_row(self, df, patient_name):
        if df is None: return None
        for col_name in ['姓名', 'Patient_Name', 'patient', 'ID', 'Name', 'English_Name']:
            if col_name in df.columns:
                row = df[df[col_name].astype(str) == patient_name]
                if len(row) > 0: return row.iloc[0]
        row = df[df.eq(patient_name).any(axis=1)]
        if len(row) > 0: return row.iloc[0]
        return None

    def _parse_dataset(self):
        samples = []
        valid_exts = ('.bmp', '.png', '.jpg', '.jpeg', '.BMP', '.PNG', '.JPG', '.JPEG')

        for root, dirs, files in os.walk(self.data_dir):
            if "Original" in dirs and "Mask" in dirs:
                p_name = os.path.basename(root)

                t_row = self._get_patient_row(self.df_text, p_name)
                if t_row is None: continue
                narrative = str(t_row.get('Clinical_Summary', ''))

                mal_val = np.nan
                for col in ["病理诊断", "良恶性", "Label", "label", "Malignant"]:
                    if col in t_row:
                        val = str(t_row[col]).strip()
                        if val in ['1', '1.0', '恶性', 'Malignant', '是']: mal_val = 1.0; break
                        elif val in ['0', '0.0', '良性', 'Benign', '否']: mal_val = 0.0; break

                if np.isnan(mal_val):
                    if "Malignant" in root: mal_val = 1.0
                    elif "Benign" in root: mal_val = 0.0
                    else: continue

                clin_labels = np.full(len(NUM_CLASSES_LIST), -100, dtype=np.int64)
                c_row = self._get_patient_row(self.df_clin, p_name)
                if c_row is not None:
                    for idx, c_col in enumerate(CONFIG["clin_col_names"]):
                        if c_col in c_row:
                            val = c_row[c_col]
                            if not pd.isna(val):
                                try: clin_labels[idx] = int(float(val))
                                except: pass

                p_orig, p_mask = os.path.join(root, "Original"), os.path.join(root, "Mask")
                all_imgs = [f for f in os.listdir(p_orig) if f.endswith(valid_exts)]

                groups = {}
                for f in all_imgs:
                    base = os.path.splitext(f)[0]
                    suffix = "_aug_" + (base.rsplit("_aug_", 1)[1] if "_aug_" in base else "")
                    if suffix not in groups: groups[suffix] = []
                    groups[suffix].append(f)

                for suffix, g_files in groups.items():
                    pc_f, pb_f, bag_fs = None, None, []
                    for f in g_files:
                        if "left" in f.lower(): pc_f = f
                        elif "right" in f.lower(): pb_f = f
                        else: bag_fs.append(f)

                    if pc_f or pb_f:
                        samples.append({
                            "p_orig": p_orig, "p_mask": p_mask,
                            "pc_f": pc_f, "pb_f": pb_f, "bag_fs": bag_fs,
                            "clin": clin_labels, "mal": np.array([mal_val], dtype=np.float32),
                            "patient": p_name, "f_name": pc_f if pc_f else pb_f,
                            "text": narrative
                        })
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        encoded_text = self.tokenizer(
            s["text"], padding='max_length', truncation=True, max_length=512, return_tensors='pt'
        )

        def _read_raw_img(f):
            if not f: return None
            img = cv2.imread(os.path.join(s["p_orig"], f), cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        def _read_raw_mask(f):
            if not f: return None
            p = os.path.join(s["p_mask"], os.path.splitext(f)[0] + ".png")
            return cv2.imread(p, 0) if os.path.exists(p) else None

        def process_roi(img, mask, force_crop=False, context_ratio=0.5):
            if img is None:
                return (np.zeros((3, *self.target_size), dtype=np.float32),
                        np.zeros((1, *self.target_size), dtype=np.float32))

            mask = mask if mask is not None else np.zeros(img.shape[:2], dtype=np.uint8)
            img_crop, mask_crop = img.copy(), mask.copy()

            if force_crop and np.any(mask > 127):
                contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
                    cx, cy = x + w // 2, y + h // 2
                    side_length = int(max(w, h) * (1 + context_ratio))
                    x1, y1 = max(0, cx - side_length // 2), max(0, cy - side_length // 2)
                    x2, y2 = min(img.shape[1], cx + side_length // 2), min(img.shape[0], cy + side_length // 2)
                    img_crop, mask_crop = img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

            img_res = cv2.resize(img_crop, self.target_size)
            mask_res = cv2.resize(mask_crop, self.target_size, interpolation=cv2.INTER_NEAREST)

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_res = (img_res.astype(np.float32) / 255.0 - mean) / std

            img_tensor = np.transpose(img_res, (2, 0, 1))
            mask_tensor = (mask_res.reshape(1, *self.target_size) > 127).astype(np.float32)

            return img_tensor, mask_tensor

        raw_pc = _read_raw_img(s["pc_f"])
        raw_pb = _read_raw_img(s["pb_f"])
        raw_mask_main = _read_raw_mask(s["pc_f"] if s["pc_f"] else s["pb_f"])

        pc_tensor, m_main = process_roi(raw_pc, raw_mask_main, force_crop=False)
        pb_tensor, _ = process_roi(raw_pb, raw_mask_main, force_crop=False)

        b_imgs, b_masks, b_v = [], [], []
        for i in range(self.max_bag):
            if i < len(s["bag_fs"]):
                f = s["bag_fs"][i]
                u_tensor, u_m_tensor = process_roi(_read_raw_img(f), _read_raw_mask(f), force_crop=True)
                b_imgs.append(u_tensor); b_masks.append(u_m_tensor); b_v.append(1.0)
            else:
                b_imgs.append(np.zeros((3, *self.target_size), dtype=np.float32))
                b_masks.append(np.zeros((1, *self.target_size), dtype=np.float32))
                b_v.append(0.0)

        return {
            "pb": torch.from_numpy(pb_tensor).float(),
            "pc": torch.from_numpy(pc_tensor).float(),
            "mask_main": torch.from_numpy(m_main).float(),
            "ubs": torch.from_numpy(np.stack(b_imgs)).float(),
            "uv": torch.from_numpy(np.array(b_v)).float(),
            "ubs_masks": torch.from_numpy(np.stack(b_masks)).float(),
            "clin": torch.from_numpy(s["clin"]).long(),
            "mal": torch.from_numpy(s["mal"]).float(),
            "p_name": s["patient"],
            "f_name": s["f_name"],
            "input_ids": encoded_text['input_ids'].squeeze(0),
            "attention_mask": encoded_text['attention_mask'].squeeze(0)
        }


def save_training_vis(epoch, p_name, f_name, raw_np, gt_np, pd_np):
    os.makedirs(CONFIG["vis_dir"], exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    raw_np = np.clip((raw_np * std + mean), 0, 1)

    vis = (raw_np.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    c_gt, _ = cv2.findContours(gt_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_pd, _ = cv2.findContours(pd_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, c_gt, -1, (0, 255, 0), 2)
    cv2.drawContours(vis, c_pd, -1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(CONFIG["vis_dir"], f"epoch_{epoch+1}_{p_name}_{f_name}.png"), vis)


def evaluate_patient_level(df_records):
    avg_seg = df_records[["DSC", "IoU", "HD", "VOE", "RVD", "PA"]].mean().to_dict()

    agg_funcs = {"Mal_Prob": "mean", "Mal_GT": "first"}
    for c_idx, num_c in enumerate(NUM_CLASSES_LIST):
        agg_funcs[f"C{c_idx}_GT"] = "first"
        for i in range(num_c):
            agg_funcs[f"C{c_idx}_Prob_{i}"] = "mean"

    df_patient = df_records.groupby("Patient").agg(agg_funcs).reset_index()

    y_true_mal = df_patient["Mal_GT"].values
    y_prob_mal = df_patient["Mal_Prob"].values
    y_pred_mal = (y_prob_mal > 0.5).astype(int)

    cls_metrics = {
        "AUC": roc_auc_score(y_true_mal, y_prob_mal) if len(np.unique(y_true_mal)) > 1 else 0.0,
        "Accuracy": accuracy_score(y_true_mal, y_pred_mal),
        "Precision": precision_score(y_true_mal, y_pred_mal, zero_division=0),
        "Recall": recall_score(y_true_mal, y_pred_mal, zero_division=0),
        "F1-Score": f1_score(y_true_mal, y_pred_mal, zero_division=0),
        "Macro-F1": f1_score(y_true_mal, y_pred_mal, average='macro', zero_division=0),
        "Weighted-F1": f1_score(y_true_mal, y_pred_mal, average='weighted', zero_division=0),
        "Kappa": cohen_kappa_score(y_true_mal, y_pred_mal),
        "Hamming-Loss": hamming_loss(y_true_mal, y_pred_mal)
    }
    return avg_seg, cls_metrics, df_patient


def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True); log(f"启动纯患者级 (Patient-Level) 训练 on {CONFIG['device']}...")

    train_ds = TruePatientLevelVLDataset(CONFIG["train_dir"], CONFIG["csv_path"], CONFIG["clin_csv_path"], CONFIG["text_model_name"])
    val_ds = TruePatientLevelVLDataset(CONFIG["val_dir"], CONFIG["csv_path"], CONFIG["clin_csv_path"], CONFIG["text_model_name"])

    m_labels = [s["mal"][0] for s in train_ds.samples]
    sampler = WeightedRandomSampler(torch.from_numpy(np.array([(1./np.bincount(np.array(m_labels).astype(int)))[int(t)] for t in m_labels])).type('torch.DoubleTensor'), len(m_labels))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=True)

    model = VL_CMGC_Net(num_classes_list=NUM_CLASSES_LIST, text_model_name=CONFIG["text_model_name"]).to(CONFIG["device"])

    text_params = list(model.text_encoder.parameters())
    base_params = [p for n, p in model.named_parameters() if "text_encoder" not in n]

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': CONFIG["lr_base"]},
        {'params': text_params, 'lr': CONFIG["lr_text"]}
    ], weight_decay=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = torch.amp.GradScaler('cuda')

    criterion_mal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(CONFIG["device"]))
    # Missing clinical indicators use -100 and do not contribute to the loss.
    criterion_c = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)

    best_dice, best_auc, best_combined = 0.0, 0.0, 0.0
    start_epoch = 0

    if CONFIG["resume_checkpoint"] and os.path.exists(CONFIG["resume_checkpoint"]):
        ckpt = torch.load(CONFIG["resume_checkpoint"]); model.load_state_dict(ckpt['model_state_dict']); optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']+1; best_dice = ckpt.get('best_dice', 0.0); best_auc = ckpt.get('best_auc', 0.0); best_combined = ckpt.get('best_combined', 0.0)
        log(f"成功恢复训练至 Epoch {start_epoch+1}")

    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train(); t_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            pb, pc, m_gt = batch["pb"].to(CONFIG["device"]), batch["pc"].to(CONFIG["device"]), batch["mask_main"].to(CONFIG["device"])
            ubs, ubs_m, uv = batch["ubs"].to(CONFIG["device"]), batch["ubs_masks"].to(CONFIG["device"]), batch["uv"].to(CONFIG["device"])
            ml, cl = batch["mal"].to(CONFIG["device"]), batch["clin"].to(CONFIG["device"])
            input_ids, attention_mask = batch["input_ids"].to(CONFIG["device"]), batch["attention_mask"].to(CONFIG["device"])

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):

                m_pd, bag_pd, pb_aux_pd, loss_itc, c_pds, mal_pd = model(pb, pc, ubs, uv, input_ids, attention_mask)

                l_seg_main = dice_loss(m_pd, m_gt) + F.binary_cross_entropy_with_logits(m_pd, m_gt)

                l_seg_pb_aux = dice_loss(pb_aux_pd, m_gt) + F.binary_cross_entropy_with_logits(pb_aux_pd, m_gt)

                l_seg = l_seg_main + 0.5 * l_seg_pb_aux

                if uv.sum() > 0:
                    b_pd_flat = bag_pd.view(-1, 1, 256, 256)
                    b_gt_flat = ubs_m.view(-1, 1, 256, 256)
                    valid_idx = torch.where(uv.view(-1) > 0)[0]
                    if len(valid_idx) > 0:
                        l_bag = dice_loss(b_pd_flat[valid_idx], b_gt_flat[valid_idx]) + F.binary_cross_entropy_with_logits(b_pd_flat[valid_idx], b_gt_flat[valid_idx])
                    else:
                        l_bag = 0.0
                else:
                    l_bag = 0.0

                l_mal = criterion_mal(mal_pd.squeeze(), ml.squeeze())
                l_c = sum([criterion_c(p, cl[:, i]) for i, p in enumerate(c_pds)]) / len(NUM_CLASSES_LIST)

                loss = 1.0 * l_seg + 0.5 * l_bag + 2.0 * l_mal + 1.0 * l_c + 0.5 * loss_itc

            if torch.isnan(loss):
                print(f"\n[Warning] NaN loss detected at Epoch {epoch+1}, skipping batch...")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        model.eval()
        val_records = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch+1}"):
                pb, pc, m_gt = batch["pb"].to(CONFIG["device"]), batch["pc"].to(CONFIG["device"]), batch["mask_main"].to(CONFIG["device"])
                ubs, uv = batch["ubs"].to(CONFIG["device"]), batch["uv"].to(CONFIG["device"])
                input_ids, attention_mask = batch["input_ids"].to(CONFIG["device"]), batch["attention_mask"].to(CONFIG["device"])

                m_pd, _, _, _, c_pds, mal_pd = model(pb, pc, ubs, uv, input_ids, attention_mask)

                p_np = torch.sigmoid(m_pd[0,0]).cpu().numpy()
                g_np = m_gt[0,0].cpu().numpy()
                seg_scores = calculate_segmentation_metrics(p_np, g_np)

                if random.random() < 0.1:
                    save_training_vis(epoch, batch["p_name"][0], batch["f_name"][0], pc[0].cpu().numpy(), (g_np*255).astype(np.uint8), ((p_np>0.5)*255).astype(np.uint8))

                rec = {
                    "Patient": batch["p_name"][0], "File": batch["f_name"][0],
                    "Mal_Prob": torch.sigmoid(mal_pd).item(), "Mal_GT": batch["mal"].item()
                }
                rec.update(seg_scores)

                for c_idx, num_c in enumerate(NUM_CLASSES_LIST):
                    prob_c = F.softmax(c_pds[c_idx], dim=1).cpu().numpy()[0]
                    for i in range(num_c): rec[f"C{c_idx}_Prob_{i}"] = prob_c[i]
                    rec[f"C{c_idx}_GT"] = batch["clin"][0, c_idx].item()
                val_records.append(rec)

        df_records = pd.DataFrame(val_records)
        avg_seg, cls_metrics, df_patient = evaluate_patient_level(df_records)
        cur_auc = cls_metrics["AUC"]
        avg_dsc = avg_seg["DSC"]
        cur_combined = (cur_auc + avg_dsc) / 2.0

        log(f"Epoch {epoch+1} Results: Loss {t_loss/len(train_loader):.4f} | DSC: {avg_dsc:.4f} | AUC: {cur_auc:.4f} | Acc: {cls_metrics['Accuracy']:.4f}")

        if avg_dsc > best_dice:
            best_dice = avg_dsc
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_seg_model.pth"))
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_auc_model.pth"))
        if cur_combined > best_combined:
            best_combined = cur_combined
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_combined_model.pth"))
            df_patient.to_csv(os.path.join(CONFIG["save_dir"], "best_patient_level_results.csv"), index=False)
            pd.DataFrame([cls_metrics]).to_csv(os.path.join(CONFIG["save_dir"], "best_cls_metrics.csv"), index=False)
            pd.DataFrame([avg_seg]).to_csv(os.path.join(CONFIG["save_dir"], "best_seg_metrics.csv"), index=False)

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_dice': best_dice, 'best_auc': best_auc, 'best_combined': best_combined}, os.path.join(CONFIG["save_dir"], "latest_checkpoint.pth"))

if __name__ == "__main__":
    train()
