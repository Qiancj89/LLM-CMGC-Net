import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import warnings

from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, cohen_kappa_score, hamming_loss)
from scipy.spatial.distance import directed_hausdorff

from vl_cmgc_ot_net import VL_CMGC_Net
warnings.filterwarnings("ignore")

CONFIG = {
    "test_dir": "./PreprocessedTestData",
    "csv_path": "./clinical_summary_Qwen-14B_testdata.csv",
    "clin_csv_path": "./test_clinical_data.csv",
    "text_model_name": "hfl/chinese-macbert-base",

    "clin_col_names": [
        "Internal_Echo", "Morphology", "Boundary", "Solid",
        "Separation", "Nipple", "Blood_Flow"
    ],

    "checkpoint_path": "./checkpoints_ot2/Qwen-14B/best_combined_model.pth",

    "batch_size": 1,
    "max_bag_size": 20,
    "img_size": (256, 256),
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "out_seg_paired": "./inference_ot2/Qwen-14B/results_seg_paired.csv",
    "out_seg_unpaired": "./inference_ot2/Qwen-14B/results_seg_unpaired.csv",
    "out_cls_patient": "./inference_ot2/Qwen-14B/results_cls_patient.csv",
    "vis_dir": "./inference_ot2/Qwen-14B/visualizations",
    "report_file": "./inference_ot2/Qwen-14B/metrics_report.txt",
}

NUM_CLASSES_LIST = [4, 2, 2, 2, 2, 2, 2]


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

    v_gt = np.sum(g)
    v_pred = np.sum(p)
    rvd = (v_pred - v_gt) / v_gt if v_gt > 0 else np.nan

    pts_p = np.argwhere(p > 0)
    pts_g = np.argwhere(g > 0)

    if len(pts_p) == 0 and len(pts_g) == 0: hd = 0.0
    elif len(pts_p) == 0 or len(pts_g) == 0: hd = np.nan
    else: hd = max(directed_hausdorff(pts_p, pts_g)[0], directed_hausdorff(pts_g, pts_p)[0])

    return {"DSC": dsc, "IoU": iou, "HD": hd, "VOE": voe, "RVD": rvd, "PA": pa}


def calculate_cls_metrics(y_true, y_prob):
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    y_true = np.array(y_true).astype(int)
    metrics = {}
    try: metrics["AUC"] = roc_auc_score(y_true, y_prob)
    except ValueError: metrics["AUC"] = np.nan

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1-Score"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["Macro-F1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["Weighted-F1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["Kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["Hamming Loss"] = hamming_loss(y_true, y_pred)
    return metrics


def calculate_clin_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true).astype(int), np.array(y_pred).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted-F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }


class TruePatientLevelVLTestDataset(Dataset):
    def __init__(self, data_dir, text_csv, clin_csv, tokenizer_name, crop_unpaired=True):
        self.data_dir = data_dir
        self.df_text = pd.read_csv(text_csv)
        self.df_clin = pd.read_csv(clin_csv) if os.path.exists(clin_csv) else None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.target_size = CONFIG["img_size"]
        self.max_bag = CONFIG["max_bag_size"]
        self.crop_unpaired = crop_unpaired
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
        valid_exts = ('.bmp', '.png', '.jpg', '.jpeg')

        for patient_name in os.listdir(self.data_dir):
            root = os.path.join(self.data_dir, patient_name)
            if not os.path.isdir(root): continue

            p_orig, p_mask = os.path.join(root, "Original"), os.path.join(root, "Mask")
            if not os.path.exists(p_orig) or not os.path.exists(p_mask): continue

            t_row = self._get_patient_row(self.df_text, patient_name)
            if t_row is None: continue
            narrative = str(t_row.get('Clinical_Summary', ''))

            mal_val = np.nan
            for col in ["病理诊断", "良恶性", "Label", "label", "Malignant", "病理"]:
                if col in t_row.keys():
                    val = str(t_row[col]).strip()
                    if val in ['1', '1.0', '恶性', 'Malignant', '是']: mal_val = 1.0; break
                    elif val in ['0', '0.0', '良性', 'Benign', '否']: mal_val = 0.0; break

            clin_labels = np.full(len(NUM_CLASSES_LIST), -100, dtype=np.int64)
            c_row = self._get_patient_row(self.df_clin, patient_name)
            if c_row is not None:
                for idx, c_col in enumerate(CONFIG["clin_col_names"]):
                    if c_col in c_row:
                        val = c_row[c_col]
                        if not pd.isna(val):
                            try: clin_labels[idx] = int(float(val))
                            except: pass

            all_imgs = [f for f in os.listdir(p_orig) if f.lower().endswith(valid_exts)]

            groups = {}
            for f in all_imgs:
                base = os.path.splitext(f)[0]
                suffix = "_aug_" + (base.rsplit("_aug_", 1)[1] if "_aug_" in base else "original")
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
                        "patient": patient_name, "f_name": pc_f if pc_f else pb_f,
                        "text": narrative
                    })
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        encoded_text = self.tokenizer(s["text"], padding='max_length', truncation=True, max_length=512, return_tensors='pt')

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
                return (np.zeros((3, *self.target_size), dtype=np.float32), np.zeros((1, *self.target_size), dtype=np.float32))

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

        b_imgs, b_masks, b_v, b_names = [], [], [], []
        for i in range(self.max_bag):
            if i < len(s["bag_fs"]):
                f = s["bag_fs"][i]
                u_tensor, u_m_tensor = process_roi(_read_raw_img(f), _read_raw_mask(f), force_crop=self.crop_unpaired)
                b_imgs.append(u_tensor); b_masks.append(u_m_tensor); b_v.append(1.0); b_names.append(f)
            else:
                b_imgs.append(np.zeros((3, *self.target_size), dtype=np.float32))
                b_masks.append(np.zeros((1, *self.target_size), dtype=np.float32))
                b_v.append(0.0); b_names.append("")

        return {
            "pb": torch.from_numpy(pb_tensor).float(),
            "pc": torch.from_numpy(pc_tensor).float(),
            "mask_main": torch.from_numpy(m_main).float(),
            "ubs": torch.from_numpy(np.stack(b_imgs)).float(),
            "uv": torch.from_numpy(np.array(b_v)).float(),
            "ubs_masks": torch.from_numpy(np.stack(b_masks)).float(),
            "clin": torch.from_numpy(s["clin"]).long(),
            "mal": torch.from_numpy(s["mal"]).float(),
            "patient_name": s["patient"],
            "main_f_name": s["f_name"],
            "bag_names": b_names,
            "input_ids": encoded_text['input_ids'].squeeze(0),
            "attention_mask": encoded_text['attention_mask'].squeeze(0)
        }


def save_patient_visualizations(p_name, f_main, pb_t, pc_t, m_gt_t, m_pd_t, ubs_t, ubs_m_t, ubs_pd_t, uv_t, bag_names, save_dir):
    """Save each image with red reference and green prediction contours."""
    patient_dir = os.path.join(save_dir, p_name)
    os.makedirs(patient_dir, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def tensor_to_bgr(t):
        arr = np.clip((t.cpu().numpy() * std + mean), 0, 1)
        img = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def draw_overlay_single(img_bgr, gt_mask, pd_mask):
        overlay = img_bgr.copy()
        gt_uint8 = (gt_mask * 255).astype(np.uint8)
        pd_uint8 = (pd_mask * 255).astype(np.uint8)

        c_gt, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_pd, _ = cv2.findContours(pd_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(overlay, c_gt, -1, (0, 0, 255), 2)
        cv2.drawContours(overlay, c_pd, -1, (0, 255, 0), 2)

        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(overlay, "GT (Red)", (10, 25), font, 0.6, (0, 0, 255), 2)
        #cv2.putText(overlay, "Pred (Green)", (10, 50), font, 0.6, (0, 255, 0), 2)

        return overlay

    gt_main = m_gt_t[0, 0].cpu().numpy()
    pd_main = (torch.sigmoid(m_pd_t[0, 0]).cpu().numpy() > 0.5).astype(np.uint8)

    if pb_t is not None:
        img_pb = tensor_to_bgr(pb_t[0])
        overlay_pb = draw_overlay_single(img_pb, gt_main, pd_main)
        pb_save_name = f_main.replace("left", "right").replace("LEFT", "RIGHT") if f_main else "paired_bmode.png"
        cv2.imwrite(os.path.join(patient_dir, pb_save_name), overlay_pb)

    if pc_t is not None:
        img_pc = tensor_to_bgr(pc_t[0])
        overlay_pc = draw_overlay_single(img_pc, gt_main, pd_main)
        pc_save_name = f_main if f_main else "paired_ceus.png"
        cv2.imwrite(os.path.join(patient_dir, pc_save_name), overlay_pc)

    uv_np = uv_t[0].cpu().numpy()
    for i in range(len(uv_np)):
        if uv_np[i] == 1.0:
            img_u = tensor_to_bgr(ubs_t[0, i])
            gt_u = ubs_m_t[0, i, 0].cpu().numpy()
            pd_u = (torch.sigmoid(ubs_pd_t[0, i, 0]).cpu().numpy() > 0.5).astype(np.uint8)

            overlay_u = draw_overlay_single(img_u, gt_u, pd_u)

            b_name = bag_names[i][0] if isinstance(bag_names[i], (list, tuple)) else bag_names[i]
            if not b_name: b_name = f"unpaired_bag_{i}.png"

            cv2.imwrite(os.path.join(patient_dir, b_name), overlay_u)


def inference():
    print(f"启动纯患者级 (Patient-Level) 多模态推理评估")
    os.makedirs(CONFIG["vis_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["out_seg_paired"]), exist_ok=True)

    test_ds = TruePatientLevelVLTestDataset(CONFIG["test_dir"], CONFIG["csv_path"], CONFIG["clin_csv_path"], CONFIG["text_model_name"])
    if len(test_ds) == 0: return
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = VL_CMGC_Net(num_classes_list=NUM_CLASSES_LIST, text_model_name=CONFIG["text_model_name"]).to(CONFIG["device"])

    if not os.path.exists(CONFIG["checkpoint_path"]):
        print(f"找不到权重文件 {CONFIG['checkpoint_path']}")
        return

    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"], weights_only=False)
    model.load_state_dict(checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict'])
    model.eval()

    res_seg_paired, res_seg_unpaired, res_cls_patient = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="推断进行中"):
            pb, pc, m_gt = batch["pb"].cuda(), batch["pc"].cuda(), batch["mask_main"].cuda()
            ubs, ubs_m, uv = batch["ubs"].cuda(), batch["ubs_masks"].cuda(), batch["uv"].cuda()
            input_ids, attention_mask = batch["input_ids"].cuda(), batch["attention_mask"].cuda()

            p_name = batch["patient_name"][0]
            f_main = batch["main_f_name"][0]
            bag_names = batch["bag_names"]

            m_pd, bag_pd, pb_aux_pd, _, c_pds, mal_pd = model(pb, pc, ubs, uv, input_ids, attention_mask)

            p_np_main = torch.sigmoid(m_pd[0, 0]).cpu().numpy()
            g_np_main = m_gt[0, 0].cpu().numpy()
            seg_metrics_main = calculate_seg_metrics(p_np_main, g_np_main)

            row_paired = {"Patient_Name": p_name, "Main_Slice": f_main}
            for k, v in seg_metrics_main.items(): row_paired[k] = round(float(v), 4) if not np.isnan(v) else "N/A"
            res_seg_paired.append(row_paired)

            uv_np = uv[0].cpu().numpy()
            for i in range(CONFIG["max_bag_size"]):
                if uv_np[i] == 1.0:
                    p_np_u = torch.sigmoid(bag_pd[0, i, 0]).cpu().numpy()
                    g_np_u = ubs_m[0, i, 0].cpu().numpy()
                    seg_metrics_u = calculate_seg_metrics(p_np_u, g_np_u)

                    b_file = bag_names[i][0] if isinstance(bag_names[i], (list, tuple)) else bag_names[i]
                    row_unpaired = {"Patient_Name": p_name, "Unpaired_Slice": b_file}
                    for k, v in seg_metrics_u.items(): row_unpaired[k] = round(float(v), 4) if not np.isnan(v) else "N/A"
                    res_seg_unpaired.append(row_unpaired)

            mal_prob = torch.sigmoid(mal_pd).item()
            gt_mal = float(batch["mal"][0].item())

            cls_row = {"Patient_Name": p_name, "Mal_Prob": round(mal_prob, 4), "Mal_Pred": 1 if mal_prob > 0.5 else 0, "Mal_GT": gt_mal}
            gt_clin_arr = batch["clin"][0].cpu().numpy()
            for j in range(len(NUM_CLASSES_LIST)):
                c_col = CONFIG["clin_col_names"][j] if j < len(CONFIG["clin_col_names"]) else f"Clin_{j+1}"
                if c_pds is not None: cls_row[f"{c_col}_Pred"] = int(torch.argmax(c_pds[j][0]).item())
                gt_c = gt_clin_arr[j]
                cls_row[f"{c_col}_GT"] = int(gt_c) if gt_c != -100 else np.nan

            res_cls_patient.append(cls_row)

            save_patient_visualizations(p_name, f_main, pb, pc, m_gt, m_pd, ubs, ubs_m, bag_pd, uv, bag_names, CONFIG["vis_dir"])

    pd.DataFrame(res_seg_paired).to_csv(CONFIG["out_seg_paired"], index=False, encoding="utf-8-sig")
    if len(res_seg_unpaired) > 0: pd.DataFrame(res_seg_unpaired).to_csv(CONFIG["out_seg_unpaired"], index=False, encoding="utf-8-sig")
    df_cls = pd.DataFrame(res_cls_patient)
    df_cls.to_csv(CONFIG["out_cls_patient"], index=False, encoding="utf-8-sig")

    report_lines = ["="*60, "纯患者级医学多模态模型评估报告", "="*60]
    seg_keys = ["DSC", "IoU", "HD", "VOE", "RVD", "PA"]

    report_lines.append("\n【1. 图像分割综合评估 (配对主序列 B超/CEUS)】")
    for k in seg_keys:
        valid_vals = [r[k] for r in res_seg_paired if r[k] != "N/A"]
        report_lines.append(f"{k.ljust(15)}: {np.mean(valid_vals):.4f}" if valid_vals else f"{k.ljust(15)}: N/A")

    if len(res_seg_unpaired) > 0:
        report_lines.append(f"\n【2. 图像分割综合评估 (未配对附属序列 B超, 共 {len(res_seg_unpaired)} 张)】")
        for k in seg_keys:
            valid_vals = [r[k] for r in res_seg_unpaired if r[k] != "N/A"]
            report_lines.append(f"{k.ljust(15)}: {np.mean(valid_vals):.4f}" if valid_vals else f"{k.ljust(15)}: N/A")

    report_lines.append("\n【3. 主任务：良恶性分类评估 (真正的患者级)】")
    valid_mal = df_cls.dropna(subset=["Mal_GT"])
    if len(valid_mal) > 0:
        cls_mets = calculate_cls_metrics(valid_mal["Mal_GT"].values, valid_mal["Mal_Prob"].values)
        for k, v in cls_mets.items(): report_lines.append(f"{k.ljust(15)}: {v:.4f}")

    report_lines.append("\n【4. 辅任务：临床指标评估 (真正的患者级)】")
    for j in range(len(NUM_CLASSES_LIST)):
        c_col = CONFIG["clin_col_names"][j] if j < len(CONFIG["clin_col_names"]) else f"Clin_{j+1}"
        pred_col, gt_col = f"{c_col}_Pred", f"{c_col}_GT"
        valid_c = df_cls.dropna(subset=[gt_col])
        report_lines.append(f"\n--- {c_col} (类别数: {NUM_CLASSES_LIST[j]}, 有效患者数: {len(valid_c)}) ---")
        if len(valid_c) > 0:
            c_mets = calculate_clin_metrics(valid_c[gt_col].values, valid_c[pred_col].values)
            for k, v in c_mets.items(): report_lines.append(f"  {k.ljust(13)}: {v:.4f}")
        else: report_lines.append("缺少金标准，无法评估。")

    report_lines.append("\n" + "="*60)
    report_text = "\n".join(report_lines)
    print(report_text)
    with open(CONFIG["report_file"], "w", encoding="utf-8") as f: f.write(report_text)

if __name__ == "__main__":
    inference()
