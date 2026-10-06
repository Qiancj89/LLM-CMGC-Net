import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("LLM_CMGC_DATA_ROOT", CODE_ROOT)).resolve()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from context_model import ContextAblationNet
from inference_vl_ot import TruePatientLevelVLTestDataset
from train_vl_ot import (
    NUM_CLASSES_LIST,
    TruePatientLevelVLDataset,
    calculate_segmentation_metrics,
    dice_loss,
)

SOURCE_FIELDS = ["年龄", "绝经状态", "部位", "HE4", "AFP", "CEA", "CA125", "CA153", "CA199", "CA724", "NSE", "CY211"]
MARKERS = SOURCE_FIELDS[3:]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def numeric(value):
    if pd.isna(value):
        return np.nan
    try:
        return float(str(value).strip())
    except ValueError:
        return np.nan


def menopause(value):
    text = str(value).strip().lower()
    return 1.0 if text in {"是", "yes", "postmenopausal", "绝经"} else 0.0


def side_vector(value):
    text = str(value).strip().lower()
    if "左" in text or "left" in text:
        index = 0
    elif "右" in text or "right" in text:
        index = 1
    elif "双" in text or "bilateral" in text:
        index = 2
    else:
        index = 3
    vector = np.zeros(4, dtype=np.float32)
    vector[index] = 1.0
    return vector


class ContextDataset(TruePatientLevelVLDataset):
    def __init__(self, *args, tabular_stats=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_lookup = {}
        for _, row in self.df_text.iterrows():
            key = str(row.get("姓名", row.get("English_Name", "")))
            self.row_lookup[key] = row
        self.tabular_stats = tabular_stats or self._fit_stats()

    def _fit_stats(self):
        rows = []
        used = set()
        for sample in self.samples:
            name = sample["patient"]
            if name in used or name not in self.row_lookup:
                continue
            used.add(name)
            row = self.row_lookup[name]
            rows.append([numeric(row.get("年龄"))] + [numeric(row.get(field)) for field in MARKERS])
        values = np.asarray(rows, dtype=float)
        means = np.nanmean(values, axis=0)
        scales = np.nanstd(values, axis=0)
        scales[scales < 1e-8] = 1.0
        return {"means": means.tolist(), "scales": scales.tolist()}

    def _tabular_vector(self, name):
        row = self.row_lookup[name]
        continuous = np.asarray(
            [numeric(row.get("年龄"))] + [numeric(row.get(field)) for field in MARKERS],
            dtype=float,
        )
        missing = np.isnan(continuous).astype(np.float32)
        means = np.asarray(self.tabular_stats["means"], dtype=float)
        scales = np.asarray(self.tabular_stats["scales"], dtype=float)
        standardized = (np.where(np.isnan(continuous), means, continuous) - means) / scales
        return np.concatenate(
            [
                standardized.astype(np.float32),
                np.asarray([menopause(row.get("绝经状态"))], dtype=np.float32),
                side_vector(row.get("部位")),
                missing,
            ]
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["tabular"] = torch.from_numpy(self._tabular_vector(item["p_name"]))
        return item


class ContextTestDataset(TruePatientLevelVLTestDataset):
    def __init__(self, *args, tabular_stats, **kwargs):
        super().__init__(*args, **kwargs)
        self.tabular_stats = tabular_stats
        self.row_lookup = {}
        for _, row in self.df_text.iterrows():
            key = str(row.get("姓名", row.get("English_Name", "")))
            self.row_lookup[key] = row

    def _tabular_vector(self, name):
        row = self.row_lookup[name]
        continuous = np.asarray(
            [numeric(row.get("年龄"))] + [numeric(row.get(field)) for field in MARKERS],
            dtype=float,
        )
        missing = np.isnan(continuous).astype(np.float32)
        means = np.asarray(self.tabular_stats["means"], dtype=float)
        scales = np.asarray(self.tabular_stats["scales"], dtype=float)
        standardized = (np.where(np.isnan(continuous), means, continuous) - means) / scales
        return np.concatenate(
            [
                standardized.astype(np.float32),
                np.asarray([menopause(row.get("绝经状态"))], dtype=np.float32),
                side_vector(row.get("部位")),
                missing,
            ]
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        name = item["patient_name"]
        item["p_name"] = name
        item["tabular"] = torch.from_numpy(self._tabular_vector(name))
        return item


def classification_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities > 0.5).astype(int)
    return {
        "AUC": roc_auc_score(labels, probabilities),
        "AP": average_precision_score(labels, probabilities),
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1-Score": f1_score(labels, predictions, zero_division=0),
        "Macro-F1": f1_score(labels, predictions, average="macro", zero_division=0),
        "Weighted-F1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "Kappa": cohen_kappa_score(labels, predictions),
    }


def evaluate(model, loader, device, include_unpaired=False):
    model.eval()
    paired_rows, unpaired_rows, patient_rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            pb = batch["pb"].to(device)
            pc = batch["pc"].to(device)
            ubs = batch["ubs"].to(device)
            uv = batch["uv"].to(device)
            tabular = batch["tabular"].to(device)
            main, bag, _, _, concepts, malignancy = model(pb, pc, ubs, uv, tabular)

            for b in range(pb.size(0)):
                patient = batch["p_name"][b]
                probability = torch.sigmoid(malignancy[b]).item()
                patient_row = {
                    "Patient_Name": patient,
                    "Mal_Prob": probability,
                    "Mal_Pred": int(probability > 0.5),
                    "Mal_GT": int(batch["mal"][b].item()),
                }
                for index, prediction in enumerate(concepts):
                    patient_row[f"C{index}_Pred"] = int(prediction[b].argmax().item())
                    patient_row[f"C{index}_GT"] = int(batch["clin"][b, index].item())
                patient_rows.append(patient_row)

                pred = torch.sigmoid(main[b, 0]).cpu().numpy()
                ground_truth = batch["mask_main"][b, 0].numpy()
                paired_rows.append(
                    {
                        "Patient_Name": patient,
                        **calculate_segmentation_metrics(pred, ground_truth),
                    }
                )
                if include_unpaired:
                    for index in range(uv.size(1)):
                        if uv[b, index].item() == 0:
                            continue
                        pred_u = torch.sigmoid(bag[b, index, 0]).cpu().numpy()
                        gt_u = batch["ubs_masks"][b, index, 0].numpy()
                        unpaired_rows.append(
                            {
                                "Patient_Name": patient,
                                "Bag_Index": index,
                                **calculate_segmentation_metrics(pred_u, gt_u),
                            }
                        )

    paired = pd.DataFrame(paired_rows)
    unpaired = pd.DataFrame(unpaired_rows)
    patients = pd.DataFrame(patient_rows)
    metrics = {f"Paired_{key}": value for key, value in paired.drop(columns="Patient_Name").mean().items()}
    if not unpaired.empty:
        metrics.update({f"Unpaired_{key}": value for key, value in unpaired.drop(columns=["Patient_Name", "Bag_Index"]).mean().items()})
    metrics.update(classification_metrics(patients["Mal_GT"], patients["Mal_Prob"]))

    concept_scores = []
    for index in range(len(NUM_CLASSES_LIST)):
        valid = patients[f"C{index}_GT"] != -100
        if valid.any():
            concept_scores.append(
                f1_score(
                    patients.loc[valid, f"C{index}_GT"],
                    patients.loc[valid, f"C{index}_Pred"],
                    average="weighted",
                    zero_division=0,
                )
            )
    metrics["Clinical_Weighted_F1"] = float(np.mean(concept_scores))
    return metrics, paired, unpaired, patients


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output / args.mode
    output.mkdir(parents=True, exist_ok=True)

    common = {
        "text_csv": ROOT / "clinical_summary_Huatuo-7B.csv",
        "clin_csv": ROOT / "clinical_data.csv",
        "tokenizer_name": "hfl/chinese-macbert-base",
    }
    train_set = ContextDataset(ROOT / "augmented_dataset" / "train", **common)
    stats = train_set.tabular_stats
    validation_set = ContextDataset(ROOT / "dataset_split" / "val", tabular_stats=stats, **common)
    test_set = ContextTestDataset(
        ROOT / "PreprocessedTestData",
        ROOT / "clinical_summary_Huatuo-7B_testdata.csv",
        ROOT / "test_clinical_data.csv",
        "hfl/chinese-macbert-base",
        tabular_stats=stats,
    )
    with open(output / "tabular_normalization.json", "w", encoding="utf-8") as handle:
        json.dump({"fields": SOURCE_FIELDS, **stats}, handle, indent=2, ensure_ascii=False)

    labels = np.asarray([sample["mal"][0] for sample in train_set.samples], dtype=int)
    counts = np.bincount(labels)
    weights = np.asarray([1.0 / counts[label] for label in labels], dtype=float)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    model = ContextAblationNet(args.mode, tabular_dim=len(train_set[0]["tabular"])).to(device)
    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=3e-2,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    malignancy_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
    concept_loss = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)

    best_score = -np.inf
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            pb, pc = batch["pb"].to(device), batch["pc"].to(device)
            main_gt = batch["mask_main"].to(device)
            ubs, ubs_gt, uv = batch["ubs"].to(device), batch["ubs_masks"].to(device), batch["uv"].to(device)
            clinical = batch["clin"].to(device)
            malignancy_gt = batch["mal"].to(device)
            tabular = batch["tabular"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                main, bag, pb_aux, context_alignment, concepts, malignancy = model(pb, pc, ubs, uv, tabular)
                loss_main = dice_loss(main, main_gt) + F.binary_cross_entropy_with_logits(main, main_gt)
                loss_pb = dice_loss(pb_aux, main_gt) + F.binary_cross_entropy_with_logits(pb_aux, main_gt)
                valid = torch.where(uv.reshape(-1) > 0)[0]
                if len(valid):
                    bag_flat = bag.reshape(-1, 1, 256, 256)[valid]
                    bag_gt_flat = ubs_gt.reshape(-1, 1, 256, 256)[valid]
                    loss_bag = dice_loss(bag_flat, bag_gt_flat) + F.binary_cross_entropy_with_logits(bag_flat, bag_gt_flat)
                else:
                    loss_bag = main.new_zeros(())
                loss_malignancy = malignancy_loss(malignancy.squeeze(-1), malignancy_gt.squeeze(-1))
                loss_concepts = sum(concept_loss(pred, clinical[:, i]) for i, pred in enumerate(concepts)) / len(concepts)
                loss = (
                    loss_main
                    + 0.5 * loss_pb
                    + 0.5 * loss_bag
                    + 2.0 * loss_malignancy
                    + loss_concepts
                    + 0.5 * context_alignment
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        validation_metrics, _, _, _ = evaluate(model, validation_loader, device)
        selection_score = (validation_metrics["Paired_DSC"] + validation_metrics["AUC"]) / 2.0
        log_rows.append(
            {
                "Epoch": epoch,
                "Training_Loss": np.mean(losses),
                "Validation_DSC": validation_metrics["Paired_DSC"],
                "Validation_AUC": validation_metrics["AUC"],
                "Validation_Accuracy": validation_metrics["Accuracy"],
                "Selection_Score": selection_score,
            }
        )
        pd.DataFrame(log_rows).to_csv(output / "training_log.csv", index=False)
        print(
            f"{args.mode} epoch {epoch:03d}: loss={np.mean(losses):.4f} "
            f"DSC={validation_metrics['Paired_DSC']:.4f} AUC={validation_metrics['AUC']:.4f} "
            f"score={selection_score:.4f}",
            flush=True,
        )
        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": args.mode,
                    "epoch": int(epoch),
                    "selection_score": float(selection_score),
                    "tabular_dim": len(train_set[0]["tabular"]),
                },
                output / "best_model.pth",
            )

    checkpoint = torch.load(output / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    test_metrics, paired, unpaired, patients = evaluate(model, test_loader, device, include_unpaired=True)
    pd.DataFrame([{**test_metrics, "Best_Epoch": checkpoint["epoch"], "Selection_Score": checkpoint["selection_score"]}]).to_csv(
        output / "test_metrics.csv", index=False
    )
    paired.to_csv(output / "test_paired_segmentation.csv", index=False)
    unpaired.to_csv(output / "test_unpaired_segmentation.csv", index=False)
    patients.to_csv(output / "test_patient_predictions.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no_text", "tabular"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    train(parser.parse_args())

if __name__ == "__main__":
    main()
