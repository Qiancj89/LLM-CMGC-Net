"""Train or test matched aggregation and mask-free CMPB controls."""

import argparse
import hashlib
import json
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import ExaminationDataset
from manifest import validate_manifest
from metrics import classification_metrics, segmentation_metrics, sign_metrics
from model_variants import build_model

HERE = Path(__file__).resolve().parent
TEXT_MODEL = "hfl/chinese-macbert-base"
SEED = 42


def run_spec(args):
    if args.study == "aggregation":
        if args.aggregation not in {"mean", "attention", "sinkhorn"}:
            raise ValueError("Aggregation study requires --aggregation mean, attention, or sinkhorn")
        if args.crop != "none":
            raise ValueError("Aggregation study fixes --crop none across all variants")
        return args.aggregation, "none", HERE / "aggregation_comparison" / args.aggregation
    if args.aggregation != "sinkhorn" or args.crop != "mask":
        raise ValueError("Crop control requires --aggregation sinkhorn --crop mask")
    return "sinkhorn", "mask", HERE / "mask_free_retraining" / "mask_guided_control"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dice_loss(logit, target):
    prob = torch.sigmoid(logit)
    intersection = (prob * target).sum(dim=(-2, -1))
    total = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1 - ((2 * intersection + 1e-5) / (total + 1e-5)).mean()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def training_loss(outputs, batch, criterion_mal, criterion_sign):
    main, bag, paired_aux, itc, signs, malignant = outputs
    gt = batch["mask_main"]
    main_loss = dice_loss(main, gt) + F.binary_cross_entropy_with_logits(main, gt)
    aux_loss = dice_loss(paired_aux, gt) + F.binary_cross_entropy_with_logits(paired_aux, gt)
    valid = batch["uv"].bool().reshape(-1)
    bag_flat = bag.reshape(-1, 1, 256, 256)
    mask_flat = batch["ubs_masks"].reshape(-1, 1, 256, 256)
    bag_loss = dice_loss(bag_flat[valid], mask_flat[valid])
    bag_loss = bag_loss + F.binary_cross_entropy_with_logits(bag_flat[valid], mask_flat[valid])
    malignant_loss = criterion_mal(malignant.reshape(-1), batch["mal"].reshape(-1))
    available = [criterion_sign(logits, batch["clin"][:, index])
                 for index, logits in enumerate(signs)
                 if (batch["clin"][:, index] != -100).any()]
    sign_loss = torch.stack(available).mean() if available else main_loss.new_zeros(())
    return main_loss + 0.5 * aux_loss + 0.5 * bag_loss + 2 * malignant_loss + sign_loss + 0.5 * itc


def evaluate(model, dataset, device, full_metrics=False):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    predictions, paired, supplementary = [], [], []
    record_lookup = {record["record_id"]: record for record in dataset.records}
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluate {dataset.partition}"):
            batch = move_batch(batch, device)
            outputs = model(
                batch["pb"], batch["pc"], batch["ubs"], batch["uv"],
                batch["input_ids"], batch["attention_mask"],
            )
            main, bag, _, _, signs, malignant = outputs
            record_id = batch["record_id"][0]
            patient_id = batch["patient_id"][0]
            row = {
                "record_id": record_id,
                "patient_id": patient_id,
                "Mal_GT": int(batch["mal"].item()),
                "Mal_Prob": float(torch.sigmoid(malignant).item()),
            }
            row["Mal_Pred"] = int(row["Mal_Prob"] > 0.5)
            for index, logits in enumerate(signs):
                row[f"sign_{index}_gt"] = int(batch["clin"][0, index].item())
                row[f"sign_{index}_pred"] = int(logits.argmax(dim=1).item())
            predictions.append(row)
            metric = segmentation_metrics(
                torch.sigmoid(main[0, 0]).cpu().numpy(),
                batch["mask_main"][0, 0].cpu().numpy(),
                include_hd=full_metrics,
            )
            paired.append({"record_id": record_id, "patient_id": patient_id, **metric})
            if full_metrics:
                names = record_lookup[record_id]["supplementary"]
                for index, image_path in enumerate(names):
                    metric = segmentation_metrics(
                        torch.sigmoid(bag[0, index, 0]).cpu().numpy(),
                        batch["ubs_masks"][0, index, 0].cpu().numpy(),
                    )
                    supplementary.append({
                        "record_id": record_id,
                        "patient_id": patient_id,
                        "image_name": image_path.name,
                        **metric,
                    })
    patient_predictions = []
    for patient_id, group in pd.DataFrame(predictions).groupby("patient_id", sort=True):
        if group["Mal_GT"].nunique() != 1:
            raise ValueError(f"Conflicting malignancy labels for patient {patient_id}")
        probability = float(group["Mal_Prob"].mean())
        patient_predictions.append({
            "patient_id": patient_id,
            "N_examinations": len(group),
            "Mal_GT": int(group["Mal_GT"].iloc[0]),
            "Mal_Prob": probability,
            "Mal_Pred": int(probability > 0.5),
        })
    classification = classification_metrics(
        [row["Mal_GT"] for row in patient_predictions],
        [row["Mal_Prob"] for row in patient_predictions],
    )
    examination_classification = classification_metrics(
        [row["Mal_GT"] for row in predictions],
        [row["Mal_Prob"] for row in predictions],
    )
    summary = {
        "N_examinations": len(predictions),
        "N_patients": len(patient_predictions),
        "Paired_DSC": float(np.mean([row["DSC"] for row in paired])),
        **classification,
        **{f"Examination_{key}": value for key, value in examination_classification.items()},
    }
    if full_metrics:
        summary["Unpaired_DSC"] = float(np.mean([row["DSC"] for row in supplementary]))
        summary["N_supplementary_images"] = len(supplementary)
        for name in ("IoU", "HD", "VOE", "RVD", "PA"):
            summary[f"Paired_{name}"] = finite_mean([row[name] for row in paired])
            summary[f"Unpaired_{name}"] = finite_mean([row[name] for row in supplementary])
    return summary, predictions, patient_predictions, paired, supplementary


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def train(args, manifest, mode, crop, run_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    if args.epochs != 100:
        raise ValueError("The comparison protocol requires exactly 100 epochs")
    model_path = run_dir / "model" / "best_model.pt"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(f"Existing model at {model_path}; use --overwrite deliberately")
    (run_dir / "training_complete.json").unlink(missing_ok=True)
    run_id = uuid4().hex
    set_seed(SEED)
    device = torch.device("cuda")
    train_ds = ExaminationDataset(manifest, "train", crop, TEXT_MODEL, repeats=10, seed=SEED)
    val_ds = ExaminationDataset(manifest, "val", crop, TEXT_MODEL, seed=SEED)
    if set(record["malignancy"] for record in val_ds.records) != {0, 1}:
        raise ValueError("Validation set needs both classes for AUC checkpoint selection")
    labels = np.asarray(train_ds.labels(), dtype=int)
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("Training set needs both classes")
    weights = 1 / counts[labels]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(labels),
        replacement=True, generator=torch.Generator().manual_seed(SEED),
    )
    loader = DataLoader(train_ds, batch_size=4, sampler=sampler, num_workers=0, pin_memory=True)
    model = build_model(mode, TEXT_MODEL).to(device)
    text_params = list(model.text_encoder.parameters())
    base_params = [parameter for name, parameter in model.named_parameters()
                   if "text_encoder" not in name]
    optimizer = torch.optim.AdamW(
        [{"params": base_params, "lr": 1e-4}, {"params": text_params, "lr": 1e-5}],
        weight_decay=3e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler("cuda")
    criterion_mal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
    criterion_sign = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
    run_dir.joinpath("model").mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("results").mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -float("inf")
    manifest_hash = file_sha256(args.manifest)
    for epoch in range(100):
        train_ds.set_epoch(epoch)
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"{mode}/{crop} epoch {epoch + 1}/100"):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                outputs = model(
                    batch["pb"], batch["pc"], batch["ubs"], batch["uv"],
                    batch["input_ids"], batch["attention_mask"],
                )
                loss = training_loss(outputs, batch, criterion_mal, criterion_sign)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()
        val, _, _, _, _ = evaluate(model, val_ds, device)
        score = (val["Paired_DSC"] + val["AUC"]) / 2
        history.append({"epoch": epoch + 1, "train_loss": np.mean(losses),
                        "validation_DSC": val["Paired_DSC"], "validation_AUC": val["AUC"],
                        "validation_Accuracy": val["Accuracy"], "selection_score": score})
        pd.DataFrame(history).to_csv(run_dir / "results" / "history.csv", index=False)
        print(f"Epoch {epoch + 1}: DSC={val['Paired_DSC']:.4f} AUC={val['AUC']:.4f} score={score:.4f}")
        if score > best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "validation": val,
                "selection_score": score,
                "aggregation": mode,
                "crop_mode": crop,
                "manifest_sha256": manifest_hash,
                "seed": SEED,
                "run_id": run_id,
            }, model_path)
    (run_dir / "training_complete.json").write_text(
        json.dumps({"epochs": 100, "best_score": best_score,
                    "manifest_sha256": manifest_hash, "aggregation": mode,
                    "crop_mode": crop, "run_id": run_id}, indent=2), encoding="utf-8"
    )


def test(args, manifest, mode, crop, run_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("Testing requires a CUDA GPU")
    complete = run_dir / "training_complete.json"
    if not complete.exists():
        raise FileNotFoundError(f"100-epoch training not completed: {complete}")
    metadata = json.loads(complete.read_text(encoding="utf-8"))
    if metadata["epochs"] != 100 or metadata["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("Incomplete training or manifest differs from training")
    set_seed(SEED)
    device = torch.device("cuda")
    dataset = ExaminationDataset(manifest, "test", crop, TEXT_MODEL, seed=SEED)
    if {record["malignancy"] for record in dataset.records} != {0, 1}:
        raise ValueError("Test set needs both classes for AUC")
    model = build_model(mode, TEXT_MODEL).to(device)
    checkpoint = torch.load(run_dir / "model" / "best_model.pt", map_location=device, weights_only=False)
    if (checkpoint["manifest_sha256"] != metadata["manifest_sha256"]
            or checkpoint["aggregation"] != mode or checkpoint["crop_mode"] != crop
            or checkpoint["run_id"] != metadata["run_id"]):
        raise ValueError("Checkpoint is incompatible with requested test condition")
    model.load_state_dict(checkpoint["model_state_dict"])
    summary, predictions, patient_predictions, paired, supplementary = evaluate(
        model, dataset, device, full_metrics=True
    )
    summary.update({
        "aggregation": mode,
        "crop_mode": crop,
        "best_validation_epoch": checkpoint["epoch"],
        "best_validation_score": checkpoint["selection_score"],
        "manifest_sha256": checkpoint["manifest_sha256"],
        "training_epochs": 100,
        "run_id": checkpoint["run_id"],
    })
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(result_dir / "test_predictions.csv", index=False)
    pd.DataFrame(patient_predictions).to_csv(result_dir / "test_patient_predictions.csv", index=False)
    pd.DataFrame(paired).to_csv(result_dir / "test_paired_segmentation.csv", index=False)
    pd.DataFrame(supplementary).to_csv(result_dir / "test_unpaired_segmentation.csv", index=False)
    pd.DataFrame(sign_metrics(predictions)).to_csv(result_dir / "test_clinical_indicators.csv", index=False)
    (result_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--study", choices=("aggregation", "crop"), required=True)
    parser.add_argument("--aggregation", choices=("mean", "attention", "sinkhorn"), default="sinkhorn")
    parser.add_argument("--crop", choices=("none", "mask"), default="none")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    mode, crop, run_dir = run_spec(args)
    manifest = validate_manifest(args.manifest)
    if args.action == "train":
        train(args, manifest, mode, crop, run_dir)
    else:
        test(args, manifest, mode, crop, run_dir)

if __name__ == "__main__":
    main()
