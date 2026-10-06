"""Segmentation, malignancy, and clinical-indicator evaluation metrics."""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import SIGN_COLUMNS


def segmentation_metrics(probability, reference, include_hd=True):
    pred = np.asarray(probability) > 0.5
    truth = np.asarray(reference) > 0.5
    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()
    tn = np.logical_and(~pred, ~truth).sum()
    denom = 2 * tp + fp + fn
    dsc = 2 * tp / denom if denom else 1.0
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    result = {
        "DSC": float(dsc),
        "IoU": float(iou),
        "VOE": float(1 - iou),
        "RVD": float((pred.sum() - truth.sum()) / truth.sum()) if truth.sum() else float("nan"),
        "PA": float((tp + tn) / (tp + tn + fp + fn)),
    }
    if include_hd:
        points_pred, points_truth = np.argwhere(pred), np.argwhere(truth)
        if len(points_pred) and len(points_truth):
            hd = max(
                directed_hausdorff(points_pred, points_truth)[0],
                directed_hausdorff(points_truth, points_pred)[0],
            )
        elif not len(points_pred) and not len(points_truth):
            hd = 0.0
        else:
            hd = float("nan")
        result["HD"] = float(hd)
    return result


def classification_metrics(truth, probability):
    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if set(truth) != {0, 1}:
        raise ValueError("Both benign and malignant cases are required for AUC")
    pred = (probability > 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(truth, probability)),
        "Accuracy": float(accuracy_score(truth, pred)),
        "Recall": float((pred[truth == 1] == 1).mean()),
        "Weighted-F1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
        "Macro-F1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "Malignant-F1": float(f1_score(truth, pred, zero_division=0)),
    }


def sign_metrics(records):
    output = []
    for index, name in enumerate(SIGN_COLUMNS):
        truth = np.asarray([row[f"sign_{index}_gt"] for row in records])
        pred = np.asarray([row[f"sign_{index}_pred"] for row in records])
        valid = truth != -100
        output.append({
            "Sign": name,
            "N": int(valid.sum()),
            "Accuracy": float(accuracy_score(truth[valid], pred[valid])) if valid.any() else float("nan"),
            "Weighted-F1": float(f1_score(truth[valid], pred[valid], average="weighted", zero_division=0)) if valid.any() else float("nan"),
        })
    return output
