import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("LLM_CMGC_DATA_ROOT", CODE_ROOT)).resolve()
STATS = CODE_ROOT / "analysis" / "statistics"
if str(STATS) not in sys.path:
    sys.path.insert(0, str(STATS))

from bootstrap_ci import bootstrap_classification, bootstrap_mean, paired_auc_bootstrap

ITERATIONS = 5000
SEED = 20260708
OUTPUT = Path(__file__).resolve().parent / "results"
rng = np.random.default_rng(SEED)

conditions = {
    "No clinical context": {
        "classification": OUTPUT / "no_text" / "test_patient_predictions.csv",
        "paired": OUTPUT / "no_text" / "test_paired_segmentation.csv",
        "unpaired": OUTPUT / "no_text" / "test_unpaired_segmentation.csv",
    },
    "Structured tabular context": {
        "classification": OUTPUT / "tabular" / "test_patient_predictions.csv",
        "paired": OUTPUT / "tabular" / "test_paired_segmentation.csv",
        "unpaired": OUTPUT / "tabular" / "test_unpaired_segmentation.csv",
    },
    "Huatuo-7B narrative": {
        "classification": ROOT / "inference_ot2" / "Huatuo-7B" / "results_cls_patient.csv",
        "paired": ROOT / "inference_ot2" / "Huatuo-7B" / "results_seg_paired.csv",
        "unpaired": ROOT / "inference_ot2" / "Huatuo-7B" / "results_seg_unpaired.csv",
    },
}

rows = []
classification = {}
for condition, paths in conditions.items():
    cls = pd.read_csv(paths["classification"])
    paired = pd.read_csv(paths["paired"])
    unpaired = pd.read_csv(paths["unpaired"])
    classification[condition] = cls
    for row in bootstrap_classification(cls, ITERATIONS, rng):
        if row["Endpoint"] in {"AUC", "Accuracy", "Sensitivity", "Weighted-F1"}:
            rows.append({"Condition": condition, **row})
    for endpoint, frame, cluster in [
        ("Paired DSC", paired, None),
        ("Unpaired DSC", unpaired, "Patient_Name"),
    ]:
        estimate, low, high = bootstrap_mean(frame, "DSC", ITERATIONS, rng, cluster)
        rows.append(
            {
                "Condition": condition,
                "Endpoint": endpoint,
                "Estimate": estimate,
                "CI_Lower": low,
                "CI_Upper": high,
            }
        )

pd.DataFrame(rows).to_csv(OUTPUT / "context_ablation_95ci.csv", index=False, float_format="%.6f")

reference = classification["Huatuo-7B narrative"]
comparisons = []
for condition in ["No clinical context", "Structured tabular context"]:
    difference, low, high, p_value = paired_auc_bootstrap(
        reference, classification[condition], ITERATIONS, rng
    )
    comparisons.append(
        {
            "Comparison": f"{condition} minus Huatuo-7B narrative",
            "AUC_Difference": difference,
            "CI_Lower": low,
            "CI_Upper": high,
            "Two_Sided_P": p_value,
        }
    )
pd.DataFrame(comparisons).to_csv(
    OUTPUT / "context_ablation_auc_differences.csv", index=False, float_format="%.6f"
)
