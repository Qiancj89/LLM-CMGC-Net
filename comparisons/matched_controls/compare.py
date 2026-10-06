"""Summarize completed matched test runs without changing their outputs."""

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def read_run(path):
    result = path / "results" / "test_metrics.json"
    complete = path / "training_complete.json"
    if not result.exists():
        raise FileNotFoundError(f"Missing completed test: {result}")
    if not complete.exists():
        raise FileNotFoundError(f"Missing completed training: {complete}")
    data = json.loads(result.read_text(encoding="utf-8"))
    training = json.loads(complete.read_text(encoding="utf-8"))
    if data["training_epochs"] != 100:
        raise ValueError(f"Not a 100-epoch run: {path}")
    if data["run_id"] != training["run_id"]:
        raise ValueError(f"Test results are stale for {path}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("aggregation", "crop"), required=True)
    args = parser.parse_args()
    if args.study == "aggregation":
        runs = {
            mode: HERE / "aggregation_comparison" / mode
            for mode in ("mean", "attention", "sinkhorn")
        }
        output = HERE / "aggregation_comparison" / "comparison.csv"
    else:
        runs = {
            "full_field": HERE / "aggregation_comparison" / "sinkhorn",
            "mask_guided": HERE / "mask_free_retraining" / "mask_guided_control",
        }
        output = HERE / "mask_free_retraining" / "comparison.csv"
    data = {name: read_run(path) for name, path in runs.items()}
    if len({row["manifest_sha256"] for row in data.values()}) != 1:
        raise ValueError("Runs used different patient manifests")
    if len({(row["N_patients"], row["N_examinations"]) for row in data.values()}) != 1:
        raise ValueError("Runs have different test patient or examination counts")
    expected = ({name: (name, "none") for name in runs}
                if args.study == "aggregation" else
                {"full_field": ("sinkhorn", "none"),
                 "mask_guided": ("sinkhorn", "mask")})
    for name, row in data.items():
        if (row["aggregation"], row["crop_mode"]) != expected[name]:
            raise ValueError(f"Unexpected experimental condition in {runs[name]}")
    fields = ["N_patients", "N_examinations", "Paired_DSC", "Unpaired_DSC",
              "AUC", "Accuracy", "Recall", "Weighted-F1", "Macro-F1",
              "best_validation_epoch"]
    frame = pd.DataFrame([{"Condition": name, **{key: row[key] for key in fields}}
                          for name, row in data.items()])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Saved {output}")

if __name__ == "__main__":
    main()
