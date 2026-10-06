import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from context_model import ContextAblationNet
from run_context_ablation import ContextTestDataset, ROOT, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no_text", "tabular"], required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    result_dir = args.output / args.mode
    with open(result_dir / "tabular_normalization.json", encoding="utf-8") as handle:
        stats = json.load(handle)
    test_set = ContextTestDataset(
        ROOT / "PreprocessedTestData",
        ROOT / "clinical_summary_Huatuo-7B_testdata.csv",
        ROOT / "test_clinical_data.csv",
        "hfl/chinese-macbert-base",
        tabular_stats=stats,
    )
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(result_dir / "best_model.pth", map_location=device, weights_only=False)
    model = ContextAblationNet(args.mode, tabular_dim=checkpoint["tabular_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    metrics, paired, unpaired, patients = evaluate(model, loader, device, include_unpaired=True)
    pd.DataFrame(
        [{**metrics, "Best_Epoch": checkpoint["epoch"], "Selection_Score": checkpoint["selection_score"]}]
    ).to_csv(result_dir / "test_metrics.csv", index=False)
    paired.to_csv(result_dir / "test_paired_segmentation.csv", index=False)
    unpaired.to_csv(result_dir / "test_unpaired_segmentation.csv", index=False)
    patients.to_csv(result_dir / "test_patient_predictions.csv", index=False)

if __name__ == "__main__":
    main()
