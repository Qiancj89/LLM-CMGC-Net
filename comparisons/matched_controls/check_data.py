"""Read every experiment input before starting a GPU training run."""

import argparse
from pathlib import Path

from data import ExaminationDataset
from manifest import validate_manifest
from run import TEXT_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_manifest(args.manifest)
    for crop_mode in ("none", "mask"):
        for partition in ("train", "val", "test"):
            dataset = ExaminationDataset(manifest, partition, crop_mode, TEXT_MODEL)
            supplementary = 0
            for index in range(len(dataset)):
                sample = dataset[index]
                if sample["pc"].shape != (3, 256, 256):
                    raise ValueError(f"Invalid CEUS shape: {sample['record_id']}")
                if sample["pb"].shape != (3, 256, 256):
                    raise ValueError(f"Invalid paired B-mode shape: {sample['record_id']}")
                if sample["input_ids"].shape != (512,):
                    raise ValueError(f"Invalid text-token shape: {sample['record_id']}")
                supplementary += int(sample["uv"].sum())
            print(
                f"{partition}, crop={crop_mode}: {len(dataset)} examinations, "
                f"{supplementary} supplementary B-mode images, images/masks/text OK"
            )
    print("All data paths and readable inputs passed preflight.")

if __name__ == "__main__":
    main()
