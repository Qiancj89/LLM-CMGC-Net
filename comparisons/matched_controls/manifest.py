"""Build and validate manifests with a held-out test cohort."""

import argparse
import hashlib
import os
from pathlib import Path
import re

import pandas as pd

ROOT = Path(os.environ.get("LLM_CMGC_DATA_ROOT", Path(__file__).resolve().parents[2])).resolve()
PARTITIONS = {"train", "val", "test", "exclude"}


def source_records():
    rows = []
    legacy = {}
    for partition in ("train", "val"):
        for cls in ("Benign", "Malignant"):
            directory = ROOT / "dataset_split" / partition / cls
            for folder in directory.iterdir():
                if folder.is_dir():
                    legacy[(cls, folder.name)] = partition

    for cls in ("Benign", "Malignant"):
        for folder in (ROOT / "PreprocessedDataSet" / cls).iterdir():
            if folder.is_dir():
                rows.append({
                    "record_id": f"development/{cls}/{folder.name}",
                    "source_path": folder.relative_to(ROOT).as_posix(),
                    "legacy_partition": legacy.get((cls, folder.name), "unassigned"),
                })
    for folder in (ROOT / "PreprocessedTestData").iterdir():
        if folder.is_dir():
            rows.append({
                "record_id": f"test/{folder.name}",
                "source_path": folder.relative_to(ROOT).as_posix(),
                "legacy_partition": "test",
            })
    return pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)


def name_key(record_id):
    folder = record_id.rsplit("/", 1)[-1]
    return re.sub(r"[12]$", "", re.sub(r"^\d+_", "", folder)).casefold()


def write_template(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    table = source_records()
    table["patient_id"] = ""
    table["partition"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(table)} record rows to {path}")
    print("Fill patient_id and partition after identity review. Train/val overlap is allowed.")


def write_legacy_manifest(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    table = source_records()
    keys = table.record_id.map(name_key)
    test_keys = set(keys[table.legacy_partition == "test"])
    table["patient_id"] = keys.map(
        lambda key: "p_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    )
    table["partition"] = table.legacy_partition
    overlap = (table.record_id.str.startswith("development/") & keys.isin(test_keys))
    table.loc[overlap, "partition"] = "exclude"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(table)} source records to {path}.")
    print(f"Excluded {int(overlap.sum())} development records with names matching test records.")
    print("Patient IDs are derived from folder names; review spelling aliases and identities before publication.")
    validate_manifest(path)


def validate_manifest(path):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"record_id", "source_path", "patient_id", "partition"}
    if not required.issubset(table.columns):
        raise ValueError(f"Manifest needs columns: {sorted(required)}")
    table = table.copy()
    for column in required:
        table[column] = table[column].astype(str).str.strip()
    if table["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id in manifest")
    source = source_records().set_index("record_id")
    actual = table.set_index("record_id")
    if set(source.index) != set(actual.index):
        raise ValueError(
            f"Manifest/source mismatch: {len(set(source.index)-set(actual.index))} missing, "
            f"{len(set(actual.index)-set(source.index))} unknown records"
        )
    if not (actual["source_path"] == source.loc[actual.index, "source_path"]).all():
        raise ValueError("source_path was changed from the generated template")
    if (table["patient_id"] == "").any():
        raise ValueError("Every record needs an anonymous patient_id")
    if not table["partition"].isin(PARTITIONS).all():
        raise ValueError("partition must be train, val, test, or exclude for every record")
    if not {"train", "val", "test"}.issubset(set(table["partition"])):
        raise ValueError("All three partitions must be nonempty")
    test_ids = set(table.loc[table.partition == "test", "patient_id"])
    development_ids = set(table.loc[table.partition.isin(["train", "val"]), "patient_id"])
    if test_ids & development_ids:
        raise ValueError("A test patient_id also occurs in train or val")
    if (table.record_id.str.startswith("test/") & (table.partition != "test")).any():
        raise ValueError("All source test examinations must remain in test")
    if (table.record_id.str.startswith("development/") & (table.partition == "test")).any():
        raise ValueError("Development examinations cannot be added to the held-out test set")
    if (table.assign(name_key=table.record_id.map(name_key))
            .groupby("name_key")["patient_id"].nunique() > 1).any():
        raise ValueError("Same-name examination records have different patient IDs")
    for relative in table["source_path"]:
        folder = ROOT / relative
        if not (folder / "Original").is_dir() or not (folder / "Mask").is_dir():
            raise ValueError(f"Missing Original/Mask directory for {relative}")
    print("Held-out test manifest checks passed; train/val patient overlap is allowed.")
    for partition in ("train", "val", "test", "exclude"):
        subset = table[table.partition == partition]
        print(f"{partition}: {subset.patient_id.nunique()} patients, {len(subset)} examinations")
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--template", type=Path)
    group.add_argument("--from-legacy", type=Path)
    group.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.template:
        write_template(args.template)
    elif args.from_legacy:
        write_legacy_manifest(args.from_legacy)
    else:
        validate_manifest(args.validate)
