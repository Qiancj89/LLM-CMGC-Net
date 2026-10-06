# LLM-CMGC-Net: research code

Code accompanying the ovarian-tumor multimodal ultrasound study. The repository
separates the proposed model, historical ablations, matched controls, other
comparators, and supporting analyses. It contains **code only**. Clinical
spreadsheets, ultrasound images, masks, generated narratives, patient manifests,
individual predictions, pretrained
weights, and trained checkpoints are not included.

## Layout

| Directory | Purpose |
| --- | --- |
| `main/` | Proposed model, training, and held-out inference |
| `preprocessing/` | Source image conversion, legacy split/augmentation, and LLM narrative generation |
| `ablations/baseline_attention_mil/` | Historical attention-MIL baseline |
| `ablations/ot_orthogonal_no_aux/` | Historical OT variant with orthogonality regularization and without the auxiliary decoder |
| `ablations/clinical_context/` | No-text and structured-tabular clinical-context controls |
| `comparisons/sota/` | Text-enabled and image-only comparator training/evaluation |
| `comparisons/matched_controls/` | 100-epoch mean/attention/Sinkhorn and mask-preparation controls |

See [CODE_MAP.md](CODE_MAP.md) for the manuscript-to-code mapping. The three
root-level Python modules are import-only compatibility shims for scripts that
originally imported `vl_cmgc_ot_net`, `train_vl_ot`, or `inference_vl_ot` from
the project root. Run the actual programs in the folders above.

## Environment and private data

Python with PyTorch/CUDA, torchvision, transformers, OpenCV, pandas, NumPy,
SciPy, scikit-learn, tqdm, albumentations, and nibabel is required. See
`requirements.txt` for import-level dependencies. The original experiment
used locally available ConvNeXt-Tiny weights and the
`hfl/chinese-macbert-base` tokenizer/encoder. Narrative generation requires
separately obtained LLM weights and additional model-specific dependencies.
Package versions and CUDA compatibility should be recorded for a new run;
`requirements.txt` is not an exact historical environment lockfile.

Keep hospital data **outside this repository**. For scripts under
`comparisons/`, and `ablations/clinical_context/`, set
`LLM_CMGC_DATA_ROOT` to the private data directory. Example in Windows CMD or
Anaconda Prompt:

```cmd
conda activate pytorch
set CODE_ROOT=C:\path\to\LLM-CMGC-Net-GitHub
set LLM_CMGC_DATA_ROOT=D:\path\to\private-data
set MANIFEST=D:\path\to\private-data\patient_manifest.csv
set HF_HUB_OFFLINE=1
```

The equivalent PowerShell assignment is
`$env:LLM_CMGC_DATA_ROOT = 'D:\path\to\private-data'`. Set
`HF_HUB_OFFLINE=1` only when required model assets are cached. Replace both
example paths with your actual clone and approved private-data locations.
`MANIFEST` points to a separate, non-public CSV; it need not live directly
under the data root.

The archived `main/`, `preprocessing/`, and two historical model-ablation
scripts still use paths relative to the **working directory**. Run them from
the private data root, or edit their configuration dictionaries to point to
your approved local data. For example, from CMD after the setup above:

```cmd
cd /d %LLM_CMGC_DATA_ROOT%
python "%CODE_ROOT%\main\train_vl_ot.py"
python "%CODE_ROOT%\main\inference_vl_ot.py"
```

Check the script's model name, narrative CSV, checkpoint and output settings
before running. The training and inference defaults refer to different LLM
narrative files, as in the original scripts. These commands are examples, not
an assertion that all private inputs are distributed here.

## Matched-control reproduction

The scripts can generate aggregate result CSVs after the 100-epoch experiments.
The original private manifest, results, and model weights are absent. Use
a locally reviewed manifest with `patient_id`, `partition`, and the source
record columns required by `manifest.py`; keep it outside Git. Validate and
inspect the source data before training:

```cmd
python "%CODE_ROOT%\comparisons\matched_controls\manifest.py" --validate "%MANIFEST%"
python "%CODE_ROOT%\comparisons\matched_controls\check_data.py" --manifest "%MANIFEST%"
```

The matched scripts load images/masks from `PreprocessedDataSet` and
`PreprocessedTestData`, development narratives/labels from
`clinical_summary_Huatuo-7B.csv` and `clinical_data.csv`, and their test
counterparts from the private data root. Example full-field Sinkhorn run:

```cmd
python "%CODE_ROOT%\comparisons\matched_controls\run.py" train --manifest "%MANIFEST%" --study aggregation --aggregation sinkhorn
python "%CODE_ROOT%\comparisons\matched_controls\run.py" test --manifest "%MANIFEST%" --study aggregation --aggregation sinkhorn
```

Use `--aggregation mean` or `--aggregation attention` for the other two
aggregation controls. The mask-guided control uses `--study crop
--aggregation sinkhorn --crop mask` for both `train` and `test`. Each condition
trains for exactly 100 epochs and evaluates the selected validation checkpoint
on the held-out test set. `compare.py --study aggregation` and `compare.py
--study crop` collate the respective local runs. Generated outputs contain
potentially identifying per-record results and are ignored by Git.

The legacy development train/validation assignments may contain repeated
examinations of the same patient. The matched-control manifest validator
checks that test patient identifiers do not appear in development partitions,
but a hospital-verified identity mapping remains preferable. Do not interpret
any aggregate CSVs as independent validation evidence. The original
`preprocessing/split_dataset.py` splits examination folders, not confirmed
unique patient identifiers.

## Publication and sharing

Review every output before publication. Do not commit real patient names,
clinical narratives, imaging, masks, manifests, pretrained/trained weights,
or patient-level prediction tables. The interpretability visual-case script
accepts a **private** `--cases-file` CSV (`Outcome,Patient_Name`) instead of
embedding names in public code. The data-quality correction script modifies
private label files and should be run only after local review and backup.

No license has been selected for this release. Add a license only after the
code owner and institution approve public distribution.
