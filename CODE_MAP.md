# Manuscript-to-code map

| Manuscript component | Code | Notes |
| --- | --- | --- |
| LLM-CMGC-Net / OT-MIL model | `main/vl_cmgc_ot_net.py` | Shared visual encoder, text encoder, OT-based bag aggregation, segmentation and classification heads |
| Main training and evaluation | `main/train_vl_ot.py`, `main/inference_vl_ot.py` | Original experimental entrypoints and configuration |
| Image and narrative preparation | `preprocessing/` | Historical preparation scripts; private source files are omitted |
| Baseline attention MIL ablation | `ablations/baseline_attention_mil/` | Archived baseline model, training and inference |
| OT with orthogonality / no auxiliary decoder | `ablations/ot_orthogonal_no_aux/` | Archived intermediate model, training and inference |
| Clinical context ablation | `ablations/clinical_context/` | No-text and tabular controls, checkpoint testing, summary bootstrap |
| SOTA/task comparators | `comparisons/sota/` | Text-enabled and image-only comparators; 100-epoch defaults and task-specific checkpoint selection |
| Mean vs attention vs Sinkhorn | `comparisons/matched_controls/` | Same backbone/input/training strategy; `compare.py --study aggregation` generates aggregate metrics |
| Full-field vs reference-mask-guided input | `comparisons/matched_controls/` | Retrained controls; `compare.py --study crop` generates aggregate metrics |

The `matched_controls` experiments are separate from the archived main and
historical ablation scripts. Their results are not supplied here. Dataset access,
model weights, and a verified patient manifest are needed for full reruns.
