# GTE-PPIS FFM: Pipeline Walkthrough

This document describes how to execute the full pipeline — embedding generation, feature pre-processing, training, and evaluation — for each feature fusion mode.

---

## 0. Stability Fixes Applied (Context)

**2026-07-30 (commit `0475828`)**: EGNN collapse fixed:
- `residual=False` across 10 EGNN layers → fixed with `residual=True` in `final_model.py`
- Unweighted CE loss vs. 15.61% positive class → replaced with `FocalLoss(alpha=[1.0, pw], gamma)` where `pw` is the per-fold neg/pos ratio
- Threshold scan included 0.0 → masked collapse as Recall=1.0/MCC=0 → fixed to start at 0.01

**2026-09-03 (commit `d0c93b5`)**: GT residual fix:
- `transformer_residual=False` in `GraghTransformer` → corrected to `True`, matching paper's Eq. 8/10 design

**2026-09-03 (commit `3b04b2a`)**: Biophysics supervision added:
- `generate_rsa_features.py` created; `compute_auxiliary_losses()` added to `FinalModel`
- `--lambda_gate` (RSA gate MSE) and `--lambda_agree` (branch agreement KL) wired into `train.py`

**2026-09-12 (commit `d6864e3`)**: `cross_attn` mode removed:
- Confirmed unused in any production run; had a silent bug (`gate_val` always `None`)
- Valid `--fusion_mode` values are now: `none`, `concat`, `gated`

---

## 1. Setup and Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# DGL requires a manual install matching your CUDA version, e.g.:
# pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

---

## 2. Generating ESM-2 Embeddings

Required for all fusion modes except `none`. Run once across all dataset splits:

```bash
python generate_esm2_embeddings.py
```

- **Input**: All protein IDs from `Dataset/Train_335.pkl`, `Test_60.pkl`, `Test_315-28.pkl`, `UBtest_31-6.pkl`
- **Output**: `Feature/esm2/{ID}.npy` (float16, shape L×1280)
- **VRAM**: Runs in fp16 on GPU; automatically falls back to CPU float32 on OOM

---

## 3. Generating RSA Features

Required for gated mode with RSA gate supervision (`--lambda_gate > 0`). Requires PDB files to be downloaded to `PDB/`.

```bash
python generate_rsa_features.py
```

- **Input**: PDB files from `./PDB/`; protein lists from the same 4 dataset pickles
- **Output**: `Feature/rsa/{ID}.npy` (float32, L-dim, range [0,1], normalised by per-residue max ASA)
- **Fallback**: Proteins without PDB files get 0.5 (neutral fallback). `2j3rA` is skipped entirely (removed from Train\_335 during training).

---

## 4. Training the Models

Training entry point: `train.py`. Runs 5-fold cross-validation on Train\_335 (334 proteins after removing `2j3rA`) then trains a full model on all data.

### A. Classical Backbone — No ESM-2 (Baseline)
```bash
source venv/bin/activate && python train.py --fusion_mode none --focal_gamma 2.0
```

### B. Naive Concat Fusion
```bash
source venv/bin/activate && python train.py --fusion_mode concat --d_proj 128 --focal_gamma 2.0
```

### C. Gated Fusion — Full Proposed Method
```bash
source venv/bin/activate && python train.py \
    --fusion_mode gated \
    --d_proj 128 \
    --focal_gamma 2.0 \
    --lambda_gate 0.1 \
    --lambda_agree 0.1
```

- `--lambda_gate 0.1`: Weight for RSA-supervised gate MSE loss. Teaches the scalar gate to prefer classical features for buried residues (RSA→0) and PLM features for surface-exposed residues (RSA→1).
- `--lambda_agree 0.1`: Weight for branch-agreement loss. Penalises divergence between EGNN and GT branch softmax predictions.

### D. Gated Fusion — Ablation (no gate supervision)
```bash
source venv/bin/activate && python train.py \
    --fusion_mode gated \
    --d_proj 128 \
    --focal_gamma 0.0 \
    --lambda_gate 0.0 \
    --lambda_agree 0.0
```
Use this to isolate the effect of RSA supervision from the architecture changes.

### Outputs
Checkpoints and training logs saved to:
```
./Log/fusion_<fusion_mode>_d<d_proj>_<timestamp>/
├── training.log
└── model/
    ├── Fold1_best_model.pkl  ... Fold5_best_model.pkl
    └── Full_model_<epoch>.pkl
```

Checkpoint criterion: best validation AUPRC across 50 epochs per fold.

---

## 5. Smoke Tests (Quick Code Verification)

Restricts to 2 samples, 1 fold, 1 epoch — verifies forward/backward pass without full training:

```bash
source venv/bin/activate && python train.py --fusion_mode none   --smoke_test --focal_gamma 2.0
source venv/bin/activate && python train.py --fusion_mode concat --smoke_test --focal_gamma 2.0
source venv/bin/activate && python train.py --fusion_mode gated  --smoke_test --focal_gamma 2.0
```

---

## 6. Testing and Evaluation

Entry point: `test.py`. Requires `--model_dir` pointing to the checkpoint folder of a completed training run.

```bash
source venv/bin/activate && python test.py \
    --fusion_mode <fusion_mode> \
    --d_proj 128 \
    --model_dir Log/fusion_<fusion_mode>_d128_<timestamp>/model/
```

Evaluates on all three benchmarks: `Test_60`, `Test_315-28`, `UBtest_31-6`.

### Threshold Protocol (No Test-Label Leakage)
For each fold checkpoint `FoldK_best_model.pkl`:
1. Reconstruct the fold-K validation split by replaying `KFold(n_splits=5, shuffle=True, random_state=SEED=2024)` on Train\_335.
2. Evaluate on the validation split; search τ ∈ [0.01, 0.99] to maximise validation F1.
3. Lock this τ. Apply it to the test set without consulting test labels.

For full models (`Full_model_*.pkl`): uses the mean of the 5 fold thresholds.

### Gate Value Collection (Gated Mode Only)
In `gated` mode, per-residue gate values are written to a CSV alongside each checkpoint's test evaluation:
```
<model_dir>/<checkpoint_name>_gate_records.csv
```
Columns:
- `gate_value`: continuous scalar ∈ [0,1]; 1.0 = classical features, 0.0 = PLM features
- `label`: binary interface indicator (ground truth)
- `rsa`: relative solvent accessibility loaded from `Feature/rsa/{id}.npy` (FreeSASA-derived, not from DSSP)

---

## 7. Current Status and Results

All numbers are 5-fold CV averages with validation-locked thresholds, from actual log files.

| Run | Fusion | γ | GT res | Gate sup | Test\_60 AUROC | Test\_60 AUPRC | UBtest AUROC | UBtest AUPRC | Log |
|---|---|---|---|---|---|---|---|---|---|
| No-ESM2 (Aug 25) | `none` | 2.0 | True | — | 0.8537 | 0.5518 | 0.7611 | 0.3115 | `fusion_none_d128_2026-08-25-11-28-09` |
| Unsup. gate (Aug 13) | `gated` | 2.0 | False† | None | 0.8236 | 0.5056 | 0.8191 | 0.4182 | `fusion_gated_d128_2026-08-13-11-53-23` |
| RSA-sup. gate (Sep 11) | `gated` | 0.0 | True | RSA λ=0.1 | 0.8116 | 0.4863 | 0.8159 | 0.4248 | `fusion_gated_d128_2026-09-11-15-22-42` |
| `concat` | — | — | — | — | not run | — | — | — | — |

† Pre-`d0c93b5` bug: GT `transformer_residual=False`.  
Published GTE-PPIS baseline: Test\_60 AUROC 0.873, AUPRC 0.611 (Wang et al. 2025).

**ESM-2 trade-off**: Gated fusion consistently improves UBtest AUPRC/MCC (+0.107 / +0.099 vs. no-ESM2 backbone) at the cost of Test\_60 performance. This pattern holds under both unsupervised and RSA-supervised gating.

**RSA supervision**: Did not improve Test\_60 or Test\_315-28 AUROC/AUPRC. Marginal UBtest AUPRC improvement (+0.0066) is below expected single-seed variance. Three confounders (γ, GT residual, λ\_gate) changed simultaneously — a confounder isolation run is the immediate next step.

Full per-fold tables and delta analysis: [`rsa_supervision_ablation_report.tex`](rsa_supervision_ablation_report.tex).

---

## 8. Where to Look for What

| Task | File(s) |
|---|---|
| Change model hyperparameters (epochs, hidden dim, layers) | `data_generator.py` (constants at top) |
| Change EGNN architecture | `EGNN_model.py`, `final_model.py` line 50–52 |
| Change GT architecture | `GraphTransformer_Block.py`, `final_model.py` line 54–56 |
| Change fusion logic | `fusion_module.py` |
| Add a new fusion mode | `fusion_module.py` → `FeatureFusionModule`, `final_model.py` → `__init__` dimension logic, `train.py`/`test.py` `choices=` list |
| Change loss function | `loss.py`, `final_model.py` line 68 |
| Change auxiliary loss targets | `final_model.py` → `compute_auxiliary_losses()` |
| Debug NaN/collapse | Enable debug prints in `fusion_module.py` (already present); check EGNN `residual=True` and `tanh=True` in `final_model.py` |
| Inspect gate behaviour | Load checkpoint, call `model.forward(...)`, read `model.last_gate_val`; or load a `_gate_records.csv` from a test run |

