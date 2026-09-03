# FFM Evaluation Walkthrough Guide

This document describes how to execute the embedding generation, run the training configurations, and evaluate the checkpoints across the different feature fusion modes.

---


## 0. Latest Changes: Stability + Focal Loss (2026-07-30)

**Problem**: Previous run (`fusion_gated_d128_2026-07-23-15-16-42`) — Folds 2, 5, and both full models collapsed (MCC=0, threshold=0.0). Fold 4 alone succeeded (AUROC 0.7585, MCC 0.3055 on Test_60).

Three root causes fixed:
- **EGNN `residual=False`** across 10 layers → fixed with `residual=True` in `final_model.py`
- **Unweighted CE loss** vs. 15.61% positive class → replaced with `FocalLoss(alpha=[1.0, 5.4056], gamma=2.0)`
- **Threshold scan included 0.0** → masked collapse as "Recall=1.0, MCC=0" → fixed to start at 0.01 in `train.py` + `test.py`

**New file**: `loss.py` — `FocalLoss(alpha, gamma=2.0)` and `compute_pos_weight()`. Measured neg/pos ratio from Train_335: **5.4056** (15.61% positive, 84.39% negative, 66,208 residues).

**Training command**: `python train.py --fusion_mode gated --d_proj 128 --focal_gamma 2.0`

Ablation (weighted CE, no focal modulation): `--focal_gamma 0.0`

**Pass criteria**: All 5 folds MCC > 0; CV avg AUROC ≥ 0.70; CV avg MCC ≥ 0.25 on Test_60.

---


## 1. Setup and Environment

Ensure that all dependencies are installed in your virtual environment:
```bash
./venv/bin/pip install transformers accelerate
```

---

## 2. Generating ESM-2 Embeddings

Before running any fusion experiments (other than the classical `none` baseline), you must pre-extract and cache the ESM-2 (650M) per-residue embeddings:
```bash
./venv/bin/python generate_esm2_embeddings.py
```
*   **Input**: Protein sequences defined in the dataset files (`Train_335.pkl`, `Test_60.pkl`, `Test_315-28.pkl`, `UBtest_31-6.pkl`).
*   **Output**: Saved as float16 `.npy` files inside the directory `./Feature/esm2/`.
*   **VRAM Safeguards**: Automatically runs in `fp16` on GPU to save memory, and automatically falls back to float32 on CPU if an Out-Of-Memory (OOM) error occurs.

---

## 3. Training the Models

To train the models with cross-validation and a full training dataset run, use `train.py` with the `--fusion_mode` argument.

### A. Classical Baseline (Original Model features only)
```bash
source venv/bin/activate && python train.py --fusion_mode none --focal_gamma 2.0
```

### B. Naive Concat Fusion Mode
```bash
source venv/bin/activate && python train.py --fusion_mode concat --d_proj 128 --focal_gamma 2.0
```

### C. Gated Fusion Mode with Biophysics Supervision & Branch Regularization (Novel Proposed Method)
```bash
source venv/bin/activate && python train.py \
    --fusion_mode gated \
    --d_proj 128 \
    --focal_gamma 2.0 \
    --lambda_gate 0.1 \
    --lambda_agree 0.1
```
*   `--lambda_gate 0.1`: Loss weight for biophysics-supervised RSA gate loss (Idea 1). Teaches the gate to prefer PLM representations for surface-exposed residues (RSA $\to$ 1) and classical evolutionary features for buried residues (RSA $\to$ 0).
*   `--lambda_agree 0.1`: Loss weight for branch agreement regularization (Idea 2). Minimises prediction disagreement between EGNN (geometric) and Graph Transformer (topological) branches.

### Model Logging and Outputs
Checkpoints and execution logs are saved in mode-specific directories inside the `./Log/` folder:
`./Log/fusion_<fusion_mode>_d<d_proj>_<timestamp>/model/`

---

## 4. Run Smoke Tests (Quick Code Verification)

To verify that the forward and backward passes run successfully on your system without training to completion, append the `--smoke_test` flag:
```bash
source venv/bin/activate && python train.py --fusion_mode concat --smoke_test --focal_gamma 2.0
source venv/bin/activate && python train.py --fusion_mode gated --smoke_test --focal_gamma 2.0
source venv/bin/activate && python train.py --fusion_mode cross_attn --smoke_test --focal_gamma 2.0
```
*   **Behavior**: Restricts the datasets to 2 samples, runs exactly 1 fold and 1 epoch, and exits immediately.

---

## 5. Testing and Evaluation

Once training has completed for a specific mode, evaluate the saved checkpoints on all test sets (`Test_60`, `Test_315-28`, and `UBtest_31-6`) by pointing `test.py` to the appropriate model directory:

```bash
source venv/bin/activate && python test.py --fusion_mode <fusion_mode> --d_proj 128 --model_dir Log/fusion_<fusion_mode>_d128_<timestamp>/model/
```

### Gate Value Collection (Gated Mode only)
When executing evaluation in `gated` mode, the script automatically dumps per-residue gate values to a CSV file inside the specified `--model_dir` directory:
`<checkpoint_name>_gate_records.csv`
This file logs:
- `gate_value` (continuous scalar in range $[0, 1]$ where 1.0 represents classical features and 0.0 represents PLM features)
- `label` (binary binding site indicator)
- `rsa` (relative solvent accessibility from DSSP)

---

## 6. Verification Status

*   **`none` (baseline)**: Verified and completed (checkpoints and evaluation results exist in `Log/fusion_none_d128_2026-07-01-09-34-00/`).
*   **`concat`**: Passed smoke test. Ready for full training.
*   **`cross_attn`**: Passed smoke test (with projection LayerNorm and gain=1.0 patch). Ready for full training.
*   **`gated`**: Passed smoke test (after enabling `tanh=True` for EGNN coordinate updates to prevent scale explosion). Ready for full training.
