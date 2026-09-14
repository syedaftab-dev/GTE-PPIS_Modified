# GTE-PPIS Feature Fusion Module (FFM) Research Summary

This document summarises the implementation and evaluation setup for the Feature Fusion Module (FFM) in the GTE-PPIS model. It integrates evolutionary/PLM-derived features (ESM-2 embeddings) with classical handcrafted sequence features (PSSM + HMM) via a learned fusion mechanism, and introduces two auxiliary training objectives grounded in structural biophysics.

## Specific Gap and Motivation

Prior work (e.g., DHEG, *Briefings in Bioinformatics* 2026) showed that naive concatenation or direct replacement of PSSM/HMM handcrafted features with ESM-2 embeddings underperforms classical handcrafted feature baselines (MCC dropping as low as 0.086 for standalone PLMs). The unresolved question is whether a **learned fusion mechanism**—as opposed to static concatenation or replacement—enables PLM embeddings to contribute a complementary signal, and whether **biophysical priors** (residue solvent accessibility) can guide that mechanism toward semantically meaningful fusion behaviour.

This work implements and evaluates a Feature Fusion Module (FFM) placed between raw node features and downstream GNN branches (`EGNN` and `GraphTransformer`), specifically isolating evolutionary streams for learned fusion while leaving structural/geometric streams separate and untouched.

---

## Architectural Specification (FFM)

Two input streams per residue $i$ are processed:
1. **`classical_i`**: Handcrafted evolutionary features — PSSM (20d) + HMM (20d) → 40d vector.
2. **`plm_i`**: Pre-extracted ESM-2 (650M) per-residue embeddings (1280d), cached as float16.

*Structural/geometric features (DSSP 14d, residue atom features 7d) are kept separate and passed directly to downstream branches without modification, isolating the impact of evolutionary fusion.*

### Steps and Modes

1. **Projection**: Both streams are projected to shared latent dimension $d$ (default $d=128$, CLI: `--d_proj`):
   - `classical_proj = LayerNorm(Linear(40→d)(classical_i))`
   - `plm_proj = LayerNorm(Linear(1280→d)(plm_i))`
   - Both projections use Xavier uniform initialisation (gain=1.0).

2. **Fusion variants** (selected via `--fusion_mode`):
   - **`none`**: Classical-only baseline; FFM is bypassed entirely. Input to GNN branches is the original 61d node feature.
   - **`concat`**: Concatenate projected streams → 2d + DSSP (14d) + resAF (7d) = **277d** input to GNN branches.
   - **`gated`**: Per-residue scalar gate $g_i \in [0,1]$:
     $$g_i = \sigma\!\bigl(\text{Linear}([\mathbf{c}_\text{proj}, \mathbf{p}_\text{proj}]\to 1)\bigr)$$
     $$\mathbf{f}_i = g_i \cdot \mathbf{c}_\text{proj} + (1-g_i) \cdot \mathbf{p}_\text{proj}$$
     Output: d + DSSP (14d) + resAF (7d) = **149d** input to GNN branches.

   > **Note:** The `cross_attn` mode was removed from the codebase (commit `d6864e3`, Sep 12 2026) after being confirmed unused in any production run and having a silent bug where `gate_val` was always `None`.

3. **Re-concatenation**: Final node features for both EGNN and GT branches:
   $$\mathbf{x}_i = [\mathbf{f}_i \;\|\; \text{DSSP}_i \;\|\; \text{resAF}_i]$$

---

## Novel Auxiliary Training Objectives

### Idea 1: Biophysics-Supervised Gate Loss (`--lambda_gate`)

The gate is taught to correlate with residue solvent exposure. The target gate value is derived from FreeSASA-computed RSA:
$$g^*_i = 1 - \text{RSA}_i, \quad \mathcal{L}_\text{gate} = \frac{1}{N}\sum_i (\hat{g}_i - g^*_i)^2$$

Interpretation: buried residues (RSA → 0) should prefer classical structural features (gate → 1); surface-exposed residues (RSA → 1) should prefer PLM sequence context (gate → 0).

RSA values are computed by `generate_rsa_features.py` (FreeSASA library, normalised by per-residue maximum ASA). Fallback to 0.5 for proteins without PDB structures (`2j3rA` is the only known case in Train\_335).

### Idea 2: Branch Agreement Regularisation (`--lambda_agree`)

Minimises prediction disagreement between the EGNN (geometric) and Graph Transformer (topological) branches:
$$\mathcal{L}_\text{agree} = \text{MSE}\bigl(\text{softmax}(\mathbf{x}_1),\; \text{softmax}(\mathbf{x}_2)\bigr)$$

where $\mathbf{x}_1$ and $\mathbf{x}_2$ are the raw EGNN and GT logits. This encourages both branches to converge on consistent interface predictions rather than compensating for each other arbitrarily.

### Total Loss

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{focal} + \lambda_\text{gate} \cdot \mathcal{L}_\text{gate} + \lambda_\text{agree} \cdot \mathcal{L}_\text{agree}$$

where $\mathcal{L}_\text{focal}$ is Focal Loss (Lin et al. 2017) with per-fold negative/positive class weight $\alpha^+ = N^-/N^+$ computed from the training fold's label distribution. For Train\_335: global ratio 5.4056 (15.61% positive across 66,208 residues). Focusing parameter $\gamma$ is configurable (`--focal_gamma`, default 2.0; $\gamma=0$ reduces to weighted cross-entropy).

---

## File Directory

### Created Files
- **`generate_esm2_embeddings.py`**: Extracts per-residue embeddings from `facebook/esm2_t33_650M_UR50D` (fp16 on GPU, CPU float32 fallback on OOM); caches to `./Feature/esm2/{ID}.npy`.
- **`generate_rsa_features.py`**: Computes per-residue RSA via FreeSASA from PDB files; normalises by per-residue maximum ASA; caches to `./Feature/rsa/{ID}.npy`.
- **`fusion_module.py`**: `FeatureFusionModule` class — `none`, `concat`, `gated` modes.
- **`loss.py`**: `FocalLoss(alpha, gamma)` and `compute_pos_weight(labels_list)`.

### Modified Files
- **`data_generator.py`**: `ProDataset` loads ESM-2 embeddings (with length alignment) and RSA features (with 0.5 fallback). `graph_collate` batches both.
- **`final_model.py`**: `FinalModel` integrates the FFM and computes auxiliary losses via `compute_auxiliary_losses(rsa_target, lambda_gate, lambda_agree)`. EGNN now uses `residual=True` (stability fix). GT branch uses `transformer_residual=True` (corrected from paper, commit `d0c93b5`).
- **`train.py`**: Adds `--fusion_mode`, `--d_proj`, `--focal_gamma`, `--lambda_gate`, `--lambda_agree`, `--smoke_test`. Per-fold `pos_weight` computed from fold training subset. Memory cleanup between folds.
- **`test.py`**: Adds same CLI flags plus `--model_dir` (required). Validation-locked threshold protocol: threshold selected on validation fold, locked before evaluating test set. Gate values exported to CSV in `gated` mode (RSA loaded from `Feature/rsa/`, not from DSSP column).

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--fusion_mode` | `none` | `none` / `concat` / `gated` |
| `--d_proj` | `128` | Shared projection dimension $d$ |
| `--focal_gamma` | `2.0` | Focal Loss $\gamma$ (0 = weighted CE) |
| `--lambda_gate` | `0.1` | RSA gate supervision weight |
| `--lambda_agree` | `0.1` | Branch agreement regularisation weight |
| `--smoke_test` | off | 2-sample, 1-fold, 1-epoch verification run |
| `--model_dir` | — | (test.py only, required) path to checkpoint directory |

---

## Quantitative Results

All numbers below are 5-fold CV averages, validation-locked threshold protocol (no test-label leakage). Source: actual log files on disk.

### Test\_60 (13,144 residues, 15.6% positive)

| System | AUROC | AUPRC | MCC | F1 | Recall |
|---|---|---|---|---|---|
| GTE-PPIS paper (Wang et al. 2025)† | 0.873 | 0.611 | 0.500 | 0.582 | 0.611 |
| No-ESM2 baseline, γ=2.0 (Aug 25) | 0.8537 | 0.5518 | 0.4500 | 0.5319 | 0.5734 |
| Unsupervised gate, γ=2.0 (Aug 13) | 0.8236 | 0.5056 | 0.4125 | 0.5089 | 0.5395 |
| RSA-supervised gate, γ=0.0 (Sep 11) | 0.8116 | 0.4863 | 0.3874 | 0.4925 | 0.5708 |

### Test\_315-28 (60,376 residues)

| System | AUROC | AUPRC | MCC | F1 |
|---|---|---|---|---|
| GTE-PPIS paper† | — | 0.598 | 0.511 | — |
| No-ESM2 baseline (Aug 25) | 0.8585 | 0.5182 | 0.4354 | 0.5189 |
| Unsupervised gate (Aug 13) | 0.8117 | 0.4381 | 0.3601 | 0.4597 |
| RSA-supervised gate (Sep 11) | 0.8029 | 0.4351 | 0.3497 | 0.4492 |

### UBtest\_31-6 (5,917 residues, unbound conformations)‡

| System | AUROC | AUPRC | MCC | F1 |
|---|---|---|---|---|
| GTE-PPIS paper (UBtest\_25)† | — | 0.343 | 0.320 | — |
| No-ESM2 baseline (Aug 25) | 0.7611 | 0.3115 | 0.2713 | 0.3542 |
| Unsupervised gate (Aug 13) | 0.8191 | 0.4182 | 0.3706 | 0.4454 |
| RSA-supervised gate (Sep 11) | 0.8159 | 0.4248 | 0.3694 | 0.4495 |

† Paper numbers from Wang et al. (2025), *Bioinformatics*, PMC12199915, Table 2–3. Paper threshold protocol unconfirmed (may include test-label tuning). ‡ Our UBtest\_31-6 ≠ paper's UBtest\_25; not directly comparable.

**Key finding**: ESM-2 gated fusion consistently improves UBtest (unbound) performance vs. the no-ESM2 backbone (+0.107 AUPRC, +0.099 MCC) at the cost of lower Test\_60 performance (−0.049 AUPRC, −0.038 MCC). RSA supervision (Sep 11) does not further improve over the unsupervised gate on any primary metric, but comparison is confounded by three simultaneous changes (see ablation report).

---

## Experiment Status

| Mode | Status | Log directory |
|---|---|---|
| `none` (backbone baseline) | ✅ Complete, full 5-fold + evaluation | `Log/fusion_none_d128_2026-08-25-11-28-09/` |
| `gated` (unsupervised gate) | ✅ Complete, full 5-fold + evaluation | `Log/fusion_gated_d128_2026-08-13-11-53-23/` |
| `gated` (RSA-supervised) | ✅ Complete, full 5-fold + evaluation | `Log/fusion_gated_d128_2026-09-11-15-22-42/` |
| `concat` | 🔲 Smoke test only; full training not yet run | — |

---

## Open Problems / Next Steps

1. **Confounder isolation**: Sep 11 run differs from Aug 13 on three axes simultaneously (RSA supervision, `transformer_residual`, `focal_gamma`). A matched run with `--lambda_gate 0.0 --lambda_agree 0.0` under the Sep 11 architecture is required before attributing any delta to RSA supervision.
2. **Multi-seed testing**: All runs use `SEED=2024`. Observed deltas (ΔAUPRC ≤ 0.019) are within expected single-seed variance. $N≥3$ seeds needed for significance testing.
3. **λ sweep**: `--lambda_gate` and `--lambda_agree` both default to 0.1 untested. Sweep over {0.01, 0.05, 0.1, 0.2} needed.
4. **Gate inspection**: Load checkpoints and compute correlation between predicted gate value $\hat{g}_i$ and RSA to verify whether the auxiliary loss is directing gate behaviour or being overwhelmed by the classification gradient.
5. **B-factor gating**: Replace RSA target with normalised crystallographic B-factor as gate supervision signal. Novelty search (Sep 11) confirmed no prior work uses B-factor as a reliability gate for PLM–structure fusion in PPIS.
6. **`concat` full run**: No complete training run exists for the concat fusion mode.




