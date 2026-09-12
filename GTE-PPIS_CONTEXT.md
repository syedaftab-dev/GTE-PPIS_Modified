# GTE-PPIS Modified — Antigravity Project Context

## 1. Project Overview

I am working on a research project based on GTE-PPIS (Graph-based Transformer/EGNN approach for Protein-Protein Interaction Site prediction).

The project is a modified version of GTE-PPIS. The main research idea is a **Feature Fusion Module (FFM)** that combines the model's raw node/residue features with evolutionary/sequence-derived information before the EGNN + GraphTransformer processing.

The goal is to evaluate whether this additional feature fusion improves PPIS prediction compared with the original GTE-PPIS model.

---

## 2. Current Architecture / Research Setup

Important components discussed so far:

- Base architecture: GTE-PPIS
- Added component: Feature Fusion Module (FFM)
- Evolutionary/sequence feature streams:
  - ESM-2 650M embeddings
  - PSSM/HMM features
- Fusion modes currently considered:
  - `none`
  - `concat`
  - `gated`
  - `cross_attn`
- Projection dimension:
  - `d_proj = 128`
- The evolutionary features are intended to complement the structural/raw node features rather than simply replace them.

Do NOT assume that every proposed experiment has already been run. Check the actual source code and logs before claiming that an experiment or result exists.

---

## 3. Dataset Information

Datasets currently relevant to evaluation:

- `Train_335.pkl`
- `Test_60.pkl`
- `Test_315-28.pkl`
- `UBtest_31-6.pkl`

Protein `2j3rA` was removed/excluded.

Cross-validation seed:

- `2024`

---

## 4. RSA Research Direction

A major current research question is how **RSA (Relative Solvent Accessibility)** can help the PPIS prediction task.

The expected research reasoning is:

- RSA provides information about how exposed a residue is to the solvent.
- Interface residues are often associated with surface accessibility.
- Therefore RSA can provide complementary structural/biophysical information to the model.
- The expected benefit is better discrimination between interface and non-interface residues, especially when geometric or sequence features alone are ambiguous.

However, do not claim that RSA definitely improves performance until experiments demonstrate it.

When discussing RSA, distinguish clearly between:
1. biological motivation,
2. expected mechanism,
3. experimentally observed result.

---

## 5. Evaluation Metrics

The main evaluation metrics being considered are:

- AUROC
- AUPRC
- MCC
- Accuracy
- Precision
- Recall
- F1

For PPIS, the dataset can be imbalanced, so accuracy alone is not sufficient.

Important interpretation:

### AUROC
Measures ranking/discrimination between positive and negative residues across classification thresholds.

### AUPRC
Especially important when the positive class is relatively rare because it focuses on precision-recall behaviour and is more informative about positive-class detection.

### MCC
Useful as a balanced single-score metric using all four confusion-matrix components (TP, TN, FP, FN). It is useful for evaluating the final classification quality under class imbalance.

### Accuracy
Can be misleading with imbalanced data because a model can obtain high accuracy while performing poorly on the minority/interface class.

For the paper, do not treat one metric as universally "best". A reasonable emphasis is:
1. AUPRC — particularly important for minority-positive PPIS detection.
2. MCC — strong threshold-based overall classification metric under imbalance.
3. AUROC — useful threshold-independent ranking/discrimination metric.
4. F1 / precision / recall — useful for positive-class behaviour.
5. Accuracy — supplementary rather than the primary metric.

Use the actual experimental results to determine the final conclusions.

---

## 6. BCE vs Focal Loss

Another possible experiment is replacing Binary Cross Entropy (BCE) with focal loss.

Research motivation:

- BCE treats training examples more uniformly.
- Focal loss reduces the relative contribution of easy examples and focuses training more strongly on difficult/misclassified examples.
- This can potentially help with class imbalance and hard interface/non-interface residue discrimination.

Important:
- A loss value greater than 1 is NOT automatically a problem for focal loss.
- Loss scales depend on the loss definition and implementation.
- Do not compare raw BCE and focal-loss numerical values as if they were directly equivalent.

Focal loss should be presented as an experimental modification, not as guaranteed improvement.

---

## 7. Known Experimental / Debugging Context

There have been training runs using gated fusion.

One earlier run:
- `fusion_gated_d128_2026-07-23-15-16-42`

This run showed cases where:
- MCC/F1/precision/recall were 0 for some validation epochs
- while AUC/AUPRC were non-zero

This can happen when the model ranks examples reasonably but the selected classification threshold produces poor positive predictions.

A later run:
- `fusion_gated_d128_2026-07-30-12-18-16`

A validation-threshold leakage issue in `test.py` was addressed:
- thresholds should be determined using the validation fold
- test labels must NOT be used to choose the test threshold

This distinction is important for fair evaluation.

Do not assume a particular epoch is the best model unless the actual logs/checkpoints confirm it.

---

## 8. Comparison With Original GTE-PPIS

The research should compare the modified model against the original GTE-PPIS fairly.

Relevant test sets:
- Test-60 / `Test_60.pkl`
- Test-315-28 / `Test_315-28.pkl`
- UB test / `UBtest_31-6.pkl`

When comparing:
- use the same dataset definitions where applicable
- use consistent evaluation procedures
- avoid comparing metrics produced using different threshold-selection procedures
- distinguish validation performance from independent test performance

Never invent original GTE-PPIS metrics or modified-model metrics. Retrieve them from the actual paper, source files, or experiment logs when needed.

---

## 9. Research Novelty

The intended novelty is not simply "using ESM-2" or "using a GNN".

The focus is on integrating additional sequence/evolutionary information into the existing GTE-PPIS architecture through a dedicated feature-fusion mechanism and investigating different fusion strategies.

Potential research questions include:

- Does feature fusion improve over the original GTE-PPIS representation?
- Which fusion strategy works best?
- Does gated fusion selectively weight useful evolutionary information?
- Does cross-attention provide better interaction between feature streams?
- Does RSA add complementary structural information?
- Does focal loss improve minority/hard-example learning?
- Are improvements consistent across Test-60, Test-315-28 and UBtest-31-6?

These are research hypotheses and must be validated experimentally.

---

## 10. Project Files / Working Rules

The important source files may include:

- `train.py`
- `test.py`
- model/architecture files
- feature extraction/preprocessing scripts
- experiment logs
- saved checkpoints
- feature files under `Feature/`
- ESM embeddings under `Feature/esm2`
- PDB-related data under `PDB/`

Current environment context:
- Ubuntu 24.04.x
- Python 3.10.x
- NVIDIA GPU environment
- project path previously used: `~/PRANAV/GTE-PPIS`
- modified GitHub repository target: `PranavNagaraji/GTE-PPIS_Modified`

Do not modify code blindly.

Before making architectural or experimental changes:
1. Inspect the relevant existing files.
2. Explain what the current implementation does.
3. Identify the smallest necessary change.
4. Make changes without unnecessarily renaming variables or restructuring unrelated code.
5. Preserve the existing experimental setup unless the requested change requires otherwise.

---

## 11. User's Coding Preferences

When helping with this project:

- Build on the existing code.
- Make minimal changes.
- Do not rename variables unnecessarily.
- Do not rewrite entire files unless explicitly requested.
- Do not add unnecessary templates/comments.
- For debugging, identify the exact issue and give the next step.
- Keep explanations short unless I explicitly ask for detailed/elaborate theory.
- Never invent results.
- If a conclusion depends on a file/log, inspect that file first.
- Clearly separate:
  - observed result,
  - expected behaviour,
  - hypothesis/speculation.

---

## 12. Critical Instruction for Antigravity

This context is background knowledge only.

**The actual repository files and experiment logs are the source of truth.**

Before answering questions about:
- architecture,
- metrics,
- training behaviour,
- test results,
- thresholds,
- RSA implementation,
- focal loss implementation,
- fusion implementation,

inspect the actual project files/logs.

If the repository contradicts this context, trust the repository and explain the discrepancy.

Do not fabricate missing metrics, experiments, code, or conclusions.
