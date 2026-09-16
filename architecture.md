# GTE-PPIS Model Architecture

This diagram follows the active `FinalModel.forward()` path and the training losses used by `train.py`.

```mermaid
flowchart TB
    %% Input preparation
    subgraph S1["1. Input preparation"]
        SEQ(["Protein sequence<br/>L residues"])
        CLASSICAL["Classical residue features<br/>61D per residue<br/>DSSP 14D + PSSM 20D + HMM 20D + resAF 7D"]
        ESM["ESM-2 residue embedding<br/>1280D per residue"]
        XYZ["Residue coordinates<br/>3D pseudo-position per residue"]
        GRAPH["Radius graph construction<br/>connect residues within 14 distance units"]
        EDGES["Graph connectivity<br/>edge index: source to target"]
        EDGEFEAT["Geometric edge features<br/>2D: normalized distance and cosine similarity"]
        RSA["RSA values<br/>scalar per residue<br/>training target only"]
        SEQ --> CLASSICAL
        SEQ --> ESM
        SEQ --> XYZ
        XYZ --> GRAPH
        GRAPH --> EDGES
        EDGES --> EDGEFEAT
    end

    %% Optional feature fusion
    subgraph S2["2. Feature fusion per residue: FeatureFusionModule"]
        CPROJ["Classical branch projection<br/>Linear 40D to d_proj<br/>LayerNorm"]
        PPROJ["ESM-2 branch projection<br/>Linear 1280D to d_proj<br/>LayerNorm"]
        GATE["Learned scalar gate<br/>g equals sigmoid of Linear(2 times d_proj to 1)"]
        GATED["Gated fusion output<br/>g times c_proj plus (1 minus g) times p_proj<br/>d_proj"]
        CONCAT["Concatenation output<br/>c_proj concatenated with p_proj<br/>2 times d_proj"]
        CLASSICAL --> CPROJ
        ESM --> PPROJ
        CPROJ --> GATE
        PPROJ --> GATE
        CPROJ --> GATED
        PPROJ --> GATED
        CPROJ --> CONCAT
        PPROJ --> CONCAT
        GATE --> GATED
    end

    NONE["Mode: none<br/>Use all classical features<br/>61D"]
    GATED_INPUT["Mode: gated<br/>fused d_proj plus DSSP 14D plus resAF 7D<br/>default: 128 plus 14 plus 7 equals 149D"]
    CONCAT_INPUT["Mode: concat<br/>fused 2 times d_proj plus DSSP 14D plus resAF 7D<br/>default: 256 plus 14 plus 7 equals 277D"]
    CLASSICAL --> NONE
    GATED --> GATED_INPUT
    CONCAT --> CONCAT_INPUT

    NODE["Final node representation<br/>61D, 149D, or 277D<br/>one vector per residue"]
    NONE --> NODE
    GATED_INPUT --> NODE
    CONCAT_INPUT --> NODE

    %% Parallel graph backbones
    subgraph S3["3. Parallel graph backbones"]
        EGNN["EGNN<br/>Equivariant Graph Neural Network<br/>Linear input to 256D hidden<br/>10 E_GCL message-passing layers<br/>residual, attention, bounded coordinate updates<br/>Linear hidden to 2 logits"]
        GT["Graph Transformer<br/>node encoder to 128D hidden<br/>4 geometric-attention layers<br/>4 heads, residual connections, dropout 0.1<br/>Linear hidden to 2 logits"]
        NODE --> EGNN
        NODE --> GT
        XYZ --> EGNN
        EDGES --> EGNN
        EDGEFEAT --> EGNN
        EDGEFEAT --> GT
        EDGES --> GT
    end

    %% Prediction head
    EGNN --> AVG["Logit ensemble<br/>element-wise average<br/>logits equal (EGNN plus GT) divided by 2"]
    GT --> AVG
    AVG --> PRED(["Per-residue PPIS prediction<br/>2 logits: non-interface or interface"])

    %% Training-only objective
    subgraph S4["4. Training objective"]
        LABELS["Ground-truth interface labels<br/>0 = non-interface, 1 = interface"]
        FOCAL["Class-weighted focal loss<br/>applied to averaged logits<br/>gamma = 2 by default"]
        GATELOSS["RSA gate supervision<br/>MSE of gate and (1 minus RSA)<br/>weight lambda_gate = 0.1"]
        AGREEMENT["Branch agreement regularization<br/>MSE of EGNN and GT softmax probabilities<br/>weight lambda_agree = 0.1"]
        TOTAL(["Total training loss<br/>focal plus gate plus agreement"])
        PRED --> FOCAL
        LABELS --> FOCAL
        RSA --> GATELOSS
        GATE --> GATELOSS
        EGNN --> AGREEMENT
        GT --> AGREEMENT
        FOCAL --> TOTAL
        GATELOSS --> TOTAL
        AGREEMENT --> TOTAL
    end

    classDef input fill:#e8f1f8,stroke:#28627d,color:#102a43,stroke-width:1px
    classDef fusion fill:#fff2cc,stroke:#b7791f,color:#4a2c00,stroke-width:1px
    classDef backbone fill:#e9f5ec,stroke:#2f855a,color:#163b25,stroke-width:1px
    classDef output fill:#fce8e6,stroke:#c53030,color:#4a1010,stroke-width:2px
    classDef objective fill:#f0eafa,stroke:#6b46c1,color:#2d1b4e,stroke-width:1px
    class SEQ,CLASSICAL,ESM,XYZ,GRAPH,EDGES,EDGEFEAT,RSA,LABELS input
    class CPROJ,PPROJ,GATE,GATED,CONCAT,NONE,GATED_INPUT,CONCAT_INPUT,NODE fusion
    class EGNN,GT backbone
    class AVG,PRED output
    class FOCAL,GATELOSS,AGREEMENT,TOTAL objective
```

### Component definitions

| Component | Definition | Implementation |
|---|---|---|
| DSSP | Per-residue secondary-structure and solvent-related descriptors. | `data_generator.py::get_node_features()` |
| PSSM | Position-specific scoring matrix encoding evolutionary conservation. | `data_generator.py::get_node_features()` |
| HMM | Profile-HMM residue features encoding sequence-family information. | `data_generator.py::get_node_features()` |
| resAF | Residue atom-feature descriptors. | `data_generator.py::get_node_features()` |
| ESM-2 | Protein language-model embedding with 1280 values per residue. | `data_generator.py::ProDataset.__getitem__()` |
| Radius graph | Structural residue graph containing edges for residue pairs within the configured cutoff. | `data_generator.py::cal_edges()` |
| E-GCL | EGNN layer that updates edge messages, node states, and coordinates. | `EGNN_model.py::E_GCL` |
| EGNN | Equivariant graph branch using 10 E-GCL layers and 2-class output logits. | `EGNN_model.py::EGNN` |
| Graph Transformer | Multi-head geometric attention branch using node and edge representations. | `GraphTransformer_Block.py::GraghTransformer` |
| FeatureFusionModule | Projects classical and ESM-2 features into a shared space, then gates or concatenates them. | `fusion_module.py::FeatureFusionModule` |
| RSA supervision | Uses relative solvent accessibility to target the gate with `1 - RSA`. | `final_model.py::compute_auxiliary_losses()` |
| Branch agreement | Penalizes disagreement between the EGNN and Graph Transformer probabilities. | `final_model.py::compute_auxiliary_losses()` |

## Runtime variants

- `--fusion_mode none`: classical 61D node features; ESM-2 is bypassed.
- `--fusion_mode gated`: learned scalar gate mixes projected classical and ESM-2 features. This is the full proposed path and the default architecture shown above when fusion is enabled.
- `--fusion_mode concat`: projected classical and ESM-2 features are concatenated instead of gated.

The dataset also prepares an adjacency matrix and `edge_att`, but the current `FinalModel.forward()` uses `node_features`, `xyz_feats`, `edges`, and `edge_feat` for its two active branches. RSA values are used as auxiliary gate targets, not as prediction inputs.
