import torch
import torch.nn as nn

from data_generator import *
from EGNN_model import *
from GraphTransformer_Block import *

from fusion_module import FeatureFusionModule
from loss import FocalLoss

# Default neg/pos ratio computed from the full Train_335 dataset
# (55872 negative / 10336 positive = 5.4056). Used when no per-fold
# pos_weight is supplied (e.g. during testing / checkpoint loading).
_DEFAULT_POS_WEIGHT = 5.4056


class FinalModel(nn.Module):
    def __init__(self, input_size, hidden_size, fliter_size, output_size, dropout_rate,
                 n_layers, fusion_mode='none', d_proj=128,
                 pos_weight=_DEFAULT_POS_WEIGHT, focal_gamma=2.0):
        """
        Args:
            pos_weight (float): neg/pos ratio for the current training fold.
                Used to build the per-class alpha weights for FocalLoss.
                Defaults to the full-dataset ratio (5.4056).
            focal_gamma (float): Focusing parameter for FocalLoss. gamma=0 reduces
                to weighted CrossEntropyLoss. Default: 2.0.
        """
        super(FinalModel, self).__init__()
        self.fusion_mode = fusion_mode
        self.d_proj = d_proj

        # Calculate actual input dimension for EGNN and GT branches
        if fusion_mode == 'none':
            self.actual_input_size = input_size  # 61
            self.fusion_module = None
        elif fusion_mode == 'concat':
            self.actual_input_size = 2 * d_proj + 21
            self.fusion_module = FeatureFusionModule(fusion_mode=fusion_mode, d_proj=d_proj)
        elif fusion_mode in ['gated', 'cross_attn']:
            self.actual_input_size = d_proj + 21
            self.fusion_module = FeatureFusionModule(fusion_mode=fusion_mode, d_proj=d_proj)
        else:
            raise ValueError(f"Unknown fusion mode {fusion_mode}")

        # residual=True: enables skip connections inside every E_GCL block.
        # This is the primary stability fix — prevents activation scale explosion
        # across the 10-layer EGNN when input feature distributions shift due to
        # the FFM gate. tanh=True bounds coordinate updates (already set).
        self.Egnn = EGNN(in_node_nf=self.actual_input_size, hidden_nf=hidden_size,
                         out_node_nf=output_size, in_edge_nf=2, n_layers=10,
                         attention=True, residual=True, tanh=True)

        self.GT = GraghTransformer(in_channels=self.actual_input_size, edge_features=2,
                                   dropout_rate=dropout_rate, num_layers=4,
                                   transformer_residual=False)

        # --- Loss ---
        # class_weights: [neg_weight, pos_weight] = [1.0, neg/pos ratio]
        # Upweights the minority positive class to counteract the ~85/15 imbalance.
        class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
        if torch.cuda.is_available():
            class_weights = class_weights.cuda()

        # FocalLoss with alpha=class_weights and gamma=focal_gamma.
        # gamma=0 → standard weighted CrossEntropyLoss (use for ablation).
        # gamma=2 → standard Focal Loss (Lin et al. 2017, RetinaNet).
        self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6)

    def forward(self, node_features, xyz_feats, edges, edge_att, edge_feat, adj,
                plm_features=None):
        self.last_gate_val = None

        if self.fusion_module is not None and plm_features is not None:
            classical_i = node_features[:, 14:54]
            dssp_i = node_features[:, 0:14]
            af_i = node_features[:, 54:61]

            fused_i, gate_val = self.fusion_module(classical_i, plm_features)
            self.last_gate_val = gate_val

            node_features = torch.cat([fused_i, dssp_i, af_i], dim=-1)

        x1 = self.Egnn(node_features, xyz_feats, edges, edge_feat)
        x2 = self.GT(node_features, edge_feat, edges)
        x = (x1 + x2) / 2
        return x
