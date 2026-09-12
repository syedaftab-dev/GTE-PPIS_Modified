import torch
import torch.nn as nn

class FeatureFusionModule(nn.Module):
    def __init__(self, fusion_mode='gated', d_proj=128):
        super(FeatureFusionModule, self).__init__()
        self.fusion_mode = fusion_mode
        self.d_proj = d_proj
        
        if fusion_mode == 'none':
            return
            
        # 1. Projections
        self.classical_proj = nn.Linear(40, d_proj)
        self.plm_proj = nn.Linear(1280, d_proj)
        
        self.classical_ln = nn.LayerNorm(d_proj)
        self.plm_ln = nn.LayerNorm(d_proj)
        
        # 2. Fusion variant components
        if fusion_mode == 'gated':
            self.gate_linear = nn.Linear(2 * d_proj, 1)
        elif fusion_mode == 'concat':
            pass
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode!r}. Valid modes: 'none', 'concat', 'gated'")
            
        # Initialize projections with standard gain
        nn.init.xavier_uniform_(self.classical_proj.weight, gain=1.0)
        nn.init.zeros_(self.classical_proj.bias)
        nn.init.xavier_uniform_(self.plm_proj.weight, gain=1.0)
        nn.init.zeros_(self.plm_proj.bias)
        
        if fusion_mode == 'gated':
            nn.init.xavier_uniform_(self.gate_linear.weight, gain=1.0)
            nn.init.zeros_(self.gate_linear.bias)
            
    def forward(self, classical_i, plm_i):
        if torch.isnan(classical_i).any() or torch.isnan(plm_i).any():
            print(f"[DEBUG] NaN detected in raw FeatureFusionModule inputs! classical: {torch.isnan(classical_i).any()}, plm: {torch.isnan(plm_i).any()}")
            
        # Project both to shared dim d
        c_proj = self.classical_ln(self.classical_proj(classical_i))  # (N, d)
        p_proj = self.plm_ln(self.plm_proj(plm_i))                  # (N, d)
        
        if torch.isnan(c_proj).any() or torch.isnan(p_proj).any():
            print(f"[DEBUG] NaN detected in projected streams! c_proj: {torch.isnan(c_proj).any()}, p_proj: {torch.isnan(p_proj).any()}")
            
        gate_val = None
        if self.fusion_mode == 'concat':
            fused_i = torch.cat([c_proj, p_proj], dim=-1)  # (N, 2d)
        elif self.fusion_mode == 'gated':
            # g_i = sigmoid(Linear(concat(classical_proj, plm_proj)))
            concat_proj = torch.cat([c_proj, p_proj], dim=-1)  # (N, 2d)
            gate_val = torch.sigmoid(self.gate_linear(concat_proj))  # (N, 1)
            if torch.isnan(gate_val).any():
                print("[DEBUG] NaN detected in gate_val!")
            fused_i = gate_val * c_proj + (1.0 - gate_val) * p_proj  # (N, d)
            
        if torch.isnan(fused_i).any():
            print("[DEBUG] NaN detected in fused_i output!")
            
        return fused_i, gate_val
