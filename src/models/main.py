import torch.nn as nn
import math
from src.models.fusion import DrugFusion
from src.models.fusion import DrugProteinFusion

class DTAnet(nn.Module):
    def __init__(self, ch_dim, gr_dim, gm_dim, prot_dim, map_dim, hidden_dims, dropout_rate):
        super().__init__()
        
        # Chembert Mapping
        self.ch_map = nn.Linear(ch_dim, map_dim)
        # Grover Mapping
        grover_layers = []
        current_dim = gr_dim
        while current_dim > map_dim:
            next_dim = max(map_dim, math.ceil(current_dim * 2 / 3))
            grover_layers.append(nn.Linear(current_dim, next_dim))
            grover_layers.append(nn.ReLU())
            current_dim = next_dim
        if current_dim != map_dim:
            grover_layers.append(nn.Linear(current_dim, map_dim))
            grover_layers.append(nn.ReLU())
        self.gr_map = nn.Sequential(*grover_layers)
        # GraphMVP Mapping
        self.gm_map = nn.Linear(gm_dim, map_dim)
        # Drug-Drug Fusion 
        self.dd_fusion = DrugFusion(map_dim * 3, map_dim)
        self.dd_fusion_dim = 3072
        # Drug-Protein Fusion
        self.dp_fusion = DrugProteinFusion(self.dd_fusion_dim, prot_dim)
        self.input_dim = self.dd_fusion_dim + prot_dim

        # MLP
        mlp_layers = []
        prev_dim = self.input_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, h))
            mlp_layers.append(nn.BatchNorm1d(h))
            mlp_layers.append(nn.Dropout(dropout_rate))
            mlp_layers.append(nn.ReLU())
            prev_dim = h
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, ch_x, gr_x, gm_x, prot_x):
        
        ch_proj = self.ch_map(ch_x)
        gr_proj = self.gr_map(gr_x)
        gm_proj = self.gm_map(gm_x)
        
        dd_fusion = self.dd_fusion(ch_proj, gr_proj, gm_proj)
        dp_fusion = self.dp_fusion(dd_fusion, prot_x)
        return self.mlp(dp_fusion)
