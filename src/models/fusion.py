import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GLU, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        Y = self.W(X) * self.sigmoid(self.V(X))
        return Y


class DrugFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.gated_ch = GLU(input_dim//3, input_dim//3)
        self.gated_gr = GLU(input_dim//3, input_dim//3)
        self.gated_gm = GLU(input_dim//3, input_dim//3)
        self.tanh = nn.Tanh()
        
    def forward(self, chembert_x, grover_x, graphmvp_x):
        
        ch = self.gated_ch(chembert_x)
        gr = self.gated_gr(grover_x)
        gm = self.gated_gm(graphmvp_x)
        d = self.tanh(torch.cat([ch, gr, gm], dim=1))

        return d


class DrugProteinFusion(nn.Module):
    def __init__(self, drug_dim, protein_dim):
        super().__init__()
        self.gated_d = GLU(drug_dim, drug_dim)
        self.gated_p = GLU(protein_dim, protein_dim)
        self.tanh = nn.Tanh()

    def forward(self, d, p):
        d = self.gated_d(d)
        p = self.gated_p(p)
        f = self.tanh(torch.cat([d, p], dim=1))
        return f

