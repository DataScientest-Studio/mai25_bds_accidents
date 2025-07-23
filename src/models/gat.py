
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class GATResNet(nn.Module):
    def __init__(self, in_dim, hidden=32, out_dim=2, heads=8, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.gats = nn.ModuleList([
            GATConv(in_dim, hidden, heads=heads, dropout=dropout),
            GATConv(hidden * heads, hidden, heads=heads, dropout=dropout),
            GATConv(hidden * heads, hidden, heads=heads, dropout=dropout),
        ])

        self.bns = nn.ModuleList([
            BatchNorm(hidden * heads),
            BatchNorm(hidden * heads),
            BatchNorm(hidden * heads),
        ])

        self.res0 = nn.Linear(in_dim, hidden * heads, bias=False)
        self.final = GATConv(hidden * heads, out_dim, heads=1, concat=False)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        h = self.gats[0](x, ei)
        h = F.elu(self.bns[0](h))
        x = self.res0(x) + h
        x = F.dropout(x, p=self.dropout, training=self.training)

        for gat, bn in zip(self.gats[1:], self.bns[1:]):
            h = F.elu(bn(gat(x, ei)))
            x = x + h
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.final(x, ei)