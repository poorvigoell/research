"""
Baseline Models Suite:
- Tabular / Feature MLP
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- GraphSAGE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLPBaseline(nn.Module):
    """Tabular Feature Baseline (Ignores Graph Topology)."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.3):
        super(MLPBaseline, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc3(h)


class SimpleGCNLayer(nn.Module):
    """Vectorized GCN Layer using pure PyTorch operations."""

    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Add self-loops for standard GCN normalization: \tilde{A} = A + I_N
        loop = torch.arange(num_nodes, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        # Degree normalization: D^{-1/2} \tilde{A} D^{-1/2}
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        # Message passing
        h_trans = self.linear(x)
        msg = h_trans[src] * norm.unsqueeze(-1)

        out = torch.zeros((num_nodes, self.linear.out_features), device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return out + self.bias


class GCNBaseline(nn.Module):
    """Standard GCN Baseline with configurable layer depth."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, num_layers: int = 2, dropout: float = 0.3):
        super(GCNBaseline, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SimpleGCNLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(SimpleGCNLayer(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


class SimpleGATLayer(nn.Module):
    """Vectorized GAT Layer using pure PyTorch operations."""

    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleGATLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.attn_src = nn.Parameter(torch.Tensor(1, out_channels))
        self.attn_dst = nn.Parameter(torch.Tensor(1, out_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Add self-loops for attention to self
        loop = torch.arange(num_nodes, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        h = self.linear(x)
        alpha = (h[src] * self.attn_src).sum(dim=-1) + (h[dst] * self.attn_dst).sum(dim=-1)
        alpha = self.leaky_relu(alpha)

        # Per-destination-node softmax (numerically stable)
        alpha_max = torch.full((num_nodes,), -1e9, device=x.device)
        alpha_max.scatter_reduce_(0, dst, alpha, reduce='amax', include_self=False)
        alpha_exp = torch.exp(alpha - alpha_max[dst])
        alpha_sum = torch.zeros(num_nodes, device=x.device)
        alpha_sum.scatter_add_(0, dst, alpha_exp)
        alpha = alpha_exp / alpha_sum[dst].clamp(min=1e-8)

        msg = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros((num_nodes, self.linear.out_features), device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return out


class GATBaseline(nn.Module):
    """Standard GAT Baseline with configurable layer depth."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, num_layers: int = 2, dropout: float = 0.3):
        super(GATBaseline, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SimpleGATLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(SimpleGATLayer(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)
