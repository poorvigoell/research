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
        self.classifier = self.fc3
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


class SimpleSAGELayer(nn.Module):
    """Vectorized GraphSAGE Layer using pure PyTorch operations."""

    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean"):
        super(SimpleSAGELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr.lower()
        if self.aggr not in ("mean", "max"):
            raise ValueError(f"Unsupported aggregation: {aggr}. Expected 'mean' or 'max'.")

        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        if self.lin_self.bias is not None:
            nn.init.zeros_(self.lin_self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Add self-loops for root node features
        loop = torch.arange(num_nodes, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        # Transform node features for message passing
        h_trans = self.lin_neigh(x)
        msg = h_trans[src]

        if self.aggr == "mean":
            # Destination node degree for mean aggregation
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))

            # Aggregate neighbor messages using scatter_add_
            agg = torch.zeros((num_nodes, self.out_channels), device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
        elif self.aggr == "max":
            # Max aggregation across incoming messages
            agg = torch.full((num_nodes, self.out_channels), -1e9, device=x.device)
            agg.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(msg), msg, reduce='amax', include_self=False)
            agg = torch.where(agg == -1e9, torch.zeros_like(agg), agg)

        # Combine root self-loop feature representation and neighborhood aggregation
        out = self.lin_self(x) + agg
        return out


class GraphSAGEBaseline(nn.Module):
    """Standard GraphSAGE Baseline with configurable layer depth and aggregation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggr: str = "mean",
    ):
        super(GraphSAGEBaseline, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SimpleSAGELayer(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 1):
            self.layers.append(SimpleSAGELayer(hidden_channels, hidden_channels, aggr=aggr))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.relu(h)
            # L2-normalize embeddings (standard GraphSAGE practice)
            h = F.normalize(h, p=2, dim=-1)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)
