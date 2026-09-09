"""
ACO-GNN: Pheromone-Guided Graph Neural Network Architecture.
Integrates bio-inspired stigmergic pheromone trails as an inductive structural prior
directly into multi-hop neighborhood aggregation to neutralize graph camouflage
and zero in on collusive financial fraud rings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class PheromoneGuidedConv(nn.Module):
    """
    Custom Relation-Aware Pheromone Message Passing Layer.
    
    Blends:
    - alpha_ij: Feature attention score (GAT mechanism)
    - hat_tau_ij: Persistent stigmergic pheromone weight from ACO engine
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int = 1,
        lambda_blend: float = 0.5,
        fixed_lambda: Optional[float] = None,
        dropout: float = 0.2,
    ):
        super(PheromoneGuidedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = max(1, num_relations)
        self.fixed_lambda = fixed_lambda

        # Learnable blending coefficient: sigmoid(lambda_param) ∈ [0, 1]
        # Clamped to prevent math domain error when lambda_blend is 0.0 or 1.0
        clamped_lambda = max(1e-4, min(1.0 - 1e-4, float(lambda_blend)))
        init_val = math.log(clamped_lambda / (1.0 - clamped_lambda))
        self.lambda_param = nn.Parameter(torch.tensor(init_val))
        self.dropout = nn.Dropout(dropout)

        # Self-connection root weight
        self.w_root = nn.Linear(in_channels, out_channels, bias=True)

        # Per-relation transformation weights
        self.w_rel = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(self.num_relations)
        ])

        # Learnable attention parameters (GAT-style)
        self.attn_src = nn.Parameter(torch.Tensor(1, out_channels))
        self.attn_dst = nn.Parameter(torch.Tensor(1, out_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_root.weight)
        for w in self.w_rel:
            nn.init.xavier_uniform_(w.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        hat_tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            edge_type: Relation type IDs [E]
            hat_tau: Normalized edge pheromone weights from ACO engine [E]
            
        Returns:
            Updated node representations [N, out_channels]
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Root self-loop transformation
        out = self.w_root(x)

        if edge_index.size(1) == 0:
            return out

        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        if hat_tau is None:
            # Uniform fallback if ACO is not active
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
            hat_tau = 1.0 / deg[dst].clamp(min=1.0)

        # Aggregate neighbor messages across relation types
        for r in range(self.num_relations):
            rel_mask = (edge_type == r)
            if not rel_mask.any():
                continue

            r_src = src[rel_mask]
            r_dst = dst[rel_mask]
            r_tau = hat_tau[rel_mask]

            # Transform neighbor representations for relation r
            h_transformed = self.w_rel[r](x)
            h_neighbors = h_transformed[r_src]

            # Compute feature self-attention alpha_ij
            alpha_logits = (h_transformed[r_src] * self.attn_src).sum(dim=-1) + \
                           (h_transformed[r_dst] * self.attn_dst).sum(dim=-1)
            alpha_logits = self.leaky_relu(alpha_logits)

            # Per-destination-node softmax (numerically stable scatter softmax)
            alpha_max = torch.full((num_nodes,), -1e9, device=x.device)
            alpha_max.scatter_reduce_(0, r_dst, alpha_logits, reduce='amax', include_self=False)
            alpha_exp = torch.exp(alpha_logits - alpha_max[r_dst])
            alpha_sum = torch.zeros(num_nodes, device=x.device)
            alpha_sum.scatter_add_(0, r_dst, alpha_exp)
            alpha_weights = alpha_exp / alpha_sum[r_dst].clamp(min=1e-8)

            # Dual-Attention Blending:
            # Combined Weight = λ * feature_attention + (1 - λ) * stigmergic_pheromone
            if self.fixed_lambda is not None:
                lam = float(self.fixed_lambda)
            else:
                lam = torch.sigmoid(self.lambda_param)
            combined_weights = lam * alpha_weights + (1.0 - lam) * r_tau
            combined_weights = self.dropout(combined_weights)

            # Scale messages
            msg = h_neighbors * combined_weights.unsqueeze(-1)

            # Scatter add to destination nodes
            agg = torch.zeros((num_nodes, self.out_channels), device=x.device)
            agg.scatter_add_(0, r_dst.unsqueeze(-1).expand_as(msg), msg)

            out = out + agg

        return out


class ACORingGNN(nn.Module):
    """
    Complete Multi-Hop ACO-GNN Model.
    Stacks PheromoneGuidedConv layers with residual connections and classification MLP.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        num_relations: int = 1,
        lambda_blend: float = 0.5,
        fixed_lambda: Optional[float] = None,
        dropout: float = 0.3,
    ):
        super(ACORingGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input layer
        self.convs.append(
            PheromoneGuidedConv(
                in_channels,
                hidden_channels,
                num_relations=num_relations,
                lambda_blend=lambda_blend,
                fixed_lambda=fixed_lambda,
                dropout=dropout,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                PheromoneGuidedConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations=num_relations,
                    lambda_blend=lambda_blend,
                    fixed_lambda=fixed_lambda,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Classifier MLP head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        hat_tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass producing class logits [N, 2].
        """
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_type=edge_type, hat_tau=hat_tau)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Residual skip connection if dimensions match
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new

        logits = self.classifier(h)
        return logits
