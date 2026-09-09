"""
Vectorized Ant Colony Stigmergy Engine.
Simulates collective ant behavior on graph edges using sparse matrix operations
to discover multi-hop collusive fraud syndicates and dissolve camouflage noise.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class VectorizedACOEngine:
    """
    Stigmergic Pheromone Manager and Vectorized Random Walker.
    
    Attributes:
        alpha: Pheromone sensitivity exponent
        beta: Heuristic desirability exponent
        rho: Pheromone evaporation rate (decay per training epoch)
        q: Base pheromone deposit quantity
        tau_min: Minimum pheromone threshold (prevents starvation)
        tau_max: Maximum pheromone threshold (prevents saturation)
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int = 1,
        alpha: float = 1.0,
        beta: float = 1.0,
        rho: float = 0.05,
        q: float = 1.0,
        tau_min: float = 0.01,
        tau_max: float = 10.0,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.temperature = temperature
        self.device = device

        # Pheromone values indexed by edge ID
        self.pheromone_values: Optional[torch.Tensor] = None

    def initialize_pheromones(self, num_edges: int):
        """Initialize all edge pheromone trails to a uniform baseline (1.0)."""
        self.pheromone_values = torch.full(
            (num_edges,),
            fill_value=1.0,
            dtype=torch.float32,
            device=self.device,
        )

    def compute_heuristic_desirability(
        self,
        edge_index: torch.Tensor,
        node_risk_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute edge heuristic desirability eta_ij.
        In lending graphs, edges connecting high-risk nodes have higher prior anomaly desirability.
        """
        if node_risk_scores is None:
            return torch.ones(edge_index.shape[1], device=self.device)

        src, dst = edge_index[0], edge_index[1]
        risk_src = node_risk_scores[src]
        risk_dst = node_risk_scores[dst]
        # Multiplicative joint risk: high when both endpoints exhibit suspicious signals
        eta = torch.sqrt(risk_src * risk_dst + 1e-6)
        return eta

    def evaporate(self):
        """Apply exponential pheromone evaporation: tau <- (1 - rho) * tau."""
        if self.pheromone_values is not None:
            self.pheromone_values.mul_(1.0 - self.rho)
            self.pheromone_values.clamp_(min=self.tau_min, max=self.tau_max)

    def reinforce(
        self,
        edge_index: torch.Tensor,
        node_predictions: torch.Tensor,
        node_labels: Optional[torch.Tensor] = None,
        reinforce_weight: float = 1.0,
    ):
        """
        Reinforce pheromone trails on suspicious collusive paths.
        Edges connecting nodes predicted as high-risk fraud rings receive deposit Delta tau.
        
        Args:
            node_labels: Labels with -1 for unknown/hidden nodes (val/test).
                         Only nodes with label != -1 contribute ground-truth reward.
        """
        if self.pheromone_values is None:
            return

        src, dst = edge_index[0], edge_index[1]
        prob_src = node_predictions[src]
        prob_dst = node_predictions[dst]

        # Reward edges connecting confirmed or high-probability fraud rings
        if node_labels is not None:
            # Only use ground truth for nodes with visible labels (label != -1)
            visible_src = (node_labels[src] != -1).float()
            visible_dst = (node_labels[dst] != -1).float()
            gt_src = (node_labels[src] == 1).float() * visible_src
            gt_dst = (node_labels[dst] == 1).float() * visible_dst
            ground_truth_reward = gt_src * gt_dst * 3.0
        else:
            ground_truth_reward = 0.0

        joint_risk_signal = prob_src * prob_dst + ground_truth_reward
        delta_tau = self.q * reinforce_weight * joint_risk_signal

        self.pheromone_values.add_(delta_tau)
        self.pheromone_values.clamp_(min=self.tau_min, max=self.tau_max)

    def get_normalized_pheromones(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute softmax-normalized pheromone weights hat{tau}_ij across each node's
        local neighborhood, optionally per relation type.
        """
        if self.pheromone_values is None:
            self.initialize_pheromones(edge_index.shape[1])

        tau = self.pheromone_values
        src, dst = edge_index[0], edge_index[1]
        num_edges = edge_index.shape[1]

        if edge_type is not None and self.num_relations > 1:
            # Relation-aware normalization: softmax within each (dst_node, relation) group
            # Create unique group keys: dst * num_relations + edge_type
            group_key = dst * self.num_relations + edge_type

            exp_tau = torch.exp(tau / self.temperature)

            # Sum exp_tau per group
            max_group = int(group_key.max().item()) + 1
            group_sum = torch.zeros(max_group, device=self.device)
            group_sum.scatter_add_(0, group_key, exp_tau)

            denom = group_sum[group_key].clamp(min=1e-8)
            hat_tau = exp_tau / denom
        else:
            # Single-relation fallback: normalize per destination node
            exp_tau = torch.exp(tau / self.temperature)
            deg = torch.zeros(self.num_nodes, device=self.device)
            deg.scatter_add_(0, dst, exp_tau)
            denom = deg[dst].clamp(min=1e-8)
            hat_tau = exp_tau / denom

        return hat_tau
