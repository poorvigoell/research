"""
Synthetic Dynamic Lending Fraud Network Generator.
Generates heterogeneous lending graphs with controllable multi-hop fraud rings,
temporal timestamps, and tunable camouflage noise (10% - 50%).
Used for controlled ablation studies and camouflage stress testing.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import networkx as nx


class SyntheticLendingGraphGenerator:
    """
    Generates a heterogeneous multi-hop financial lending network:
    - Nodes: Applicants (users), Devices, BankAccounts, Guarantors
    - Edges: SHARES_DEVICE, SHARES_ACCOUNT, GUARANTEES, CO_APPLIES
    - Fraud Patterns: Multi-hop collusive fraud rings with intentional camouflage edges
    """

    def __init__(
        self,
        num_applicants: int = 5000,
        fraud_ratio: float = 0.05,
        num_devices: int = 3000,
        num_accounts: int = 3500,
        num_guarantors: int = 800,
        num_timesteps: int = 50,
        feature_dim: int = 16,
        seed: int = 42,
    ):
        self.num_applicants = num_applicants
        self.fraud_ratio = fraud_ratio
        self.num_devices = num_devices
        self.num_accounts = num_accounts
        self.num_guarantors = num_guarantors
        self.num_timesteps = num_timesteps
        self.feature_dim = feature_dim
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate(
        self,
        camouflage_ratio: float = 0.0,
        ring_min_size: int = 3,
        ring_max_size: int = 6,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate the synthetic network with a specified camouflage ratio.
        
        Args:
            camouflage_ratio: Fraction of camouflage edges connecting fraudsters to legitimate entities (0.0 to 0.5)
            ring_min_size: Minimum number of applicants in a collusive fraud syndicate
            ring_max_size: Maximum number of applicants in a collusive fraud syndicate
            seed: Optional random seed override for reproducibility
            
        Returns:
            Dict containing:
                - x: Node feature tensor [num_applicants, feature_dim]
                - y: Binary label tensor [num_applicants] (0 = legitimate, 1 = fraud)
                - edge_index: [2, num_edges] homogeneous projection or applicant-applicant graph
                - edge_type: [num_edges] relation type ID
                - edge_time: [num_edges] timestamp
                - is_camouflage: [num_edges] boolean indicator for camouflage noise evaluation
        """
        active_seed = seed if seed is not None else self.seed
        np.random.seed(active_seed)
        torch.manual_seed(active_seed)
        num_fraud = int(self.num_applicants * self.fraud_ratio)
        labels = np.zeros(self.num_applicants, dtype=np.int64)
        fraud_indices = np.random.choice(self.num_applicants, size=num_fraud, replace=False)
        labels[fraud_indices] = 1
        fraud_set = set(fraud_indices)

        # 1. Generate node features
        # Fraudulent nodes have subtly shifted feature distributions but significant overlap (simulating camouflage)
        features = np.random.normal(loc=0.0, scale=1.0, size=(self.num_applicants, self.feature_dim))
        # Add subtle risk signal to a subset of features (imperfect signal)
        features[fraud_indices, :4] += np.random.normal(loc=0.8, scale=0.5, size=(num_fraud, 4))

        # 2. Build multi-relational edges
        edges_src = []
        edges_dst = []
        edges_type = []
        edges_time = []
        is_camouflage = []

        # Relation Types:
        # 0: SHARES_DEVICE (Applicant <-> Applicant via Device)
        # 1: SHARES_ACCOUNT (Applicant <-> Applicant via Bank Account)
        # 2: GUARANTEED_BY (Applicant <-> Applicant via Guarantor)
        # 3: TRANSACTION_FLOW (Directed temporal transfer)

        # Normal benign connections (sparse, organic)
        for i in range(self.num_applicants):
            if i not in fraud_set:
                # Organic connections to 1-3 other benign applicants
                degree = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
                neighbors = np.random.choice(self.num_applicants, size=degree, replace=False)
                t = np.random.randint(1, self.num_timesteps + 1)
                for neighbor in neighbors:
                    if neighbor != i:
                        rel = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
                        edges_src.extend([i, neighbor])
                        edges_dst.extend([neighbor, i])
                        edges_type.extend([rel, rel])
                        edges_time.extend([t, t])
                        is_cam = (neighbor in fraud_set)  # If connected to a fraud node organically
                        is_camouflage.extend([is_cam, is_cam])

        # 3. Build Collusive Fraud Rings (Densely connected multi-hop cliques)
        unassigned_fraud = list(fraud_indices.copy())
        np.random.shuffle(unassigned_fraud)

        while len(unassigned_fraud) >= ring_min_size:
            ring_size = min(len(unassigned_fraud), np.random.randint(ring_min_size, ring_max_size + 1))
            ring_members = [unassigned_fraud.pop() for _ in range(ring_size)]
            
            # Coordinated fraud ring timeframe
            ring_time = np.random.randint(5, self.num_timesteps)

            # High density of shared devices and guarantors within the ring
            for idx_a in range(len(ring_members)):
                for idx_b in range(idx_a + 1, len(ring_members)):
                    u = ring_members[idx_a]
                    v = ring_members[idx_b]
                    # Multiple collusive relations connecting the ring
                    for rel in [0, 1, 2]:
                        if np.random.rand() > 0.2:  # 80% chance of sharing this resource
                            edges_src.extend([u, v])
                            edges_dst.extend([v, u])
                            edges_type.extend([rel, rel])
                            edges_time.extend([ring_time, ring_time])
                            is_camouflage.extend([False, False])

        # Any remaining fraud nodes connect to a random existing fraud ring
        for remaining in unassigned_fraud:
            partner = np.random.choice(fraud_indices)
            if partner != remaining:
                edges_src.extend([remaining, partner])
                edges_dst.extend([partner, remaining])
                edges_type.extend([0, 0])
                edges_time.extend([self.num_timesteps // 2, self.num_timesteps // 2])
                is_camouflage.extend([False, False])

        # 4. Inject Camouflage Edges
        # Fraudsters intentionally connect to legitimate entities to blend in
        if camouflage_ratio > 0.0:
            total_current_edges = len(edges_src) // 2
            num_cam_edges = int(total_current_edges * camouflage_ratio)
            legitimate_indices = np.array([i for i in range(self.num_applicants) if i not in fraud_set])

            for _ in range(num_cam_edges):
                f_node = np.random.choice(fraud_indices)
                legit_node = np.random.choice(legitimate_indices)
                rel = np.random.choice([0, 1, 2, 3])
                t = np.random.randint(1, self.num_timesteps + 1)

                edges_src.extend([f_node, legit_node])
                edges_dst.extend([legit_node, f_node])
                edges_type.extend([rel, rel])
                edges_time.extend([t, t])
                is_camouflage.extend([True, True])

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_type = torch.tensor(edges_type, dtype=torch.long)
        edge_time = torch.tensor(edges_time, dtype=torch.long)
        is_camouflage = torch.tensor(is_camouflage, dtype=torch.bool)
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)

        # Assign each applicant a realistic application timestamp t_i in [1, num_timesteps]
        # ensuring balanced representation of benign and fraud across train, val, test
        train_t_max = int(self.num_timesteps * 0.70)
        val_t_max = int(self.num_timesteps * 0.85)

        # Distribute applicant entry times across the entire timeline
        applicant_time = np.random.randint(1, self.num_timesteps + 1, size=self.num_applicants)

        train_mask = torch.tensor(applicant_time <= train_t_max, dtype=torch.bool)
        val_mask = torch.tensor(
            (applicant_time > train_t_max) & (applicant_time <= val_t_max),
            dtype=torch.bool,
        )
        test_mask = torch.tensor(applicant_time > val_t_max, dtype=torch.bool)

        return {
            "x": x,
            "y": y,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "edge_time": edge_time,
            "is_camouflage": is_camouflage,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "num_relations": 4,
            "num_nodes": self.num_applicants,
        }
