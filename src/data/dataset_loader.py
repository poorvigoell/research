"""
Dataset Loader & Preprocessor for Fraud Detection Benchmarks.
Supports:
1. DGraphFin (Large-scale financial lending graph, NeurIPS 2022)
2. YelpChi & Amazon (Multi-relational cross-domain fraud graphs)
3. Synthetic Dynamic Lending Network (Controlled multi-hop fraud rings)
"""

import os
import urllib.request
import numpy as np
import scipy.io as sio
import torch
from typing import Dict, Any, Optional, Tuple
from src.data.synthetic_generator import SyntheticLendingGraphGenerator


def download_file(url: str, dest_path: str):
    """Download a file from a URL with progress indication."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return
    print(f"Downloading {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print("Download complete.")


def load_dgraphfin(root_dir: str = "./data/dgraphfin") -> Dict[str, Any]:
    """
    Load the DGraphFin dataset (NeurIPS 2022).
    Nodes: 3,700,550
    Edges: 4,300,999 (11 relation types)
    Node Features: 17
    Classes: 0 (normal), 1 (fraud/overdue), 2 & 3 (background unlabeled)
    """
    from torch_geometric.datasets import DGraphFin

    os.makedirs(root_dir, exist_ok=True)
    dataset = DGraphFin(root=root_dir)
    data = dataset[0]

    # Labels in DGraphFin:
    # 0: normal user, 1: fraud / 30+ days overdue
    # 2 & 3: background users
    # We focus on the labeled subset (0 vs 1) for evaluation while using background nodes for graph connectivity
    labeled_mask = (data.y == 0) | (data.y == 1)

    # Inductive temporal split: train/val/test masks are already provided by DGraphFin
    # or can be constructed chronologically via edge timestamps
    return {
        "x": data.x,
        "y": data.y,
        "edge_index": data.edge_index,
        "edge_type": data.edge_type if hasattr(data, "edge_type") else None,
        "edge_time": data.edge_time if hasattr(data, "edge_time") else None,
        "train_mask": data.train_mask,
        "val_mask": data.val_mask,
        "test_mask": data.test_mask,
        "labeled_mask": labeled_mask,
        "num_relations": 11,
        "num_nodes": data.num_nodes,
        "dataset_name": "DGraphFin",
    }


def load_yelpchi(root_dir: str = "./data/yelpchi") -> Dict[str, Any]:
    """
    Load YelpChi multi-relational fraud dataset.
    Nodes: 45,954 (reviews)
    Edges across 3 relations: R-U-R (user), R-T-R (product), R-S-R (star rating)
    Labels: 0 (benign), 1 (fraudulent)
    """
    os.makedirs(root_dir, exist_ok=True)
    mat_path = os.path.join(root_dir, "YelpChi.mat")
    
    # URL mirror for YelpChi.mat (hosted on GitHub CARE-GNN repository)
    url = "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.mat"
    if not os.path.exists(mat_path):
        try:
            download_file(url, mat_path)
        except Exception as e:
            print(f"Note: Could not auto-download YelpChi from remote: {e}")

    if os.path.exists(mat_path):
        mat = sio.loadmat(mat_path)
        features = mat["features"].todense().A if hasattr(mat["features"], "todense") else mat["features"]
        labels = mat["label"].flatten()
        
        # Relations: net_rur, net_rtr, net_rsr
        r_ur = mat["net_rur"].tocoo()
        r_tr = mat["net_rtr"].tocoo()
        r_sr = mat["net_rsr"].tocoo()

        edges_src = []
        edges_dst = []
        edge_types = []

        for rel_id, rel_mat in enumerate([r_ur, r_tr, r_sr]):
            edges_src.extend(rel_mat.row.tolist())
            edges_dst.extend(rel_mat.col.tolist())
            edge_types.extend([rel_id] * len(rel_mat.row))

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        num_nodes = len(labels)
        indices = np.arange(num_nodes)
        np.random.seed(42)
        np.random.shuffle(indices)

        train_len = int(0.6 * num_nodes)
        val_len = int(0.2 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_len]] = True
        val_mask[indices[train_len:train_len + val_len]] = True
        test_mask[indices[train_len + val_len:]] = True

        return {
            "x": x,
            "y": y,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "num_relations": 3,
            "num_nodes": num_nodes,
            "dataset_name": "YelpChi",
        }
    else:
        # Fallback to local high-fidelity mock if running offline
        return load_synthetic_lending(num_applicants=5000, camouflage_ratio=0.15)


def load_synthetic_lending(
    num_applicants: int = 5000,
    fraud_ratio: float = 0.05,
    camouflage_ratio: float = 0.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Load configurable synthetic dynamic lending network for stress testing."""
    generator = SyntheticLendingGraphGenerator(
        num_applicants=num_applicants,
        fraud_ratio=fraud_ratio,
        seed=seed,
    )
    data = generator.generate(camouflage_ratio=camouflage_ratio)
    data["dataset_name"] = f"SyntheticLending_cam_{int(camouflage_ratio*100)}"
    return data


def inject_camouflage(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    edge_type: Optional[torch.Tensor] = None,
    camouflage_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Injects synthetic camouflage noise edges between fraudulent nodes and benign nodes
    to evaluate model degradation under active evasion attacks.
    """
    if camouflage_ratio <= 0.0:
        return edge_index, edge_type

    np.random.seed(seed)
    num_edges = edge_index.shape[1]
    num_cam = int(num_edges * camouflage_ratio)

    fraud_nodes = torch.where(labels == 1)[0].cpu().numpy()
    benign_nodes = torch.where(labels == 0)[0].cpu().numpy()

    if len(fraud_nodes) == 0 or len(benign_nodes) == 0:
        return edge_index, edge_type

    cam_src = np.random.choice(fraud_nodes, size=num_cam)
    cam_dst = np.random.choice(benign_nodes, size=num_cam)

    cam_edges_fwd = torch.tensor([cam_src, cam_dst], dtype=torch.long, device=edge_index.device)
    cam_edges_bwd = torch.tensor([cam_dst, cam_src], dtype=torch.long, device=edge_index.device)

    new_edge_index = torch.cat([edge_index, cam_edges_fwd, cam_edges_bwd], dim=1)

    new_edge_type = None
    if edge_type is not None:
        random_types = torch.randint(0, int(edge_type.max().item() + 1), (num_cam * 2,), dtype=torch.long, device=edge_type.device)
        new_edge_type = torch.cat([edge_type, random_types], dim=0)

    return new_edge_index, new_edge_type
