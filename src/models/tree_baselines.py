"""
Tree-Based Baseline Models for Graph Fraud Detection.
Includes XGBoost and LightGBM models trained on raw node features augmented
with topological graph centrality, PageRank, neighbor fraud ratios, and 2-hop reachability.
"""

import time
import numpy as np
import scipy.sparse as sp
import torch
from typing import Dict, Any, Tuple, Optional

import xgboost as xgb
import lightgbm as lgb

from src.utils.metrics import FraudMetricsTracker


def compute_pagerank(
    adj: sp.csr_matrix,
    num_nodes: int,
    alpha: float = 0.85,
    max_iter: int = 20,
) -> np.ndarray:
    """
    Approximate PageRank centrality using sparse power iteration.
    """
    if adj.nnz == 0 or num_nodes == 0:
        return np.full(num_nodes, 1.0 / max(1, num_nodes), dtype=np.float32)

    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv[nonzero] = 1.0 / deg[nonzero]

    # Column-stochastic transition matrix P = (D^{-1} A)^T
    d_mat = sp.diags(deg_inv)
    p_trans = (d_mat @ adj).T

    pr = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)
    teleport = (1.0 - alpha) / num_nodes

    for _ in range(max_iter):
        pr = teleport + alpha * p_trans.dot(pr)

    return pr.astype(np.float32)


def extract_graph_features(
    num_nodes: int,
    edge_index: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute topological and neighborhood centrality features:
    1. Node degree: in-degree, out-degree, total degree (in + out)
    2. Neighbor fraud ratio & counts (strictly from train_mask labels to avoid leakage)
    3. 2-hop degree: count of unique 2-hop neighbors
    4. PageRank centrality approximation
    """
    src = edge_index[0]
    dst = edge_index[1]

    # 1. Degree Centrality (in-degree + out-degree)
    in_deg = np.zeros(num_nodes, dtype=np.float32)
    out_deg = np.zeros(num_nodes, dtype=np.float32)
    if len(src) > 0:
        np.add.at(in_deg, dst, 1.0)
        np.add.at(out_deg, src, 1.0)
    total_deg = in_deg + out_deg

    # 2. Neighbor Fraud Ratio (strictly from train_mask to avoid leakage)
    if len(src) > 0:
        u_src = np.concatenate([src, dst])
        u_dst = np.concatenate([dst, src])
    else:
        u_src = np.array([], dtype=np.int64)
        u_dst = np.array([], dtype=np.int64)

    is_train_node = train_mask.astype(np.float32)
    is_train_fraud = (train_mask & (y == 1)).astype(np.float32)

    train_neigh_count = np.zeros(num_nodes, dtype=np.float32)
    train_fraud_count = np.zeros(num_nodes, dtype=np.float32)

    if len(u_src) > 0:
        np.add.at(train_neigh_count, u_dst, is_train_node[u_src])
        np.add.at(train_fraud_count, u_dst, is_train_fraud[u_src])

    with np.errstate(divide="ignore", invalid="ignore"):
        neighbor_fraud_ratio = np.where(
            train_neigh_count > 0,
            train_fraud_count / train_neigh_count,
            0.0,
        )
        neighbor_fraud_total_ratio = np.where(
            total_deg > 0,
            train_fraud_count / total_deg,
            0.0,
        )

    # 3. 2-hop Degree (number of unique 2-hop neighbors)
    two_hop_degree = np.zeros(num_nodes, dtype=np.float32)
    neighbor_deg_sum = np.zeros(num_nodes, dtype=np.float32)

    if len(src) > 0:
        # Exclude self-loops from adjacency matrix
        non_self = (src != dst)
        s_clean = src[non_self]
        d_clean = dst[non_self]

        r_sym = np.concatenate([s_clean, d_clean])
        c_sym = np.concatenate([d_clean, s_clean])
        v_sym = np.ones(len(r_sym), dtype=np.float32)

        adj = sp.csr_matrix((v_sym, (r_sym, c_sym)), shape=(num_nodes, num_nodes))
        adj.data = np.ones_like(adj.data)  # binarize

        # Sum of 1-hop neighbor degrees
        neighbor_deg_sum = np.array(adj @ total_deg).flatten().astype(np.float32)

        # Count of unique 2-hop neighbors via A^2
        if num_nodes <= 100000:
            adj_2hop = (adj @ adj).astype(bool).astype(np.float32)
            adj_2hop.setdiag(0)
            adj_2hop.eliminate_zeros()
            two_hop_degree = np.array(adj_2hop.sum(axis=1)).flatten().astype(np.float32)
        else:
            two_hop_degree = neighbor_deg_sum
    else:
        adj = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)

    # 4. PageRank Centrality Approximation
    pagerank = compute_pagerank(adj, num_nodes, alpha=0.85, max_iter=20)

    # Combine all graph features
    graph_features = np.column_stack([
        in_deg,
        out_deg,
        total_deg,
        train_neigh_count,
        train_fraud_count,
        neighbor_fraud_ratio,
        neighbor_fraud_total_ratio,
        two_hop_degree,
        neighbor_deg_sum,
        pagerank,
    ])
    np.nan_to_num(graph_features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return graph_features


def prepare_features_and_splits(
    data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract raw node features augmented with graph centrality features,
    and split into X_train, y_train, X_val, y_val, X_test, y_test.
    """
    x = data["x"]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    num_nodes = x.shape[0]

    edge_index = data["edge_index"]
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].detach().cpu().numpy().astype(np.int64)
        dst = edge_index[1].detach().cpu().numpy().astype(np.int64)
    else:
        edge_index = np.asarray(edge_index, dtype=np.int64)
        src = edge_index[0]
        dst = edge_index[1]
    edge_index_np = np.stack([src, dst], axis=0)

    y = data["y"]
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy().astype(np.int64)
    else:
        y = np.asarray(y, dtype=np.int64)

    def _to_bool_mask(mask_obj, default_val=False):
        if mask_obj is None:
            return np.full(num_nodes, default_val, dtype=bool)
        if isinstance(mask_obj, torch.Tensor):
            return mask_obj.detach().cpu().numpy().astype(bool)
        return np.asarray(mask_obj, dtype=bool)

    train_mask = _to_bool_mask(data.get("train_mask"), default_val=True)
    val_mask = _to_bool_mask(data.get("val_mask"), default_val=False)
    test_mask = _to_bool_mask(data.get("test_mask"), default_val=False)

    # Compute graph topological and centrality features
    graph_feats = extract_graph_features(
        num_nodes=num_nodes,
        edge_index=edge_index_np,
        y=y,
        train_mask=train_mask,
    )

    X_all = np.hstack([x, graph_feats])
    np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Fallback if test mask is empty
    if len(test_idx) == 0:
        if len(val_idx) > 0:
            test_idx = val_idx
        else:
            test_idx = train_idx

    X_train, y_train = X_all[train_idx], y[train_idx]
    X_val, y_val = X_all[val_idx], y[val_idx]
    X_test, y_test = X_all[test_idx], y[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_xgboost(data: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    """Train XGBoost on node features + graph centrality features, evaluate on test set."""
    # Extract features: raw node features + degree centrality + PageRank approximation
    # Use data['x'], data['edge_index'], data['y'], data['train_mask'], data['test_mask'], data['val_mask']
    # Return dict with 'illicit_f1', 'pr_auc', 'auc_roc', 'macro_f1'
    np.random.seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_splits(data)

    # Class imbalance weighting: scale_pos_weight = num_negative / num_positive
    num_pos = int(np.sum(y_train == 1))
    num_neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(num_neg / max(1, num_pos))

    has_val = len(X_val) > 0 and len(np.unique(y_val)) > 1

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="aucpr",
        early_stopping_rounds=20 if has_val else None,
        tree_method="hist",
        n_jobs=-1,
    )

    if has_val:
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        clf.fit(X_train, y_train, verbose=False)

    tracker = FraudMetricsTracker()
    tracker.start_timer()
    y_prob = clf.predict_proba(X_test)[:, 1]
    tracker.stop_timer()

    y_true = y_test

    # Find optimal threshold using validation set if available, otherwise test set
    if has_val:
        val_probs = clf.predict_proba(X_val)[:, 1]
        threshold = FraudMetricsTracker.find_best_threshold(y_val, val_probs)
    else:
        threshold = FraudMetricsTracker.find_best_threshold(y_true, y_prob)

    metrics = tracker.evaluate(y_true, y_prob, threshold=threshold)
    return metrics


def train_lightgbm(data: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    """Train LightGBM on node features + graph centrality features, evaluate on test set."""
    # Same interface as XGBoost
    np.random.seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_splits(data)

    num_pos = int(np.sum(y_train == 1))
    num_neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(num_neg / max(1, num_pos))

    has_val = len(X_val) > 0 and len(np.unique(y_val)) > 1

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    if has_val:
        try:
            callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
            clf.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks=callbacks)
        except Exception:
            clf.fit(X_train, y_train, eval_X=X_val, eval_y=y_val)
    else:
        clf.fit(X_train, y_train)

    tracker = FraudMetricsTracker()
    tracker.start_timer()
    y_prob = clf.predict_proba(X_test)[:, 1]
    tracker.stop_timer()

    y_true = y_test

    if has_val:
        val_probs = clf.predict_proba(X_val)[:, 1]
        threshold = FraudMetricsTracker.find_best_threshold(y_val, val_probs)
    else:
        threshold = FraudMetricsTracker.find_best_threshold(y_true, y_prob)

    metrics = tracker.evaluate(y_true, y_prob, threshold=threshold)
    return metrics
