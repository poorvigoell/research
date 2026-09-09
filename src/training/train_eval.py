"""
Coupled Training & Evaluation Engine for ACO-GNN and Baselines.
Implements the closed-loop optimization cycle:
Forward -> Loss -> Backward -> Pheromone Reinforcement -> Evaporation -> Validation -> Early Stopping.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple

from src.aco.matrix_ant import VectorizedACOEngine
from src.training.losses import FocalLoss
from src.utils.metrics import FraudMetricsTracker


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    data: Dict[str, Any],
    aco_engine: Optional[VectorizedACOEngine] = None,
    device: str = "cpu",
) -> float:
    """Run a single training epoch."""
    model.train()
    optimizer.zero_grad()

    x = data["x"].to(device)
    edge_index = data["edge_index"].to(device)
    edge_type = data.get("edge_type", None)
    if edge_type is not None:
        edge_type = edge_type.to(device)
    y = data["y"].to(device)
    train_mask = data["train_mask"].to(device)

    # 1. Obtain normalized pheromones if ACO is active
    hat_tau = None
    if aco_engine is not None:
        hat_tau = aco_engine.get_normalized_pheromones(edge_index, edge_type=edge_type)

    # 2. Forward pass
    logits = model(x, edge_index, edge_type=edge_type, hat_tau=hat_tau)

    # 3. Compute loss on training nodes
    loss = criterion(logits[train_mask], y[train_mask])

    # 4. Backward pass & gradient update
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()

    # 5. Closed-loop ACO Pheromone Update
    if aco_engine is not None:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1]
            # Mask labels: only training nodes have visible ground truth
            masked_labels = torch.full_like(y, -1)  # -1 = unknown
            masked_labels[train_mask] = y[train_mask]
            # Reinforce paths connecting high-confidence fraud predictions
            aco_engine.reinforce(
                edge_index=edge_index,
                node_predictions=probs,
                node_labels=masked_labels,
            )
            # Evaporate transient noise trails
            aco_engine.evaporate()

    return float(loss.item())


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: Dict[str, Any],
    mask_name: str = "val_mask",
    aco_engine: Optional[VectorizedACOEngine] = None,
    threshold: Optional[float] = None,
    device: str = "cpu",
) -> Tuple[Dict[str, float], float]:
    """Evaluate model on a given data mask (val_mask or test_mask)."""
    model.eval()
    tracker = FraudMetricsTracker()

    x = data["x"].to(device)
    edge_index = data["edge_index"].to(device)
    edge_type = data.get("edge_type", None)
    if edge_type is not None:
        edge_type = edge_type.to(device)
    y = data["y"].to(device)
    mask = data[mask_name].to(device)

    tracker.start_timer()
    hat_tau = None
    if aco_engine is not None:
        hat_tau = aco_engine.get_normalized_pheromones(edge_index, edge_type=edge_type)

    logits = model(x, edge_index, edge_type=edge_type, hat_tau=hat_tau)
    tracker.stop_timer()

    probs = torch.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
    targets = y[mask].cpu().numpy()

    if threshold is None:
        threshold = FraudMetricsTracker.find_best_threshold(targets, probs)

    metrics = tracker.evaluate(targets, probs, threshold=threshold)
    return metrics, threshold


def train_and_evaluate(
    model: nn.Module,
    data: Dict[str, Any],
    aco_engine: Optional[VectorizedACOEngine] = None,
    epochs: int = 60,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str = "cpu",
    model_name: str = "ACO-GNN",
) -> Dict[str, Any]:
    """
    Complete training pipeline with early stopping on validation Illicit-F1.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    best_val_f1 = -1.0
    best_val_thresh = 0.5
    best_model_state = None
    best_pheromone_state = None
    best_epoch = 0
    patience_counter = 0

    print(f"\n--- Starting Training for {model_name} ({epochs} epochs) ---")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data=data,
            aco_engine=aco_engine,
            device=device,
        )

        val_metrics, val_thresh = evaluate_model(
            model=model,
            data=data,
            mask_name="val_mask",
            aco_engine=aco_engine,
            threshold=None,
            device=device,
        )

        val_f1 = val_metrics["illicit_f1"]

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                f"Val Illicit-F1: {val_metrics['illicit_f1']:.4f} (Thresh: {val_thresh:.2f}) | "
                f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
                f"Val AUC-ROC: {val_metrics['auc_roc']:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_thresh = val_thresh
            best_model_state = copy.deepcopy(model.state_dict())
            if aco_engine is not None and aco_engine.pheromone_values is not None:
                best_pheromone_state = aco_engine.pheromone_values.clone()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Best epoch: {best_epoch})")
                break

    # Load best model checkpoint and pheromone state for final test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_pheromone_state is not None and aco_engine is not None:
        aco_engine.pheromone_values = best_pheromone_state.clone()

    test_metrics, _ = evaluate_model(
        model=model,
        data=data,
        mask_name="test_mask",
        aco_engine=aco_engine,
        threshold=best_val_thresh,
        device=device,
    )

    print("\n" + "=" * 60)
    print(f"FINAL TEST RESULTS: {FraudMetricsTracker.format_summary(test_metrics, model_name)} (Thresh: {best_val_thresh:.2f})")
    print("=" * 60)

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "model_name": model_name,
    }
