"""
Evaluation metrics suite for graph fraud detection.
Focuses on gold-standard metrics for extreme class imbalance:
- Minority Illicit-F1 (F1 score strictly on the positive/fraud class)
- Macro-F1
- AUC-ROC (Area Under Receiver Operating Characteristic)
- PR-AUC / AP (Average Precision / Area Under Precision-Recall Curve)
- Inference runtime latency tracking (ms per batch)
"""

import time
import numpy as np
import torch
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


class FraudMetricsTracker:
    """Computes and tracks classification performance and inference efficiency."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.elapsed_time_ms = 0.0

    def start_timer(self):
        self.start_time = time.perf_counter()

    def stop_timer(self):
        if self.start_time is not None:
            self.elapsed_time_ms = (time.perf_counter() - self.start_time) * 1000.0

    @staticmethod
    def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find the optimal decision threshold that maximizes Illicit-F1 on validation data."""
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob)
        if len(np.unique(y_true)) < 2:
            return 0.5
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresholds):
            best_thresh = float(thresholds[best_idx])
            return max(0.05, min(0.95, best_thresh))
        return 0.5

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth binary labels (0 = benign, 1 = fraud)
            y_prob: Predicted probability of being fraud (class 1)
            threshold: Decision boundary threshold
            
        Returns:
            Dict containing illicit_f1, macro_f1, auc_roc, pr_auc, latency_ms
        """
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        # 1. Minority Illicit-F1 (pos_label=1)
        illicit_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # 2. Macro-F1
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # 3. AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_roc = 0.5

        # 4. PR-AUC (Average Precision)
        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            pr_auc = 0.0

        metrics = {
            "illicit_f1": float(illicit_f1),
            "macro_f1": float(macro_f1),
            "auc_roc": float(auc_roc),
            "pr_auc": float(pr_auc),
            "latency_ms": float(self.elapsed_time_ms),
        }

        return metrics

    @staticmethod
    def format_summary(metrics: Dict[str, float], model_name: str = "Model") -> str:
        """Format metrics into a publication-ready string."""
        return (
            f"[{model_name}] "
            f"Illicit-F1: {metrics['illicit_f1']:.4f} | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | "
            f"AUC-ROC: {metrics['auc_roc']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Latency: {metrics['latency_ms']:.2f}ms"
        )
