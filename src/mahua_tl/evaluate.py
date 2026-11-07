from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images).view(-1)
        logits.append(outputs.detach().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(labels).astype(int)


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    return compute_metrics_at_threshold(probs, labels, threshold=0.5)


def compute_metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    cm = binary_confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    healthy_precision = _safe_div(tn, tn + fn)
    healthy_recall = _safe_div(tn, tn + fp)
    healthy_f1 = _safe_div(2 * healthy_precision * healthy_recall, healthy_precision + healthy_recall)
    unhealthy_precision = _safe_div(tp, tp + fp)
    unhealthy_recall = _safe_div(tp, tp + fn)
    unhealthy_f1 = _safe_div(2 * unhealthy_precision * unhealthy_recall, unhealthy_precision + unhealthy_recall)
    specificity = healthy_recall
    metrics = {
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": unhealthy_precision,
        "recall": unhealthy_recall,
        "f1": unhealthy_f1,
        "healthy_precision": healthy_precision,
        "healthy_recall": healthy_recall,
        "healthy_f1": healthy_f1,
        "unhealthy_precision": unhealthy_precision,
        "unhealthy_recall": unhealthy_recall,
        "unhealthy_f1": unhealthy_f1,
        "specificity": specificity,
        "balanced_accuracy": (specificity + unhealthy_recall) / 2,
        "youden_j": unhealthy_recall + specificity - 1.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    metrics["roc_auc"] = binary_roc_auc(labels, probs)
    return metrics


def find_best_threshold(logits: np.ndarray, labels: np.ndarray, metric: str = "youden_j") -> tuple[float, dict[str, float]]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    if metric == "fixed_0_5":
        return 0.5, compute_metrics_at_threshold(probs, labels, threshold=0.5)
    best_threshold = 0.5
    best_metrics = compute_metrics_at_threshold(probs, labels, best_threshold)
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = compute_metrics_at_threshold(probs, labels, float(threshold))
        if metrics[metric] > best_metrics[metric]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def plot_confusion_matrix(logits: np.ndarray, labels: np.ndarray, output_path: Path, threshold: float = 0.5) -> None:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    cm = binary_confusion_matrix(labels, preds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Healthy", "Unhealthy"])
    ax.set_yticks([0, 1], ["Healthy", "Unhealthy"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    cutoff = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > cutoff else "black"
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    labels = labels.astype(int)
    preds = preds.astype(int)
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def binary_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = labels.astype(int)
    positive = probs[labels == 1]
    negative = probs[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=float)
    sorted_probs = probs[order]
    start = 0
    while start < len(sorted_probs):
        end = start + 1
        while end < len(sorted_probs) and sorted_probs[end] == sorted_probs[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - positive.size * (positive.size + 1) / 2.0) / (positive.size * negative.size)
