from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import compute_metrics_at_threshold, find_best_threshold, plot_confusion_matrix, predict_logits
from .utils import ensure_dir


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    for images, targets in tqdm(loader, leave=False, desc="train" if is_train else "eval"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()
        with torch.set_grad_enabled(is_train):
            logits = model(images).view(-1)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
    return total_loss / max(1, total_items)


def train_one_model(
    model: nn.Module,
    model_name: str,
    dataloaders: dict[str, DataLoader],
    output_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    scheduler_patience: int,
    early_stopping_patience: int,
    device: torch.device,
    threshold_metric: str = "youden_j",
    checkpoint_metric: str = "f1",
) -> dict[str, float]:
    model_dir = ensure_dir(output_dir / model_name)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=scheduler_patience)
    best_score = -1.0
    best_path = model_dir / "best.pt"
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, dataloaders["train"], criterion, device, optimizer)
        val_loss = _run_epoch(model, dataloaders["val"], criterion, device)
        val_logits, val_labels = predict_logits(model, dataloaders["val"], device)
        threshold, val_metrics = find_best_threshold(val_logits, val_labels, metric=threshold_metric)
        score = val_metrics[checkpoint_metric]
        scheduler.step(score)

        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": int(epoch),
                    "val_score": float(best_score),
                    "checkpoint_metric": checkpoint_metric,
                    "threshold": float(threshold),
                    "threshold_metric": threshold_metric,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        print(
            f"{model_name} epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f} val_{checkpoint_metric}={score:.4f}"
        )
        if stale_epochs >= early_stopping_patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    threshold = float(checkpoint.get("threshold", 0.5))
    test_logits, test_labels = predict_logits(model, dataloaders["test"], device)
    probs = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
    # keep raw probabilities so thresholds, bootstrap CIs and ROC curves can be computed post hoc
    val_logits, val_labels = predict_logits(model, dataloaders["val"], device)
    np.savez(
        model_dir / "predictions.npz",
        val_probs=torch.sigmoid(torch.from_numpy(val_logits)).numpy(),
        val_labels=val_labels,
        test_probs=probs,
        test_labels=test_labels,
    )
    metrics = compute_metrics_at_threshold(probs, test_labels, threshold=threshold)
    metrics["threshold"] = threshold
    metrics["threshold_metric"] = threshold_metric
    metrics["best_epoch"] = int(checkpoint["epoch"])
    plot_confusion_matrix(test_logits, test_labels, model_dir / "confusion_matrix.png", threshold=threshold)
    return metrics
