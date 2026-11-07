from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.mahua_tl.config import ExperimentConfig
from src.mahua_tl.data import build_dataloaders, discover_samples, load_split_manifest, split_summary, stratified_split
from src.mahua_tl.models import build_model
from src.mahua_tl.train import train_one_model
from src.mahua_tl.utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary transfer learning on the Mahua Leaf Dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=ExperimentConfig.dataset_dir)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    parser.add_argument("--models", nargs="+", default=ExperimentConfig().models)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument("--image-size", type=int, default=ExperimentConfig.image_size)
    parser.add_argument("--num-workers", type=int, default=ExperimentConfig.num_workers)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--mobileplantvit-checkpoint", type=str, default=None)
    parser.add_argument("--no-imagenet-pretrained", action="store_true")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument(
        "--threshold-metric",
        choices=["youden_j", "balanced_accuracy", "f1", "fixed_0_5"],
        default="youden_j",
        help="Validation criterion for selecting the binary decision threshold.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=["f1", "roc_auc", "youden_j", "balanced_accuracy"],
        default="f1",
        help="Validation criterion for checkpoint selection, LR scheduling and early stopping.",
    )
    parser.add_argument("--split-manifest", type=Path, default=None, help="Fixed split JSON; overrides the random split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        models=args.models,
    )
    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)
    samples = discover_samples(cfg.dataset_dir)
    if args.split_manifest is not None:
        splits = load_split_manifest(args.split_manifest, samples)
    else:
        splits = stratified_split(samples, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    save_json(split_summary(splits), output_dir / "split_summary.json")
    dataloaders = build_dataloaders(
        splits,
        cfg.image_size,
        cfg.batch_size,
        cfg.num_workers,
        balanced_sampler=not args.no_balanced_sampler,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for model_name in cfg.models:
        checkpoint = args.mobileplantvit_checkpoint if model_name.lower() == "mobileplantvit" else None
        model = build_model(model_name, pretrained=not args.no_imagenet_pretrained, checkpoint_path=checkpoint)
        metrics = train_one_model(
            model=model,
            model_name=model_name,
            dataloaders=dataloaders,
            output_dir=output_dir,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            scheduler_patience=cfg.scheduler_patience,
            early_stopping_patience=cfg.early_stopping_patience,
            device=device,
            threshold_metric=args.threshold_metric,
            checkpoint_metric=args.checkpoint_metric,
        )
        rows.append({"model": model_name, **metrics})

    rows = sorted(rows, key=lambda row: float(row["f1"]), reverse=True)
    fieldnames = ["model"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "test_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    widths = {name: max(len(name), *(len(f"{row.get(name, ''):.4f}") if isinstance(row.get(name), float) else len(str(row.get(name, ""))) for row in rows)) for name in fieldnames}
    print(" ".join(name.rjust(widths[name]) for name in fieldnames))
    for row in rows:
        cells = []
        for name in fieldnames:
            value = row.get(name, "")
            cells.append((f"{value:.4f}" if isinstance(value, float) else str(value)).rjust(widths[name]))
        print(" ".join(cells))


if __name__ == "__main__":
    main()
