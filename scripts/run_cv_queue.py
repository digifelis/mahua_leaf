"""Grouped 5-fold CV queue: 4 architectures x {weighted, standard} x 5 folds (seed 42), same settings as
run_revision2_queue.py. Output artifacts/revision2/cv/runs/<sampler>/fold<k>/<model>/predictions.npz. Resumable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "artifacts/revision2/cv"
FOLDS = range(5)
MODELS = ["resnet50", "convnext_tiny", "efficientnet_b0", "vit_b_16"]
SAMPLERS = ["weighted", "standard"]


def main() -> None:
    for sampler in SAMPLERS:
        for k in FOLDS:
            out_dir = CV / "runs" / sampler / f"fold{k}"
            todo = [m for m in MODELS if not (out_dir / m / "predictions.npz").exists()]
            if not todo:
                continue
            command = [
                sys.executable, "run_transfer_learning.py",
                "--dataset-dir", "dataset",
                "--split-manifest", str((CV / f"fold{k}.json").relative_to(ROOT)),
                "--output-dir", str(out_dir.relative_to(ROOT)),
                "--models", *todo,
                "--epochs", "25", "--batch-size", "32", "--num-workers", "0",
                "--seed", "42",
                "--threshold-metric", "youden_j",
                "--checkpoint-metric", "roc_auc",
            ]
            if sampler == "standard":
                command.append("--no-balanced-sampler")
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[cv] {sampler} fold={k} models={todo}", flush=True)
            with (out_dir / "train.log").open("a", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
