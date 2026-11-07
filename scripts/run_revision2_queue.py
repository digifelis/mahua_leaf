"""Revision 2 training queue: 4 architectures x {weighted, standard} sampler x 5 seeds on the grouped split.

Checkpoints are selected by validation ROC-AUC (threshold-free), so the fixed-0.50 and
Youden-J conditions are evaluated post hoc on the same trained model. Resumable: a
(sampler, seed, model) whose predictions.npz exists is skipped.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/revision2/runs"
MANIFEST = ROOT / "artifacts/revision2/split/grouped_split.json"
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["resnet50", "convnext_tiny", "efficientnet_b0", "vit_b_16"]
SAMPLERS = ["weighted", "standard"]


def main() -> None:
    for sampler in SAMPLERS:
        for seed in SEEDS:
            out_dir = OUT / sampler / f"seed{seed}"
            todo = [m for m in MODELS if not (out_dir / m / "predictions.npz").exists()]
            if not todo:
                continue
            command = [
                sys.executable, "run_transfer_learning.py",
                "--dataset-dir", "dataset",
                "--split-manifest", str(MANIFEST.relative_to(ROOT)),
                "--output-dir", str(out_dir.relative_to(ROOT)),
                "--models", *todo,
                "--epochs", "25", "--batch-size", "32", "--num-workers", "0",
                "--seed", str(seed),
                "--threshold-metric", "youden_j",
                "--checkpoint-metric", "roc_auc",
            ]
            if sampler == "standard":
                command.append("--no-balanced-sampler")
            log = out_dir / f"train_{'_'.join(todo)}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[queue] {sampler} seed={seed} models={todo}", flush=True)
            with log.open("a", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
