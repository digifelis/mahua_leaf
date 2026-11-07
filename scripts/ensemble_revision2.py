"""Seed ensemble on the fixed grouped split: average P(Unhealthy) of the 5 seeds per (model, sampler).

Threshold for Youden conditions is selected on the averaged validation probabilities only.
Writes artifacts/revision2/analysis/ensemble_metrics.csv (+ image-level bootstrap 95% CI of balanced accuracy).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_revision2 import CONDITIONS, NAMES, fast_metrics, youden_threshold  # noqa: E402
from src.mahua_tl.evaluate import compute_metrics_at_threshold  # noqa: E402

RUNS = ROOT / "artifacts/revision2/runs"
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    rng = np.random.default_rng(42)
    rows, idx = [], None
    for model in NAMES:
        for cond, (sampler, rule) in CONDITIONS.items():
            data = [np.load(RUNS / sampler / f"seed{s}" / model / "predictions.npz") for s in SEEDS]
            yv, yt = data[0]["val_labels"], data[0]["test_labels"]
            assert all(np.array_equal(d["test_labels"], yt) and np.array_equal(d["val_labels"], yv) for d in data)
            pv = np.mean([d["val_probs"] for d in data], axis=0)
            pt = np.mean([d["test_probs"] for d in data], axis=0)
            t = 0.5 if rule == "fixed" else youden_threshold(pv, yv)
            m = compute_metrics_at_threshold(pt, yt, t)
            if idx is None:
                idx = rng.integers(0, len(yt), size=(2000, len(yt)))
            boot = np.array([fast_metrics(pt[b], yt[b], t)["balanced_accuracy"] for b in idx])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({"model": model, "condition": cond, "threshold": t, **m, "bal_acc_ci_lo": lo, "bal_acc_ci_hi": hi})
    df = pd.DataFrame(rows)
    out = ROOT / "artifacts/revision2/analysis/ensemble_metrics.csv"
    df.to_csv(out, index=False)
    cols = ["model", "condition", "threshold", "balanced_accuracy", "bal_acc_ci_lo", "bal_acc_ci_hi", "healthy_recall",
            "healthy_f1", "unhealthy_recall", "accuracy", "roc_auc"]
    print(df[cols].round(3).to_string())


if __name__ == "__main__":
    main()
