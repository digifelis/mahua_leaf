"""Grouped 5-fold CV analysis: conditions A0-A3 per architecture.

Per fold, the Youden threshold is selected on that fold's validation subset. Reports mean ± SD over folds and
pooled out-of-fold metrics (every image tested once) with image-level bootstrap 95% CI of balanced accuracy.
Writes artifacts/revision2/cv/analysis/{cv_per_fold.csv, cv_summary.csv}.
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

RUNS = ROOT / "artifacts/revision2/cv/runs"
OUT = ROOT / "artifacts/revision2/cv/analysis"
KEYS = ["balanced_accuracy", "healthy_recall", "healthy_f1", "unhealthy_recall", "f1", "accuracy", "roc_auc"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_fold, pooled, oof = [], [], {}
    rng = np.random.default_rng(42)
    for model in NAMES:
        for cond, (sampler, rule) in CONDITIONS.items():
            probs, labels, preds = [], [], []
            for k in range(5):
                f = RUNS / sampler / f"fold{k}" / model / "predictions.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                t = 0.5 if rule == "fixed" else youden_threshold(d["val_probs"], d["val_labels"])
                m = compute_metrics_at_threshold(d["test_probs"], d["test_labels"], t)
                per_fold.append({"model": model, "condition": cond, "fold": k, "threshold": t, **{x: m[x] for x in KEYS}})
                probs.append(d["test_probs"])
                labels.append(d["test_labels"])
                preds.append((d["test_probs"] >= t).astype(int))
            if not probs:
                continue
            y, p = np.concatenate(labels), np.concatenate(preds)
            oof[(model, cond)] = (y, p)
            # pooled OOF metrics: thresholded per fold, so pass predictions as 0/1 "probabilities" with t=0.5
            m = compute_metrics_at_threshold(p.astype(float), y, 0.5)
            idx = rng.integers(0, len(y), size=(2000, len(y)))
            boot = np.array([fast_metrics(p[b].astype(float), y[b], 0.5)["balanced_accuracy"] for b in idx])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            pooled.append({"model": model, "condition": cond, "n_folds": len(probs), "n_images": len(y),
                           "pooled_balanced_accuracy": m["balanced_accuracy"], "pooled_ba_ci_lo": lo, "pooled_ba_ci_hi": hi,
                           "pooled_healthy_recall": m["healthy_recall"], "pooled_healthy_f1": m["healthy_f1"],
                           "pooled_unhealthy_recall": m["unhealthy_recall"], "pooled_accuracy": m["accuracy"]})
    # paired bootstrap on pooled OOF predictions (same image order in every model/condition: fold-wise test lists)
    ref_y = oof[("resnet50", "A3")][0]
    idx = rng.integers(0, len(ref_y), size=(2000, len(ref_y)))
    ba = lambda key, b: fast_metrics(oof[key][1][b].astype(float), ref_y[b], 0.5)["balanced_accuracy"]  # noqa: E731
    pairs = [(("resnet50", "A3"), (m, "A3")) for m in NAMES if m != "resnet50"]
    pairs += [((m, "A3"), (m, c)) for m in NAMES for c in ("A2", "A1")]
    rows = []
    for a, b in pairs:
        assert np.array_equal(oof[a][0], oof[b][0])
        diff = np.array([ba(a, i) - ba(b, i) for i in idx])
        obs = ba(a, np.arange(len(ref_y))) - ba(b, np.arange(len(ref_y)))
        p_two = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
        rows.append({"a": "/".join(a), "b": "/".join(b), "diff": obs, "ci_lo": np.percentile(diff, 2.5),
                     "ci_hi": np.percentile(diff, 97.5), "p": p_two})
    pd.DataFrame(rows).to_csv(OUT / "cv_paired.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    pf = pd.DataFrame(per_fold)
    pf.to_csv(OUT / "cv_per_fold.csv", index=False)
    agg = pf.groupby(["model", "condition"])[KEYS + ["threshold"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    summary = agg.reset_index().merge(pd.DataFrame(pooled), on=["model", "condition"])
    summary.to_csv(OUT / "cv_summary.csv", index=False)
    show = ["model", "condition", "n_folds", "balanced_accuracy_mean", "balanced_accuracy_std", "pooled_balanced_accuracy",
            "pooled_ba_ci_lo", "pooled_ba_ci_hi", "healthy_recall_mean", "healthy_f1_mean", "roc_auc_mean", "accuracy_mean"]
    print(summary[show].round(3).to_string())


if __name__ == "__main__":
    main()
