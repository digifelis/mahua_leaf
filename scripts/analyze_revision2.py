"""Post-hoc analysis of the revision-2 runs (grouped split, 5 seeds).

Reads artifacts/revision2/runs/<sampler>/seed<k>/<model>/predictions.npz and writes
artifacts/revision2/analysis/*: per-run metrics, mean +/- SD tables for the 2x2 design
(sampler x threshold rule) per architecture, image-level bootstrap 95% CIs, paired
bootstrap differences against the best model, threshold sensitivity and ROC figures.
Positive class = Unhealthy (label 1); Healthy (label 0) is the minority class.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mahua_tl.evaluate import compute_metrics_at_threshold  # noqa: E402

RUNS = ROOT / "artifacts/revision2/runs"
OUT = ROOT / "artifacts/revision2/analysis"
GRID = np.linspace(0.05, 0.95, 91)
CONDITIONS = {"A0": ("standard", "fixed"), "A1": ("weighted", "fixed"), "A2": ("standard", "youden"), "A3": ("weighted", "youden")}
KEYS = ["balanced_accuracy", "healthy_recall", "healthy_precision", "healthy_f1", "unhealthy_recall", "f1", "accuracy", "roc_auc", "youden_j"]
NAMES = {"resnet50": "ResNet50", "convnext_tiny": "ConvNeXt-Tiny", "efficientnet_b0": "EfficientNet-B0", "vit_b_16": "ViT-B/16"}
N_BOOT = 2000


def youden_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    # same grid and tie rule (first maximum, starting from 0.50) as src.mahua_tl.evaluate.find_best_threshold
    best_t, best_j = 0.5, compute_metrics_at_threshold(probs, labels, 0.5)["youden_j"]
    for t in GRID:
        j = compute_metrics_at_threshold(probs, labels, float(t))["youden_j"]
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


def load_runs() -> list[dict]:
    runs = []
    for npz in sorted(RUNS.glob("*/seed*/*/predictions.npz")):
        model, seed, sampler = npz.parent.name, int(npz.parent.parent.name[4:]), npz.parent.parent.parent.name
        data = np.load(npz)
        runs.append({"sampler": sampler, "seed": seed, "model": model, **{k: data[k] for k in data.files},
                     "youden_t": youden_threshold(data["val_probs"], data["val_labels"])})
    return runs


def fast_metrics(probs: np.ndarray, labels: np.ndarray, t: float) -> dict[str, float]:
    pred = probs >= t
    pos, neg = labels == 1, labels == 0
    tp, fn = np.sum(pred & pos), np.sum(~pred & pos)
    tn, fp = np.sum(~pred & neg), np.sum(pred & neg)
    hr, ur = tn / max(tn + fp, 1), tp / max(tp + fn, 1)
    hp = tn / max(tn + fn, 1)
    return {"balanced_accuracy": (hr + ur) / 2, "healthy_recall": hr,
            "healthy_f1": 2 * hp * hr / max(hp + hr, 1e-12), "accuracy": (tp + tn) / len(labels)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("no predictions found")
    labels = runs[0]["test_labels"]
    assert all(np.array_equal(r["test_labels"], labels) for r in runs), "test order must be identical across runs"

    rows = []
    for r in runs:
        for cond, (sampler, rule) in CONDITIONS.items():
            if r["sampler"] != sampler:
                continue
            t = 0.5 if rule == "fixed" else r["youden_t"]
            m = compute_metrics_at_threshold(r["test_probs"], labels, t)
            rows.append({"model": r["model"], "condition": cond, "seed": r["seed"], "threshold": t, **m})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(OUT / "per_run_metrics.csv", index=False)

    agg = per_run.groupby(["model", "condition"])[KEYS + ["threshold"]].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT / "summary_mean_sd.csv", index=False)

    n_h, n_u = int(np.sum(labels == 0)), int(np.sum(labels == 1))
    baseline = {"test_healthy": n_h, "test_unhealthy": n_u, "majority_accuracy": n_u / len(labels),
                "majority_balanced_accuracy": 0.5, "majority_healthy_recall": 0.0}

    # image-level bootstrap: resample test images, recompute each seed's metric, average over seeds
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(labels), size=(N_BOOT, len(labels)))
    boot = {}
    for (model, cond), grp in per_run.groupby(["model", "condition"]):
        sampler = CONDITIONS[cond][0]
        rs = [r for r in runs if r["model"] == model and r["sampler"] == sampler]
        ts = {r["seed"]: float(grp.loc[grp.seed == r["seed"], "threshold"].iloc[0]) for r in rs}
        samples = np.array([[np.mean([fast_metrics(r["test_probs"][b], labels[b], ts[r["seed"]])[k] for r in rs])
                             for k in ("balanced_accuracy", "healthy_recall", "healthy_f1", "accuracy")] for b in idx])
        boot[(model, cond)] = samples
    ci_rows = []
    for (model, cond), s in boot.items():
        lo, hi = np.percentile(s, [2.5, 97.5], axis=0)
        ci_rows.append({"model": model, "condition": cond,
                        **{f"{k}_ci_lo": lo[i] for i, k in enumerate(("balanced_accuracy", "healthy_recall", "healthy_f1", "accuracy"))},
                        **{f"{k}_ci_hi": hi[i] for i, k in enumerate(("balanced_accuracy", "healthy_recall", "healthy_f1", "accuracy"))}})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT / "bootstrap_ci.csv", index=False)

    # paired differences in balanced accuracy (main condition A3 and every model's best condition)
    main_cond = "A3"
    a3 = agg[agg.condition == main_cond].sort_values("balanced_accuracy_mean", ascending=False)
    best = a3.iloc[0]["model"]
    diffs = []
    for model in a3.model:
        if model == best or (model, main_cond) not in boot:
            continue
        d = boot[(best, main_cond)][:, 0] - boot[(model, main_cond)][:, 0]
        diffs.append({"reference": best, "model": model, "condition": main_cond, "delta_bal_acc_mean": float(d.mean()),
                      "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
                      "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))})
    # sampler and threshold effects within each model
    for model in agg.model.unique():
        for a, b in (("A2", "A0"), ("A1", "A0"), ("A3", "A2"), ("A3", "A1")):
            if (model, a) in boot and (model, b) in boot:
                d = boot[(model, a)][:, 0] - boot[(model, b)][:, 0]
                diffs.append({"reference": f"{model}:{a}", "model": f"{model}:{b}", "condition": f"{a}-{b}",
                              "delta_bal_acc_mean": float(d.mean()), "ci_lo": float(np.percentile(d, 2.5)),
                              "ci_hi": float(np.percentile(d, 97.5)),
                              "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))})
    pd.DataFrame(diffs).to_csv(OUT / "paired_bootstrap_differences.csv", index=False)

    # threshold sensitivity around each run's Youden threshold (main condition)
    sens = []
    for r in runs:
        if r["sampler"] != "weighted":
            continue
        for delta in (-0.10, -0.05, 0.0, 0.05, 0.10):
            t = float(np.clip(r["youden_t"] + delta, 0.01, 0.99))
            m = compute_metrics_at_threshold(r["test_probs"], labels, t)
            sens.append({"model": r["model"], "seed": r["seed"], "delta": delta, "threshold": t,
                         **{k: m[k] for k in ("balanced_accuracy", "healthy_recall", "unhealthy_recall", "healthy_f1")}})
    sens_df = pd.DataFrame(sens)
    sens_df.groupby(["model", "delta"])[["threshold", "balanced_accuracy", "healthy_recall", "unhealthy_recall", "healthy_f1"]] \
        .agg(["mean", "std"]).to_csv(OUT / "threshold_sensitivity.csv")

    # ROC (Healthy-vs-Unhealthy, positive = Unhealthy) with the validation-selected Youden point, best model
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for r in sorted((r for r in runs if r["model"] == best and r["sampler"] == "weighted"), key=lambda r: r["seed"]):
        for ax, split in zip(axes, ("val", "test")):
            p, y = r[f"{split}_probs"], r[f"{split}_labels"]
            ts = np.unique(np.concatenate([[1.01], np.sort(p)[::-1], [-0.01]]))[::-1]
            tpr = [np.mean(p[y == 1] >= t) for t in ts]
            fpr = [np.mean(p[y == 0] >= t) for t in ts]
            line, = ax.plot(fpr, tpr, lw=1.2, label=f"seed {r['seed']}")
            pm = compute_metrics_at_threshold(p, y, r["youden_t"])
            ax.plot(1 - pm["healthy_recall"], pm["unhealthy_recall"], "o", color=line.get_color(), ms=6, mec="black")
    for ax, split in zip(axes, ("Validation", "Test")):
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
        ax.set(xlabel="1 - Healthy recall (false-positive rate)", ylabel="Unhealthy recall (true-positive rate)",
               title=f"{NAMES.get(best, best)} - {split}", xlim=(0, 1), ylim=(0, 1.01))
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("ROC curves (positive class = Unhealthy); dots = threshold selected on validation by Youden J", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "roc_best_model.png", dpi=300)
    plt.close(fig)

    summary = {"baseline": baseline, "n_runs": len(runs), "best_model_A3": best,
               "youden_thresholds": {m: sorted(r["youden_t"] for r in runs if r["model"] == m and r["sampler"] == "weighted")
                                     for m in NAMES}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    show = agg[["model", "condition"] + [f"{k}_mean" for k in ("balanced_accuracy", "healthy_recall", "healthy_f1", "accuracy", "roc_auc", "threshold")]
               + [f"{k}_std" for k in ("balanced_accuracy", "healthy_f1")] + ["balanced_accuracy_count"]]
    print(show.round(4).to_string(index=False))
    print(ci.round(4).to_string(index=False))
    print(pd.read_csv(OUT / "paired_bootstrap_differences.csv").round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
