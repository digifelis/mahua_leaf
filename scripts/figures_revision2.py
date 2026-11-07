"""Manuscript figures for revision 2 (grouped split, 5 seeds). Run after analyze_revision2.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mahua_tl.evaluate import binary_confusion_matrix  # noqa: E402

A = ROOT / "artifacts/revision2/analysis"
OUT = ROOT / "artifacts/revision2/figures"
NAMES = {"resnet50": "ResNet50", "convnext_tiny": "ConvNeXt-Tiny", "efficientnet_b0": "EfficientNet-B0", "vit_b_16": "ViT-B/16"}
COND_LABEL = {"A0": "A0: standard, t=0.50", "A1": "A1: weighted, t=0.50", "A2": "A2: standard, Youden", "A3": "A3: weighted, Youden"}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def split_figure() -> None:
    s = json.loads((ROOT / "artifacts/revision2/split/split_audit.json").read_text())["new_split_summary"]
    names, labels = ["train", "val", "test"], ["Training", "Validation", "Test"]
    h = np.array([s[n]["healthy"] for n in names])
    u = np.array([s[n]["unhealthy"] for n in names])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(labels, h, label="Healthy", color="#2f7d59")
    ax.bar(labels, u, bottom=h, label="Unhealthy", color="#b85c38")
    for i in range(3):
        ax.text(i, h[i] + u[i] + 15, f"{h[i] + u[i]} ({h[i]} H / {u[i]} U)", ha="center", fontsize=8.5)
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, (h + u).max() * 1.12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Grouped (leakage-controlled) split: class distribution", fontsize=10)
    save(fig, "fig1_grouped_split_distribution")


def pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.5, 0.94, "Experimental workflow", ha="center", fontsize=15, fontweight="bold", color="#1F3A5F")
    panels = [
        ((0.06, 0.56), "1. Data and labeling",
         "2,632 annotated images (63 unannotated excluded)\nImage = Unhealthy if >=1 Unhealthy Leaf box\nPositive class = Unhealthy (1)", "#EEF7F0", "#2F7D59"),
        ((0.54, 0.56), "2. Grouped split",
         "Pre-augmented twins + perceptual near-duplicates\nmerged into 1,353 groups; groups never cross splits\nStratified 70/15/15 at group level",
         "#F7F9FC", "#446F9F"),
        ((0.06, 0.20), "3. Training (5 seeds per setting)",
         "4 ImageNet-pretrained backbones\nStandard vs. WeightedRandomSampler\nCheckpoint = best validation ROC-AUC",
         "#FFF8E6", "#B7791F"),
        ((0.54, 0.20), "4. Threshold and evaluation",
         "t = 0.50 vs. Youden J threshold (validation only)\nTest: balanced accuracy, Healthy recall/F1\nmean +/- SD, bootstrap 95% CI, Grad-CAM",
         "#F8EEF7", "#8A4F7D"),
    ]
    w, hgt = 0.40, 0.25
    for (x, y), title, body, face, edge in panels:
        ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=face, edgecolor=edge, lw=1.5))
        ax.text(x + w / 2, y + hgt - 0.05, title, ha="center", fontsize=10.5, fontweight="bold", color="#1F4D78")
        ax.text(x + w / 2, y + hgt / 2 - 0.035, body, ha="center", va="center", fontsize=8.4, linespacing=1.4)
    kw = dict(arrowstyle="-|>", lw=1.4, color="#4A5568")
    ax.annotate("", xy=(0.54, 0.685), xytext=(0.46, 0.685), arrowprops=kw)
    ax.plot([0.74, 0.74, 0.26], [0.56, 0.505, 0.505], color="#4A5568", lw=1.4)
    ax.annotate("", xy=(0.26, 0.45), xytext=(0.26, 0.505), arrowprops=kw)
    ax.annotate("", xy=(0.54, 0.325), xytext=(0.46, 0.325), arrowprops=kw)
    ax.text(0.5, 0.09, "The test split was used once per trained model, after checkpoint and threshold selection on the validation split.",
            ha="center", fontsize=9, color="#4A5568")
    save(fig, "fig2_workflow")


def condition_figure() -> None:
    df = pd.read_csv(A / "summary_mean_sd.csv")
    models = [m for m in NAMES if m in set(df.model)]
    conds = ["A0", "A1", "A2", "A3"]
    colors = ["#9aa9bd", "#446f9f", "#d9a76a", "#b85c38"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, key, ttl in zip(axes, ("balanced_accuracy", "healthy_recall"), ("Balanced accuracy", "Healthy recall")):
        x = np.arange(len(models))
        for k, c in enumerate(conds):
            sub = df[df.condition == c].set_index("model").reindex(models)
            ax.bar(x + (k - 1.5) * 0.2, sub[f"{key}_mean"], 0.2, yerr=sub[f"{key}_std"], capsize=2,
                   color=colors[k], label=COND_LABEL[c], error_kw=dict(lw=0.8))
        ax.set_xticks(x, [NAMES[m] for m in models])
        ax.set_title(f"{ttl} (test, mean +/- SD over 5 seeds)", fontsize=10)
        lo = np.nanmin(df[f"{key}_mean"] - df[f"{key}_std"])
        ax.set_ylim(max(0, lo - 0.05), 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].axhline(0.5, ls="--", c="grey", lw=0.8)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_condition_by_model")


def confusion_figure(model: str) -> None:
    runs = pd.read_csv(A / "per_run_metrics.csv")
    sel = runs[(runs.model == model) & (runs.condition == "A3")].sort_values("balanced_accuracy")
    row = sel.iloc[len(sel) // 2]
    npz = np.load(ROOT / f"artifacts/revision2/runs/weighted/seed{int(row.seed)}/{model}/predictions.npz")
    cm = binary_confusion_matrix(npz["test_labels"], (npz["test_probs"] >= row.threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Healthy", "Unhealthy"])
    ax.set_yticks([0, 1], ["Healthy", "Unhealthy"])
    ax.set(xlabel="Predicted", ylabel="True")
    ax.set_title(f"{NAMES[model]} (median seed {int(row.seed)}, t={row.threshold:.2f})", fontsize=9)
    fig.tight_layout()
    save(fig, "fig5_confusion_matrix")
    print("confusion", model, int(row.seed), cm.tolist())


def main() -> None:
    best = json.loads((A / "summary.json").read_text())["best_model_A3"]
    split_figure()
    pipeline_figure()
    condition_figure()
    confusion_figure(best)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
