"""Grad-CAM examples (qualitative only) for the revision-2 grouped-split model.

Uses the run of the chosen architecture (weighted sampler, Youden threshold) whose test
balanced accuracy is the median across seeds. Target layer: last convolutional stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gradcam_utils import GradCAM, overlay_cam  # noqa: E402
from src.mahua_tl.data import build_transforms, discover_samples, load_split_manifest  # noqa: E402
from src.mahua_tl.models import build_model  # noqa: E402

LAYERS = {"convnext_tiny": ("features[-1]", lambda m: m.features[-1]),
          "resnet50": ("layer4", lambda m: m.layer4),
          "efficientnet_b0": ("features[-1]", lambda m: m.features[-1])}
NAMES = {"resnet50": "ResNet50", "convnext_tiny": "ConvNeXt-Tiny", "efficientnet_b0": "EfficientNet-B0"}
CATS = ["True Healthy", "True Unhealthy", "False alarm (Healthy -> Unhealthy)", "Missed Unhealthy (Unhealthy -> Healthy)"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convnext_tiny", choices=list(LAYERS))
    parser.add_argument("--per-category", type=int, default=2)
    args = parser.parse_args()

    runs = pd.read_csv(ROOT / "artifacts/revision2/analysis/per_run_metrics.csv")
    sel = runs[(runs.model == args.model) & (runs.condition == "A3")].sort_values("balanced_accuracy")
    row = sel.iloc[len(sel) // 2]
    seed, threshold = int(row.seed), float(row.threshold)
    ckpt = ROOT / f"artifacts/revision2/runs/weighted/seed{seed}/{args.model}/best.pt"
    probs = np.load(ckpt.with_name("predictions.npz"))["test_probs"]

    samples = load_split_manifest(ROOT / "artifacts/revision2/split/grouped_split.json", discover_samples(Path("dataset")))["test"]
    labels = np.array([s.label for s in samples])
    preds = (probs >= threshold).astype(int)
    cat_of = {(0, 0): CATS[0], (1, 1): CATS[1], (0, 1): CATS[2], (1, 0): CATS[3]}
    picks: dict[str, list[int]] = {c: [] for c in CATS}
    # most confident examples per category (deterministic)
    order = np.argsort(-np.abs(probs - threshold))
    for i in order:
        c = cat_of[(int(labels[i]), int(preds[i]))]
        if len(picks[c]) < args.per_category:
            picks[c].append(int(i))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
    model.eval()
    layer_name, layer_fn = LAYERS[args.model]
    cam = GradCAM(model, layer_fn(model))
    tfm = build_transforms(224)["test"]

    items = [(c, i) for c in CATS for i in picks[c]]
    fig, axes = plt.subplots(len(items) // 2, 4, figsize=(11, 2.9 * (len(items) // 2)))
    for k, (c, i) in enumerate(items):
        image = Image.open(samples[i].image_path).convert("RGB")
        cam_map, p = cam(tfm(image).unsqueeze(0).to(device), int(preds[i]))
        assert abs(p - probs[i]) < 1e-3, "probability mismatch with saved predictions"
        ax0, ax1 = axes[k // 2, (k % 2) * 2], axes[k // 2, (k % 2) * 2 + 1]
        ax0.imshow(image.resize((224, 224)))
        ax0.set_title(c, fontsize=8)
        ax1.imshow(overlay_cam(image, cam_map))
        ax1.set_title(f"Grad-CAM, P(Unhealthy)={p:.2f}", fontsize=8)
        ax0.axis("off")
        ax1.axis("off")
    cam.remove()
    fig.suptitle(f"{NAMES[args.model]} Grad-CAM (target layer: {layer_name}; seed {seed}; threshold {threshold:.2f})", fontsize=10)
    fig.tight_layout()
    out = ROOT / "artifacts/revision2/analysis/gradcam_examples.png"
    fig.savefig(out, dpi=300)
    print(out, {c: len(v) for c, v in picks.items()}, "seed", seed, "threshold", threshold)


if __name__ == "__main__":
    main()
