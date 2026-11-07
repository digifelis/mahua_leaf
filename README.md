# mahua_leaf

Leakage-controlled, multi-seed evaluation of transfer learning for **binary health classification of Mahua leaf
images** (Healthy vs. Unhealthy) under class imbalance.

The repository contains:

* a duplicate / near-duplicate audit of the public dataset and a **grouped train/validation/test split** in which no
  group of related images crosses subsets;
* training code for four ImageNet-pretrained backbones (ResNet50, ConvNeXt-Tiny, EfficientNet-B0, ViT-B/16);
* a 2 × 2 design of **standard vs. class-balanced sampling** (`WeightedRandomSampler`) and a **fixed 0.50 vs.
  validation-selected (Youden J) decision threshold**, with five random seeds per setting;
* **grouped, class-stratified 5-fold cross-validation**, a **five-seed ensemble**, and a control experiment on an
  image-level random split;
* statistical analysis (mean ± SD, image-level bootstrap 95% CIs, paired bootstrap tests, threshold sensitivity, ROC),
  figures and Grad-CAM examples;
* all split files, the test/validation **predictions of every trained model**, and the resulting metric tables, so
  that every number can be re-computed without a GPU.

## Main results

Positive class = Unhealthy. Majority-class baseline on the grouped test split (n = 398; 65 Healthy, 333 Unhealthy):
accuracy 0.837, balanced accuracy 0.500, Healthy recall 0.000.

Balanced accuracy with class-balanced sampling and Youden J threshold (condition A3):

| Backbone | Grouped split, 5 seeds (mean ± SD) [95% CI] | Five-seed ensemble [95% CI] | Grouped 5-fold CV (mean ± SD over folds) |
|---|---|---|---|
| ResNet50 | 0.742 ± 0.042 [0.699, 0.784] | 0.799 [0.743, 0.848] | 0.770 ± 0.020 |
| ConvNeXt-Tiny | 0.725 ± 0.041 [0.676, 0.773] | 0.767 [0.706, 0.827] | 0.761 ± 0.032 |
| EfficientNet-B0 | 0.718 ± 0.037 [0.668, 0.766] | 0.722 [0.657, 0.782] | 0.730 ± 0.027 |
| ViT-B/16 | 0.617 ± 0.023 [0.569, 0.669] | 0.620 [0.560, 0.684] | 0.717 ± 0.028 |

Leakage control matters: with the same pipeline, an image-level random split (where augmented copies and
near-duplicates of test images end up in training) gives 0.886 ± 0.041 (ResNet50) and 0.884 ± 0.041 (ConvNeXt-Tiny).
Full tables are in `artifacts/revision2/analysis/` and `artifacts/revision2/cv/analysis/`.

## Repository layout

```
run_transfer_learning.py            training entry point (one or more backbones, one seed)
src/mahua_tl/                       data loading, models, training loop, metrics
scripts/
  audit_split_leakage.py            duplicate / near-duplicate audit helpers
  build_grouped_binary_split.py     audit of the image-level split + grouped split
  run_revision2_queue.py            4 backbones x 2 samplers x 5 seeds on the grouped split
  analyze_revision2.py              conditions A0-A3, mean +/- SD, bootstrap CIs, paired tests, threshold sensitivity, ROC
  ensemble_revision2.py             five-seed ensemble
  build_grouped_cv_folds.py         grouped, class-stratified 5-fold CV manifests
  run_cv_queue.py                   4 backbones x 2 samplers x 5 folds
  analyze_cv.py                     per-fold and pooled out-of-fold CV metrics, paired bootstrap
  figures_revision2.py              figures
  gradcam_revision2.py, gradcam_utils.py   Grad-CAM examples
splits/                             grouped_split.json, split_audit.json, cv/fold{0..4}.json
artifacts/revision2/
  split/                            split files at the path the scripts read
  runs/{standard,weighted}/seed{42..46}/<backbone>/predictions.npz
  cv/fold{0..4}.json, cv/runs/{standard,weighted}/fold{0..4}/<backbone>/predictions.npz
  runs_imagelevel_control/weighted/seed{42..46}/<backbone>/predictions.npz
  analysis/, cv/analysis/           metric tables (CSV/JSON) and ROC / Grad-CAM figures
  figures/                          split, workflow, condition-by-backbone and confusion-matrix figures
```

Each `predictions.npz` holds `val_probs`, `val_labels`, `test_probs` and `test_labels` (probability of Unhealthy),
in the order of the `val` / `test` lists of the corresponding split file. Model checkpoints are not included.

## Data

Annotated Mahua Leaf Dataset for Disease Detection and Precision Agriculture, Version 1, Mendeley Data,
https://doi.org/10.17632/6xjn6p73h7.1. The dataset is not redistributed here. Extract it to `./dataset` so that

```
dataset/{train,val,test}/images/*.jpg
dataset/{train,val,test}/annotations/*.json
```

The original folder split is ignored; all annotated images are pooled and re-split.

* **Label rule:** an image is **Unhealthy (1, positive class)** if it contains at least one `Unhealthy Leaf` box,
  otherwise **Healthy (0)**. Images without an annotation file (63) are excluded, leaving 2,632 images
  (424 Healthy, 2,208 Unhealthy).
* **Grouping:** the dataset ships 1,083 pre-augmented copies (`img_XXXXXX_aug0.jpg`). Each original is grouped with
  its copies, and groups linked by a near-duplicate pair (difference hash, perceptual hash + pixel correlation) are
  merged, giving 1,353 groups. Whole groups are assigned to train/val/test (≈70/15/15%, class-stratified).
* Paths in the split files are relative to the repository root, so run all commands from the root.

## Environment

Python 3.11, PyTorch 2.6.0 (CUDA 12.4), torchvision 0.21.0:

```bash
pip install -r requirements.txt
```

Experiments were run on a single NVIDIA GeForce RTX 4060 Laptop GPU (8 GB); one training run takes about 10 minutes.

## Reproduce the analyses from the included predictions (no GPU, no dataset needed)

```bash
python scripts/analyze_revision2.py    # grouped split: A0-A3, CIs, paired tests, threshold sensitivity, ROC
python scripts/ensemble_revision2.py   # five-seed ensemble
python scripts/analyze_cv.py           # grouped 5-fold CV
python scripts/figures_revision2.py    # figures
```


## Full reproduction (training)

```bash
# 1) audit of the image-level split + grouped split (regenerates artifacts/revision2/split/ identically)
python scripts/build_grouped_binary_split.py --dataset-dir dataset --output artifacts/revision2/split/grouped_split.json

# 2) 4 backbones x {standard, weighted} sampler x 5 seeds (40 runs; resumable)
python scripts/run_revision2_queue.py

# 3) grouped 5-fold CV: folds (identical to splits/cv/) and 40 runs (seed 42; resumable)
python scripts/build_grouped_cv_folds.py
python scripts/run_cv_queue.py

# 4) analyses and figures (see above); Grad-CAM needs the checkpoints from step 2
python scripts/gradcam_revision2.py --model resnet50
```

Control experiment on an image-level random split (leakage present):

```bash
for s in 42 43 44 45 46; do
  python run_transfer_learning.py --dataset-dir dataset --output-dir artifacts/revision2/runs_imagelevel_control/weighted/seed$s \
    --models resnet50 convnext_tiny --epochs 25 --batch-size 32 --num-workers 0 --seed $s \
    --threshold-metric youden_j --checkpoint-metric roc_auc
done
```

A single run on the grouped split:

```bash
python run_transfer_learning.py --dataset-dir dataset --split-manifest splits/grouped_split.json \
  --models convnext_tiny --epochs 25 --batch-size 32 --num-workers 0 --seed 42 \
  --threshold-metric youden_j --checkpoint-metric roc_auc
```

## Training and evaluation settings

224 × 224 input, ImageNet normalisation, flips / rotation / colour jitter (training only), all layers fine-tuned,
single-logit head with `BCEWithLogitsLoss`, AdamW (lr 1e-4, weight decay 1e-3), batch size 32, up to 25 epochs,
`ReduceLROnPlateau` (factor 0.5, patience 3) and early stopping (patience 8) on validation ROC-AUC; the checkpoint with
the best validation ROC-AUC is evaluated once on the test split. The Youden J threshold is searched on the validation
split over 0.05–0.95 (step 0.01).

Conditions: A0 = standard sampling, threshold 0.50; A1 = weighted sampling, 0.50; A2 = standard sampling, Youden J;
A3 = weighted sampling, Youden J. Confidence intervals: 2,000 image-level bootstrap resamples.

## License

MIT (see `LICENSE`). Citation metadata: `CITATION.cff`.
