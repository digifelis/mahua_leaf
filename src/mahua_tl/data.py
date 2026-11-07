from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEALTHY_NAMES = {"healthy leaf", "healthy", "healthyleaf"}


@dataclass(frozen=True)
class MahuaSample:
    image_path: Path
    annotation_path: Path
    label: int


def _annotation_to_binary_label(annotation_path: Path) -> int:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    boxes = payload.get("boxes", [])
    classes = [str(box.get("class", "")).strip().lower() for box in boxes]
    if not classes:
        raise ValueError(f"No labels found in {annotation_path}")
    return int(any(label.replace(" ", "") not in HEALTHY_NAMES for label in classes))


def discover_samples(dataset_dir: Path) -> list[MahuaSample]:
    samples: list[MahuaSample] = []
    for annotation_path in sorted(dataset_dir.glob("*/annotations/*.json")):
        split_dir = annotation_path.parent.parent
        image_dir = split_dir / "images"
        image_path = next(
            (image_dir / f"{annotation_path.stem}{ext}" for ext in IMAGE_EXTS if (image_dir / f"{annotation_path.stem}{ext}").exists()),
            None,
        )
        if image_path is None:
            raise FileNotFoundError(f"No image found for annotation {annotation_path}")
        samples.append(
            MahuaSample(
                image_path=image_path,
                annotation_path=annotation_path,
                label=_annotation_to_binary_label(annotation_path),
            )
        )
    if not samples:
        raise FileNotFoundError(f"No annotation JSON files found under {dataset_dir}")
    return samples


def stratified_split(
    samples: list[MahuaSample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[MahuaSample]]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    rng = random.Random(seed)
    by_label: dict[int, list[MahuaSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    for label_samples in by_label.values():
        rng.shuffle(label_samples)

    train_target = int(len(samples) * train_ratio)
    train_counts = _allocate_stratified_counts(by_label, train_target)
    train_samples: list[MahuaSample] = []
    holdout_by_label: dict[int, list[MahuaSample]] = {}
    for label, label_samples in by_label.items():
        count = train_counts[label]
        train_samples.extend(label_samples[:count])
        holdout_by_label[label] = label_samples[count:]

    holdout_total = sum(len(items) for items in holdout_by_label.values())
    val_target = int(round(holdout_total * (val_ratio / (val_ratio + test_ratio))))
    val_counts = _allocate_stratified_counts(holdout_by_label, val_target)
    val_samples: list[MahuaSample] = []
    test_samples: list[MahuaSample] = []
    for label, label_samples in holdout_by_label.items():
        count = val_counts[label]
        val_samples.extend(label_samples[:count])
        test_samples.extend(label_samples[count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return {"train": train_samples, "val": val_samples, "test": test_samples}


def load_split_manifest(manifest_path: Path, samples: list[MahuaSample]) -> dict[str, list[MahuaSample]]:
    """Fixed (e.g. grouped, leakage-free) split saved by scripts/build_grouped_binary_split.py."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_path = {sample.image_path.as_posix(): sample for sample in samples}
    splits = {name: [by_path[path] for path in manifest[name]] for name in ("train", "val", "test")}
    if sum(len(items) for items in splits.values()) != len(samples):
        raise ValueError("Split manifest does not cover every annotated sample exactly once.")
    return splits


def _allocate_stratified_counts(by_label: dict[int, list[MahuaSample]], target_total: int) -> dict[int, int]:
    total = sum(len(items) for items in by_label.values())
    raw = {label: target_total * len(items) / total for label, items in by_label.items()}
    counts = {label: int(value) for label, value in raw.items()}
    remaining = target_total - sum(counts.values())
    remainders = sorted(
        raw,
        key=lambda label: (raw[label] - counts[label], len(by_label[label])),
        reverse=True,
    )
    for label in remainders[:remaining]:
        counts[label] += 1
    return counts


class MahuaBinaryDataset(Dataset):
    def __init__(self, samples: list[MahuaSample], transform=None, cache_size: int | None = None) -> None:
        self.samples = samples
        self.transform = transform
        # Decode+resize once and keep in RAM; the transform's own Resize is then a no-op,
        # so augmentation output is identical. Only effective with num_workers=0.
        self.cache_resize = transforms.Resize((cache_size, cache_size)) if cache_size else None
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.cache_resize is None:
            image = Image.open(sample.image_path).convert("RGB")
        else:
            if index not in self._cache:
                self._cache[index] = self.cache_resize(Image.open(sample.image_path).convert("RGB"))
            image = self._cache[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, float(sample.label)


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    return {"train": train_tfms, "val": eval_tfms, "test": eval_tfms}


def build_dataloaders(
    splits: dict[str, list[MahuaSample]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    balanced_sampler: bool = True,
) -> dict[str, DataLoader]:
    tfms = build_transforms(image_size)
    dataloaders: dict[str, DataLoader] = {}
    for split, samples in splits.items():
        sampler = None
        shuffle = split == "train"
        if split == "train" and balanced_sampler:
            labels = [sample.label for sample in samples]
            class_counts = {label: labels.count(label) for label in set(labels)}
            weights = [1.0 / class_counts[label] for label in labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False
        dataloaders[split] = DataLoader(
            MahuaBinaryDataset(samples, tfms[split], cache_size=image_size if num_workers == 0 else None),
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    return dataloaders


def split_summary(splits: dict[str, list[MahuaSample]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split, samples in splits.items():
        healthy = sum(sample.label == 0 for sample in samples)
        unhealthy = sum(sample.label == 1 for sample in samples)
        summary[split] = {"total": len(samples), "healthy": healthy, "unhealthy": unhealthy}
    return summary
