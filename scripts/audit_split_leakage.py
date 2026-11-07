"""Audit exact and near-duplicate images across train, validation, and test splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit image leakage between YOLO splits.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset_yolo"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data_audit/split_leakage_audit.json"),
    )
    parser.add_argument(
        "--max-hamming",
        type=int,
        default=2,
        help="Maximum 64-bit dHash Hamming distance for a near-duplicate candidate.",
    )
    parser.add_argument("--max-phash", type=int, default=6)
    parser.add_argument("--min-correlation", type=float, default=0.985)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    comparisons = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in comparisons.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def perceptual_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(np.float32(resized))
    low_frequency = coefficients[:8, :8]
    median = float(np.median(low_frequency.reshape(-1)[1:]))
    comparisons = low_frequency > median
    value = 0
    for bit in comparisons.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def normalized_embedding(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    embedding = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    embedding -= float(embedding.mean())
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding /= norm
    return embedding.reshape(-1)


def collect(root: Path) -> list[dict]:
    rows = []
    for split in ("train", "val", "test"):
        directory = root / split / "images"
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows.append(
                {
                    "split": split,
                    "path": str(path.resolve()),
                    "name": path.name.lower(),
                    "stem": path.stem.lower(),
                    "sha256": file_sha256(path),
                    "dhash": difference_hash(path),
                    "phash": perceptual_hash(path),
                    "embedding": normalized_embedding(path),
                }
            )
    return rows


def grouped_cross_split(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    findings = []
    for value, members in groups.items():
        splits = sorted({member["split"] for member in members})
        if len(splits) > 1:
            findings.append(
                {
                    "value": value,
                    "splits": splits,
                    "paths": [member["path"] for member in members],
                }
            )
    return findings


def near_duplicates(
    rows: list[dict],
    max_hamming: int,
    max_phash: int,
    min_correlation: float,
) -> list[dict]:
    findings = []
    split_order = {"train": 0, "val": 1, "test": 2}
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in split_order
    }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for split_a, split_b in pairs:
        for row_a in by_split[split_a]:
            for row_b in by_split[split_b]:
                distance = (row_a["dhash"] ^ row_b["dhash"]).bit_count()
                if distance <= max_hamming and row_a["sha256"] != row_b["sha256"]:
                    phash_distance = (row_a["phash"] ^ row_b["phash"]).bit_count()
                    correlation = float(np.dot(row_a["embedding"], row_b["embedding"]))
                    findings.append(
                        {
                            "split_a": split_a,
                            "path_a": row_a["path"],
                            "split_b": split_b,
                            "path_b": row_b["path"],
                            "hamming_distance": distance,
                            "phash_distance": phash_distance,
                            "correlation": correlation,
                            "high_confidence": (
                                phash_distance <= max_phash
                                and correlation >= min_correlation
                            ),
                        }
                    )
    return sorted(
        findings,
        key=lambda item: (
            item["hamming_distance"],
            item["phash_distance"],
            -item["correlation"],
            split_order[item["split_a"]],
            item["path_a"],
            item["path_b"],
        ),
    )


def main() -> None:
    args = parse_args()
    rows = collect(args.data_root)
    exact_hash = grouped_cross_split(rows, "sha256")
    exact_name = grouped_cross_split(rows, "name")
    same_stem = grouped_cross_split(rows, "stem")
    perceptual = near_duplicates(
        rows,
        args.max_hamming,
        args.max_phash,
        args.min_correlation,
    )
    high_confidence = [item for item in perceptual if item["high_confidence"]]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root.resolve()),
        "images": len(rows),
        "dhash_bits": 64,
        "phash_bits": 64,
        "near_duplicate_max_hamming": args.max_hamming,
        "near_duplicate_max_phash": args.max_phash,
        "near_duplicate_min_correlation": args.min_correlation,
        "cross_split_exact_sha256_groups": exact_hash,
        "cross_split_exact_filename_groups": exact_name,
        "cross_split_same_stem_groups": same_stem,
        "cross_split_perceptual_candidates": perceptual,
        "counts": {
            "exact_sha256": len(exact_hash),
            "exact_filename": len(exact_name),
            "same_stem": len(same_stem),
            "perceptual_candidates": len(perceptual),
            "high_confidence_perceptual_candidates": len(high_confidence),
        },
        "passed_strict": not exact_hash and not exact_name and not same_stem and not high_confidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "split_a",
            "path_a",
            "split_b",
            "path_b",
            "hamming_distance",
            "phash_distance",
            "correlation",
            "high_confidence",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(perceptual)
    print(json.dumps({"counts": report["counts"], "passed_strict": report["passed_strict"]}, indent=2))


if __name__ == "__main__":
    main()
