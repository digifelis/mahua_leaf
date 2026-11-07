"""Leakage audit of the original image-level binary split and construction of a grouped split.

Groups = same base stem (img_X and its pre-augmented copy img_X_aug0) united with
perceptual near-duplicates (dHash/pHash/pixel correlation). Whole groups are assigned
to train/val/test with label stratification, so no group crosses a split.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_split_leakage import difference_hash, normalized_embedding, perceptual_hash  # noqa: E402
from src.mahua_tl.data import discover_samples, stratified_split, split_summary  # noqa: E402

AUG_RE = re.compile(r"_aug\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/revision2/split/grouped_split.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-dhash", type=int, default=2)
    parser.add_argument("--max-phash", type=int, default=6)
    parser.add_argument("--min-correlation", type=float, default=0.985)
    return parser.parse_args()


def base_stem(path: Path) -> str:
    return AUG_RE.sub("", path.stem)


def popcount_matrix(values: np.ndarray) -> np.ndarray:
    xor = values[:, None] ^ values[None, :]
    return np.unpackbits(xor.view(np.uint8).reshape(len(values), len(values), 8), axis=2).sum(axis=2)


def near_duplicate_pairs(paths: list[Path], max_dhash: int, max_phash: int, min_corr: float) -> list[tuple[int, int]]:
    dh = np.array([difference_hash(p) for p in paths], dtype=np.uint64)
    ph = np.array([perceptual_hash(p) for p in paths], dtype=np.uint64)
    emb = np.stack([normalized_embedding(p) for p in paths])
    d_dist = popcount_matrix(dh)
    p_dist = popcount_matrix(ph)
    corr = emb @ emb.T
    # conservative for leakage: either a near-identical dHash or a pHash+correlation match
    hit = (d_dist <= max_dhash) | ((p_dist <= max_phash) & (corr >= min_corr))
    i, j = np.where(np.triu(hit, k=1))
    return list(zip(i.tolist(), j.tolist()))


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def grouped_stratified_split(groups: list[list[int]], labels: list[int], seed: int) -> dict[str, list[int]]:
    """Assign whole groups to splits, greedily filling 70/15/15 targets within each label stratum."""
    rng = random.Random(seed)
    group_label = [max(labels[i] for i in g) for g in groups]  # any unhealthy member -> unhealthy stratum
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for label in (0, 1):
        idx = [k for k, gl in enumerate(group_label) if gl == label]
        rng.shuffle(idx)
        total = sum(len(groups[k]) for k in idx)
        targets = {"train": 0.70 * total, "val": 0.15 * total, "test": 0.15 * total}
        filled = {name: 0 for name in out}
        for k in idx:
            # split with the largest remaining deficit relative to its target
            name = max(out, key=lambda s: (targets[s] - filled[s]) / targets[s])
            out[name].extend(groups[k])
            filled[name] += len(groups[k])
    return out


def cross_split_leaks(split_of: dict[int, str], group_of: list[int]) -> dict[str, int]:
    members: dict[int, set[str]] = {}
    for i, g in enumerate(group_of):
        members.setdefault(g, set()).add(split_of[i])
    leaking_groups = [g for g, s in members.items() if len(s) > 1]
    leaking_test = sum(1 for i, g in enumerate(group_of) if split_of[i] == "test" and len(members[g]) > 1)
    leaking_val = sum(1 for i, g in enumerate(group_of) if split_of[i] == "val" and len(members[g]) > 1)
    return {"groups_crossing_splits": len(leaking_groups), "test_images_with_partner_elsewhere": leaking_test,
            "val_images_with_partner_elsewhere": leaking_val}


def main() -> None:
    args = parse_args()
    all_images = sorted(p for p in args.dataset_dir.glob("*/images/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    samples = discover_samples(args.dataset_dir)
    paths = [s.image_path for s in samples]
    labels = [s.label for s in samples]
    index = {p: i for i, p in enumerate(paths)}

    uf = UnionFind(len(paths))
    by_stem: dict[str, int] = {}
    for i, p in enumerate(paths):
        stem = base_stem(p)
        if stem in by_stem:
            uf.union(i, by_stem[stem])
        else:
            by_stem[stem] = i
    stem_groups = len({uf.find(i) for i in range(len(paths))})
    pairs = near_duplicate_pairs(paths, args.max_dhash, args.max_phash, args.min_correlation)
    perceptual_only = [(a, b) for a, b in pairs if base_stem(paths[a]) != base_stem(paths[b])]
    for a, b in pairs:
        uf.union(a, b)
    root_to_group: dict[int, int] = {}
    group_of = [root_to_group.setdefault(uf.find(i), len(root_to_group)) for i in range(len(paths))]
    groups: list[list[int]] = [[] for _ in root_to_group]
    for i, g in enumerate(group_of):
        groups[g].append(i)
    mixed = [g for g in groups if len({labels[i] for i in g}) > 1]

    # leakage of the original image-level random split (seed 42)
    old = stratified_split(samples, seed=42)
    old_split_of = {index[s.image_path]: name for name, items in old.items() for s in items}
    old_stem_only = {}
    for i, p in enumerate(paths):
        old_stem_only.setdefault(base_stem(p), set()).add(old_split_of[i])
    old_aug_test = sum(1 for i, p in enumerate(paths) if old_split_of[i] == "test" and len(old_stem_only[base_stem(p)]) > 1)

    new = grouped_stratified_split(groups, labels, args.seed)
    new_split_of = {i: name for name, items in new.items() for i in items}
    assert len(new_split_of) == len(paths)
    new_leaks = cross_split_leaks(new_split_of, group_of)
    assert new_leaks["groups_crossing_splits"] == 0

    summary = {name: {"total": len(items), "healthy": sum(labels[i] == 0 for i in items),
                      "unhealthy": sum(labels[i] == 1 for i in items)} for name, items in new.items()}
    report = {
        "images_on_disk": len(all_images),
        "annotated_images_used": len(paths),
        "images_without_annotation_excluded": len(all_images) - len(paths),
        "pre_augmented_copies": sum(1 for p in paths if AUG_RE.search(p.stem)),
        "unique_base_images": stem_groups,
        "perceptual_pairs_total": len(pairs),
        "perceptual_pairs_between_different_stems": len(perceptual_only),
        "final_groups": len(groups),
        "largest_group": max(len(g) for g in groups),
        "mixed_label_groups": len(mixed),
        "old_split_summary": split_summary(old),
        "old_split_leakage": {
            **cross_split_leaks(old_split_of, group_of),
            "test_images_whose_aug_twin_is_in_another_split": old_aug_test,
        },
        "new_split_summary": summary,
        "new_split_leakage": new_leaks,
        "params": {k: str(v) for k, v in vars(args).items()},
    }
    manifest = {name: [str(paths[i].as_posix()) for i in sorted(items)] for name, items in new.items()}
    manifest["group_id"] = {paths[i].as_posix(): group_of[i] for i in range(len(paths))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    args.output.with_name("split_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "params"}, indent=2))
    print("group size distribution:", sorted(Counter(len(g) for g in groups).items()))


if __name__ == "__main__":
    main()
