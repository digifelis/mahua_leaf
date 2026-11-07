"""Grouped, class-stratified 5-fold CV manifests from the duplicate groups of grouped_split.json.

Fold k: test = fold k (~20%); validation = ~15% of all images taken group-wise from the other folds;
training = the rest (~65%). Groups never cross subsets. Writes artifacts/revision2/cv/fold{k}.json
(same format as grouped_split.json) and cv_summary.json.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mahua_tl.data import _annotation_to_binary_label  # noqa: E402

SPLIT = ROOT / "artifacts/revision2/split/grouped_split.json"
OUT = ROOT / "artifacts/revision2/cv"
K, SEED = 5, 42


def label_of(path: str) -> int:
    ann = (ROOT / path).parent.parent / "annotations" / (Path(path).stem + ".json")
    return _annotation_to_binary_label(ann)  # same rule as training


def greedy(groups: dict[int, list[str]], gids: list[int], glabel: dict[int, int], targets: dict[str, float], rng) -> dict[str, list[int]]:
    out = {n: [] for n in targets}
    for label in (0, 1):
        ids = [g for g in gids if glabel[g] == label]
        rng.shuffle(ids)
        total = sum(len(groups[g]) for g in ids)
        filled = {n: 0 for n in targets}
        for g in ids:
            n = max(targets, key=lambda s: (targets[s] * total - filled[s]) / (targets[s] * total))
            out[n].append(g)
            filled[n] += len(groups[g])
    return out


def main() -> None:
    gid = json.loads(SPLIT.read_text())["group_id"]
    groups: dict[int, list[str]] = defaultdict(list)
    for p, g in gid.items():
        groups[g].append(p)
    labels = {p: label_of(p) for p in gid}
    glabel = {g: max(labels[p] for p in ps) for g, ps in groups.items()}
    rng = random.Random(SEED)
    folds = greedy(groups, sorted(groups), glabel, {f"f{k}": 1 / K for k in range(K)}, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for k in range(K):
        test_g = folds[f"f{k}"]
        rest = [g for j in range(K) if j != k for g in folds[f"f{j}"]]
        tv = greedy(groups, sorted(rest), glabel, {"train": 0.65 / 0.80, "val": 0.15 / 0.80}, random.Random(SEED + k))
        man = {name: sorted(p for g in gs for p in groups[g]) for name, gs in (("train", tv["train"]), ("val", tv["val"]), ("test", test_g))}
        sets = [set(man[n]) for n in ("train", "val", "test")]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]) and sum(map(len, sets)) == len(gid)
        man["group_id"] = gid
        (OUT / f"fold{k}.json").write_text(json.dumps(man, indent=1))
        summary[f"fold{k}"] = {n: {"total": len(man[n]), "healthy": sum(1 - labels[p] for p in man[n])} for n in ("train", "val", "test")}
    all_test = [p for k in range(K) for p in json.loads((OUT / f"fold{k}.json").read_text())["test"]]
    assert sorted(all_test) == sorted(gid), "every image must be in exactly one test fold"
    (OUT / "cv_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
