from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    dataset_dir: Path = Path("dataset")
    output_dir: Path = Path("artifacts/transfer_learning")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-3
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    scheduler_patience: int = 3
    early_stopping_patience: int = 8
    models: list[str] = field(
        default_factory=lambda: [
            "mobileplantvit",
            "resnet50",
            "efficientnet_b0",
            "mobilenet_v3_large",
            "vit_b_16",
        ]
    )

