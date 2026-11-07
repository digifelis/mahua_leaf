from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .mobileplantvit import MobilePlantViT


def _load_state_dict_if_given(model: nn.Module, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)


def build_model(name: str, pretrained: bool = True, checkpoint_path: str | None = None) -> nn.Module:
    name = name.lower()
    if name == "mobileplantvit":
        model = MobilePlantViT(num_outputs=1)
        _load_state_dict_if_given(model, checkpoint_path)
        return model

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model

    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model

    if name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model

    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, 1)
        return model

    raise ValueError(f"Unknown model: {name}")
