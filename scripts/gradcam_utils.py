"""Minimal Grad-CAM helper used by gradcam_revision2.py."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __call__(self, image_tensor: torch.Tensor, target_class: int) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        logit = self.model(image_tensor).view(-1)
        score = logit if target_class == 1 else -logit
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np, float(torch.sigmoid(logit).item())


def overlay_cam(image: Image.Image, cam: np.ndarray) -> np.ndarray:
    rgb = image.resize((224, 224)).convert("RGB")
    base = np.asarray(rgb).astype(np.float32) / 255.0
    heat = plt.colormaps["jet"](cam)[..., :3]
    return np.clip(0.58 * base + 0.42 * heat, 0, 1)


