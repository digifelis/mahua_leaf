from __future__ import annotations

import torch
from torch import nn


class DepthConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GroupConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(1, channels // 2)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        channel_attn = torch.sigmoid(self.mlp(avg) + self.mlp(max_pool))
        x = x * channel_attn
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.amax(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial(torch.cat([spatial_avg, spatial_max], dim=1)))
        return x * spatial_attn


class LinearSelfAttention(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 1 + 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        query, key, value = torch.split(qkv, [1, x.shape[-1], x.shape[-1]], dim=-1)
        attn = torch.softmax(query, dim=1)
        attn = self.dropout(attn)
        context = torch.sum(attn * key, dim=1, keepdim=True)
        return self.proj(torch.nn.functional.gelu(value) * context)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 256, ffn_dim: int = 512, dropout: float = 0.30) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearSelfAttention(dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ffn(x))


class MobilePlantViT(nn.Module):
    """PyTorch implementation of the arXiv:2503.16628 MobilePlantViT idea for binary logits."""

    def __init__(
        self,
        num_outputs: int = 1,
        embed_dim: int = 256,
        ffn_dim: int = 512,
        dropout: float = 0.20,
        encoder_dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.stem = DepthConvBlock(3, 32, stride=2)
        self.stage1 = nn.Sequential(GroupConvBlock(32), CBAM(32), DepthConvBlock(32, 64, stride=2))
        self.stage2 = nn.Sequential(GroupConvBlock(64), GroupConvBlock(64), CBAM(64), DepthConvBlock(64, 128, stride=2))
        self.stage3 = nn.Sequential(
            GroupConvBlock(128),
            GroupConvBlock(128),
            GroupConvBlock(128),
            GroupConvBlock(128),
            CBAM(128),
        )
        self.patch_embed = nn.Sequential(DepthConvBlock(128, embed_dim, stride=2), CBAM(embed_dim))
        self.encoder = EncoderBlock(embed_dim, ffn_dim, encoder_dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(self.dropout(x)).squeeze(1)

