"""Segmentation models for MapGraph.AI.

The baseline is a small U-Net. It uses GroupNorm instead of BatchNorm because
student experiments often run with batch_size=1; BatchNorm can be unstable with
very small batches and can contribute to poor all-black/all-white predictions.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _groups(channels: int) -> int:
    """Choose a safe GroupNorm group count that divides the channel count."""
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return g

    return 1


class ConvBlock(nn.Module):
    """Two Conv2D + GroupNorm + ReLU layers used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleUNet(nn.Module):
    """Lightweight U-Net for binary road segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()

        c = base_channels

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)


class GatingNetwork(nn.Module):
    """Tiny router that gives each image a probability over experts."""

    def __init__(self, in_channels: int = 3, num_experts: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


class MoEUNet(nn.Module):
    """Simple future-ready Mixture-of-Experts U-Net."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
        num_experts: int = 3,
    ):
        super().__init__()

        self.gate = GatingNetwork(in_channels, num_experts)

        self.experts = nn.ModuleList(
            [
                SimpleUNet(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    base_channels=base_channels,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.gate(x)

        expert_logits = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1,
        )

        logits = (expert_logits * weights[:, :, None, None, None]).sum(dim=1)

        return logits, weights


def build_model(config: dict) -> nn.Module:
    """Build either the baseline U-Net or the MoE U-Net."""
    model_cfg = config.get("model", {})
    architecture = model_cfg.get("architecture", "baseline").lower()

    if architecture == "baseline":
        return SimpleUNet(
            in_channels=model_cfg.get("in_channels", 3),
            out_channels=model_cfg.get("out_channels", 1),
            base_channels=model_cfg.get("base_channels", 32),
        )

    if architecture == "moe":
        return MoEUNet(
            in_channels=model_cfg.get("in_channels", 3),
            out_channels=model_cfg.get("out_channels", 1),
            base_channels=model_cfg.get("base_channels", 16),
            num_experts=model_cfg.get("num_experts", 3),
        )

    raise ValueError(f"Unknown architecture: {architecture}")