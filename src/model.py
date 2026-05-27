"""Segmentation models for MapGraph.AI.

This file contains two model families:

1. SimpleUNet
   - The stable baseline model for road segmentation.
   - This path is kept unchanged so baseline experiments remain reproducible.

2. MoEUNet
   - A Mixture-of-Experts version of U-Net.
   - A small router looks at each satellite tile and assigns probabilities
     over several U-Net experts.
   - The experts produce road-mask logits, and the router combines them.

Why MoE for this project?
Road appearance differs across cities. For example, Paris, Shanghai, Vegas,
and Khartoum may have different road widths, textures, building density,
colors, and satellite conditions. A Mixture-of-Experts model can learn multiple
specialized segmentation behaviours instead of forcing one U-Net to handle all
cities in the same way.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    """Choose a safe GroupNorm group count that divides the channel count.

    GroupNorm is used instead of BatchNorm because this project often trains
    with batch_size=1. BatchNorm can become unstable with very small batches.
    """
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


class TileRouter(nn.Module):
    """Router network for the Mixture-of-Experts U-Net.

    The router receives the input satellite tile and produces a probability
    distribution over experts.

    Example:
        expert_0 = 0.70
        expert_1 = 0.20
        expert_2 = 0.10

    This means the router thinks expert 0 should contribute the most for this
    particular image tile.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_experts: int = 3,
        hidden_channels: int = 32,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(hidden_channels, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_features = self.features(x)
        router_logits = self.classifier(router_features)
        router_probs = F.softmax(router_logits, dim=1)

        return router_probs


class MoEUNet(nn.Module):
    """Mixture-of-Experts U-Net for road segmentation.

    Each expert is a small U-Net. The router decides how much each expert
    should contribute for each image.

    This implementation is intentionally conservative:
    - It keeps the baseline SimpleUNet untouched.
    - It is compatible with the existing train.py because the first returned
      item is always the segmentation logits.
    - It also returns router information that we can later log during training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
        num_experts: int = 3,
        router_hidden_channels: int = 32,
        top_k: int | None = None,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()

        if num_experts < 2:
            raise ValueError("MoEUNet requires at least 2 experts.")

        if top_k is not None and (top_k < 1 or top_k > num_experts):
            raise ValueError("top_k must be between 1 and num_experts.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        self.router = TileRouter(
            in_channels=in_channels,
            num_experts=num_experts,
            hidden_channels=router_hidden_channels,
        )

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

    def _apply_top_k(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Keep only the top-k expert probabilities and renormalize them.

        If top_k is None, all experts contribute.

        Example with top_k=2:
            original: [0.60, 0.30, 0.10]
            top-k:    [0.67, 0.33, 0.00]

        This makes routing more specialized because not every expert is used
        equally for every image.
        """
        if self.top_k is None or self.top_k == self.num_experts:
            return router_probs

        topk_values, topk_indices = torch.topk(
            router_probs,
            k=self.top_k,
            dim=1,
        )

        sparse_probs = torch.zeros_like(router_probs)
        sparse_probs.scatter_(dim=1, index=topk_indices, src=topk_values)

        sparse_probs = sparse_probs / sparse_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return sparse_probs

    def _load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Encourage the router to use all experts instead of collapsing to one.

        Without this, the router may learn to always select the same expert.
        That would make the model look like MoE, but practically behave like
        a single U-Net.

        The loss compares average expert usage against a uniform distribution.
        Lower is better.
        """
        mean_usage = router_probs.mean(dim=0)

        target_usage = torch.full_like(
            mean_usage,
            fill_value=1.0 / self.num_experts,
        )

        balance_loss = F.mse_loss(mean_usage, target_usage)

        return self.load_balance_weight * balance_loss

    def _router_diagnostics(self, router_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return useful router statistics for logging and analysis."""
        with torch.no_grad():
            mean_usage = router_probs.mean(dim=0)

            entropy = -(
                router_probs * torch.log(router_probs.clamp_min(1e-8))
            ).sum(dim=1).mean()

            selected_expert = torch.argmax(router_probs, dim=1)

        return {
            "mean_usage": mean_usage,
            "entropy": entropy,
            "selected_expert": selected_expert,
        }

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Returns:
            logits:
                Final segmentation logits with shape [B, 1, H, W].

            router_probs:
                Probability given to each expert with shape [B, num_experts].

            aux_loss:
                Load-balancing loss. Later, train.py can add this to the main
                Dice + BCE loss.

            diagnostics:
                Extra router information for experiment analysis.
        """
        router_probs = self.router(x)
        routing_weights = self._apply_top_k(router_probs)

        expert_logits = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1,
        )

        logits = (
            expert_logits
            * routing_weights[:, :, None, None, None]
        ).sum(dim=1)

        aux_loss = self._load_balance_loss(router_probs)
        diagnostics = self._router_diagnostics(router_probs)

        return logits, router_probs, aux_loss, diagnostics


def build_model(config: dict) -> nn.Module:
    """Build either the baseline U-Net or the MoE U-Net.

    The architecture is controlled by config:

        model:
          architecture: baseline

    or:

        model:
          architecture: moe

    This keeps the baseline experiment safe and reproducible.
    """
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
            router_hidden_channels=model_cfg.get("router_hidden_channels", 32),
            top_k=model_cfg.get("top_k", None),
            load_balance_weight=model_cfg.get("load_balance_weight", 0.01),
        )

    raise ValueError(f"Unknown architecture: {architecture}")