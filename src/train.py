"""Train the road segmentation model.

This training script supports both:

1. Baseline U-Net
   - Standard road segmentation model.
   - Uses Dice + weighted BCE loss.

2. MoE U-Net
   - Mixture-of-Experts U-Net.
   - Uses the same segmentation loss.
   - Also supports MoE auxiliary load-balancing loss.
   - Logs router entropy and expert usage for experiment analysis.

Why the extra MoE logging matters:
A Mixture-of-Experts model should not only improve Dice/IoU. We also want to
show whether different experts are actually being used. This script therefore
records expert usage and router entropy in the training CSV.
"""

from __future__ import annotations

import argparse
import csv
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RoadSegmentationDataset, split_dataset
from .model import build_model
from .utils import ensure_dir, get_device, load_config, set_seed


class DiceBCELoss(nn.Module):
    """Binary segmentation loss = weighted BCE + Dice loss."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
    ):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice_loss = 1.0 - (
            (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        ).mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


def dice_score_from_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute Dice score from raw logits."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

    dice = (2 * intersection + 1e-6) / (union + 1e-6)

    return float(dice.mean().item())


def iou_score_from_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute IoU score from raw logits."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    return float(iou.mean().item())


def parse_model_output(output):
    """Handle both baseline and MoE model outputs.

    Baseline SimpleUNet returns:
        logits

    Polished MoEUNet returns:
        logits, router_probs, aux_loss, diagnostics

    Older/simple MoE variants may return:
        logits, router_probs

    Returns:
        logits:
            Segmentation logits.

        aux_loss:
            Optional MoE auxiliary load-balancing loss.

        diagnostics:
            Dictionary containing router statistics such as entropy and
            mean expert usage.
    """
    if torch.is_tensor(output):
        return output, None, {}

    if not isinstance(output, (tuple, list)):
        raise TypeError(f"Unexpected model output type: {type(output)}")

    logits = output[0]
    aux_loss = None
    diagnostics = {}

    if len(output) >= 2 and torch.is_tensor(output[1]):
        router_probs = output[1]

        with torch.no_grad():
            diagnostics["mean_usage"] = router_probs.mean(dim=0)
            diagnostics["entropy"] = -(
                router_probs * torch.log(router_probs.clamp_min(1e-8))
            ).sum(dim=1).mean()
            diagnostics["selected_expert"] = torch.argmax(router_probs, dim=1)

    if len(output) >= 3 and torch.is_tensor(output[2]):
        aux_loss = output[2]

    if len(output) >= 4 and isinstance(output[3], dict):
        diagnostics.update(output[3])

    return logits, aux_loss, diagnostics


def estimate_pos_weight(dataset, max_batches: int = 50) -> tuple[float, float]:
    """Estimate road-pixel fraction and BCE pos_weight from the dataset."""
    positives = 0.0
    total = 0.0

    for i in range(min(len(dataset), max_batches)):
        mask = dataset[i]["mask"]
        positives += float(mask.sum().item())
        total += float(mask.numel())

    pos_fraction = positives / max(total, 1.0)

    if positives <= 0:
        return 1.0, 0.0

    neg = total - positives

    # Cap keeps the model from turning everything white on very sparse masks.
    pos_weight = min(max(neg / positives, 1.0), 50.0)

    return float(pos_weight), float(pos_fraction)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    threshold: float,
    num_experts: int = 0,
) -> dict:
    """Run one training or validation epoch.

    For baseline:
        tracks segmentation loss, Dice, and IoU.

    For MoE:
        also tracks auxiliary load-balancing loss, router entropy,
        and average expert usage.
    """
    model.train(train)

    total_loss = 0.0
    total_seg_loss = 0.0
    total_aux_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_entropy = 0.0

    expert_usage_sum = None
    expert_usage_steps = 0

    steps = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(train):
            output = model(images)
            logits, aux_loss, diagnostics = parse_model_output(output)

            seg_loss = criterion(logits, masks)

            router_loss = None

            if isinstance(output, (tuple, list)) and len(output) >= 2 and "domain_label" in batch:
                router_probs = output[1]
                domain_labels = batch["domain_label"].to(device)

                valid = domain_labels >= 0
                if valid.any():
                    router_loss = torch.nn.functional.nll_loss(
                        torch.log(router_probs[valid].clamp_min(1e-8)),
                        domain_labels[valid],
                    )

            loss = seg_loss

            if aux_loss is not None:
                loss = loss + aux_loss

            if router_loss is not None:
                router_supervision_weight = 0.05
                loss = loss + router_supervision_weight * router_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += float(loss.item())
        total_seg_loss += float(seg_loss.item())

        if aux_loss is not None:
            total_aux_loss += float(aux_loss.item())

        total_dice += dice_score_from_logits(logits.detach(), masks, threshold)
        total_iou += iou_score_from_logits(logits.detach(), masks, threshold)

        if "entropy" in diagnostics:
            total_entropy += float(diagnostics["entropy"].detach().cpu().item())

        if "mean_usage" in diagnostics:
            usage = diagnostics["mean_usage"].detach().cpu()

            if expert_usage_sum is None:
                expert_usage_sum = torch.zeros_like(usage)

            expert_usage_sum += usage
            expert_usage_steps += 1

        steps += 1

    metrics = {
        "loss": total_loss / max(steps, 1),
        "seg_loss": total_seg_loss / max(steps, 1),
        "aux_loss": total_aux_loss / max(steps, 1),
        "dice": total_dice / max(steps, 1),
        "iou": total_iou / max(steps, 1),
        "router_entropy": total_entropy / max(steps, 1),
        "expert_usage": [],
    }

    if expert_usage_sum is not None and expert_usage_steps > 0:
        metrics["expert_usage"] = (
            expert_usage_sum / expert_usage_steps
        ).tolist()
    elif num_experts > 0:
        metrics["expert_usage"] = [0.0 for _ in range(num_experts)]

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    torch.set_num_threads(cfg["train"].get("num_threads", 1))
    set_seed(cfg["train"].get("seed", 42))

    device = get_device()

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {})

    architecture = model_cfg.get("architecture", "baseline").lower()
    num_experts = int(model_cfg.get("num_experts", 0)) if architecture == "moe" else 0

    run_dir = ensure_dir(cfg["output"]["run_dir"])

    dataset = RoadSegmentationDataset(
        image_dir=data_cfg.get("image_dirs", data_cfg.get("image_dir")),
        mask_dir=data_cfg.get("mask_dirs", data_cfg.get("mask_dir")),
        img_size=data_cfg.get("img_size", 256),
        strict_pairing=data_cfg.get("strict_pairing", True),
        mask_threshold=data_cfg.get("mask_threshold", 127),
        invert_mask=data_cfg.get("invert_mask", False),
    )

    pos_weight_value, pos_fraction = estimate_pos_weight(dataset)

    print(f"Loaded {len(dataset)} paired samples")
    print(f"Estimated road pixel fraction: {pos_fraction:.4%}")
    print(f"Using BCE pos_weight: {pos_weight_value:.2f}")

    if architecture == "moe":
        print(
            "Using MoE architecture | "
            f"num_experts={num_experts}, "
            f"top_k={model_cfg.get('top_k', None)}, "
            f"load_balance_weight={model_cfg.get('load_balance_weight', 0.0)}"
        )

    if pos_fraction == 0.0:
        raise ValueError(
            "All masks appear empty. Check mask_dir, mask_threshold, and invert_mask in the config."
        )

    if pos_fraction > 0.50:
        print(
            "Warning: masks are more than 50% road. "
            "If roads should be white lines on black background, check invert_mask."
        )

    train_set, val_set = split_dataset(
        dataset,
        val_split=train_cfg.get("val_split", 0.2),
        seed=train_cfg.get("seed", 42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
    )

    model = build_model(cfg).to(device)

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = DiceBCELoss(
        pos_weight=pos_weight,
        dice_weight=train_cfg.get("dice_weight", 1.0),
        bce_weight=train_cfg.get("bce_weight", 1.0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    threshold = float(train_cfg.get("metric_threshold", 0.5))

    best_val = math.inf
    log_path = run_dir / "training_log.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "train_seg_loss",
        "train_aux_loss",
        "train_dice",
        "train_iou",
        "train_router_entropy",
        "val_loss",
        "val_seg_loss",
        "val_aux_loss",
        "val_dice",
        "val_iou",
        "val_router_entropy",
    ]

    for expert_idx in range(num_experts):
        fieldnames.append(f"train_expert_{expert_idx}_usage")
        fieldnames.append(f"val_expert_{expert_idx}_usage")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for epoch in range(1, train_cfg.get("epochs", 50) + 1):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
                threshold=threshold,
                num_experts=num_experts,
            )

            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                train=False,
                threshold=threshold,
                num_experts=num_experts,
            )

            scheduler.step(val_metrics["loss"])

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_seg_loss": train_metrics["seg_loss"],
                "train_aux_loss": train_metrics["aux_loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "train_router_entropy": train_metrics["router_entropy"],
                "val_loss": val_metrics["loss"],
                "val_seg_loss": val_metrics["seg_loss"],
                "val_aux_loss": val_metrics["aux_loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_router_entropy": val_metrics["router_entropy"],
            }

            for expert_idx in range(num_experts):
                row[f"train_expert_{expert_idx}_usage"] = train_metrics["expert_usage"][
                    expert_idx
                ]
                row[f"val_expert_{expert_idx}_usage"] = val_metrics["expert_usage"][
                    expert_idx
                ]

            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"dice={train_metrics['dice']:.4f} "
                f"iou={train_metrics['iou']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"dice={val_metrics['dice']:.4f} "
                f"iou={val_metrics['iou']:.4f}"
            )

            if num_experts > 0:
                train_usage_text = ", ".join(
                    [
                        f"e{idx}={usage:.2f}"
                        for idx, usage in enumerate(train_metrics["expert_usage"])
                    ]
                )

                val_usage_text = ", ".join(
                    [
                        f"e{idx}={usage:.2f}"
                        for idx, usage in enumerate(val_metrics["expert_usage"])
                    ]
                )

                print(
                    f"  MoE router | "
                    f"train_entropy={train_metrics['router_entropy']:.4f} "
                    f"val_entropy={val_metrics['router_entropy']:.4f}"
                )
                print(f"  Train expert usage: {train_usage_text}")
                print(f"  Val expert usage:   {val_usage_text}")

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "pos_weight": pos_weight_value,
                "road_pixel_fraction": pos_fraction,
            }

            torch.save(checkpoint, run_dir / "last.pt")

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(checkpoint, run_dir / "best.pt")

    print(f"Training complete. Checkpoints and log saved in: {run_dir}")


if __name__ == "__main__":
    main()