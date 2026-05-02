"""Train the road segmentation model.

Important fix: this version uses Dice + weighted BCE loss. Road pixels are a
small minority of a satellite tile, so plain BCE often learns the lazy solution:
predict background everywhere. Dice directly rewards overlap with thin roads,
and weighted BCE compensates for road/background imbalance.
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
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    return float(iou.mean().item())


def unpack_model_output(output):
    """MoE returns (logits, weights), baseline returns logits only."""
    return output[0] if isinstance(output, tuple) else output


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
) -> tuple[float, float, float]:
    model.train(train)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    steps = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(train):
            logits = unpack_model_output(model(images))
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += float(loss.item())
        total_dice += dice_score_from_logits(logits.detach(), masks, threshold)
        total_iou += iou_score_from_logits(logits.detach(), masks, threshold)
        steps += 1

    return (
        total_loss / max(steps, 1),
        total_dice / max(steps, 1),
        total_iou / max(steps, 1),
    )


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

    run_dir = ensure_dir(cfg["output"]["run_dir"])

    dataset = RoadSegmentationDataset(
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        img_size=data_cfg.get("img_size", 256),
        strict_pairing=data_cfg.get("strict_pairing", True),
        mask_threshold=data_cfg.get("mask_threshold", 127),
        invert_mask=data_cfg.get("invert_mask", False),
    )

    pos_weight_value, pos_fraction = estimate_pos_weight(dataset)

    print(f"Loaded {len(dataset)} paired samples")
    print(f"Estimated road pixel fraction: {pos_fraction:.4%}")
    print(f"Using BCE pos_weight: {pos_weight_value:.2f}")

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

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_dice",
                "train_iou",
                "val_loss",
                "val_dice",
                "val_iou",
            ],
        )

        writer.writeheader()

        for epoch in range(1, train_cfg.get("epochs", 50) + 1):
            train_loss, train_dice, train_iou = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
                threshold=threshold,
            )

            val_loss, val_dice, val_iou = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                train=False,
                threshold=threshold,
            )

            scheduler.step(val_loss)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }

            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} dice={train_dice:.4f} iou={train_iou:.4f} | "
                f"val_loss={val_loss:.4f} dice={val_dice:.4f} iou={val_iou:.4f}"
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "pos_weight": pos_weight_value,
                "road_pixel_fraction": pos_fraction,
            }

            torch.save(checkpoint, run_dir / "last.pt")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(checkpoint, run_dir / "best.pt")

    print(f"Training complete. Checkpoints and log saved in: {run_dir}")


if __name__ == "__main__":
    main()