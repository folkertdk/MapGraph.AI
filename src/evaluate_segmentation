"""Evaluate predicted road masks against ground-truth masks.

Example:
    python -m src.evaluate_segmentation --config configs/baseline.yaml

Outputs:
    outputs/evaluation/segmentation_metrics.csv
    outputs/evaluation/summary.json
    outputs/evaluation/comparison_grid.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .utils import ensure_dir, load_config

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def pair_key(path: Path) -> str:
    stem = path.stem.lower()

    # Remove prediction suffix if present.
    stem = stem.replace("_pred_mask", "")
    stem = stem.replace("_prediction", "")
    stem = stem.replace("_mask", "")

    match = re.search(r"img[_-]?(\d+)", stem)

    if match:
        return f"img{int(match.group(1))}"

    return stem


def list_files(folder: str | Path):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def load_binary(path: Path, size=None, threshold: int = 127) -> np.ndarray:
    img = Image.open(path).convert("L")

    if size is not None:
        img = img.resize(size, resample=Image.NEAREST)

    arr = np.asarray(img)

    return arr > threshold


def load_rgb(path: Path, size=None) -> np.ndarray:
    img = Image.open(path).convert("RGB")

    if size is not None:
        img = img.resize(size, resample=Image.BILINEAR)

    return np.asarray(img) / 255.0


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    eps = 1e-8

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def save_comparison_grid(rows, out_path: Path, max_show: int = 6):
    rows = rows[:max_show]

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.2 * len(rows)))

    if len(rows) == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows):
        image = row["image"]
        gt = row["gt"]
        pred = row["pred"]

        overlay = image.copy()
        overlay[pred > 0] = [1.0, 0.0, 0.0]
        overlay = 0.65 * image + 0.35 * overlay

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Input Image")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Prediction")

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(
            f"Overlay\nDice={row['dice']:.3f}, IoU={row['iou']:.3f}"
        )

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--prediction_dir", default=None)
    parser.add_argument("--out_dir", default="outputs/evaluation")
    parser.add_argument("--max_show", type=int, default=6)

    args = parser.parse_args()

    cfg = load_config(args.config)

    image_dir = Path(cfg["data"]["image_dir"])
    mask_dir = Path(cfg["data"]["mask_dir"])
    prediction_dir = Path(args.prediction_dir or cfg["output"]["prediction_dir"])

    out_dir = ensure_dir(args.out_dir)

    image_files = list_files(image_dir)
    gt_mask_files = list_files(mask_dir)
    pred_files = list_files(prediction_dir)

    images_by_key = {pair_key(p): p for p in image_files}
    gt_by_key = {pair_key(p): p for p in gt_mask_files}
    pred_by_key = {pair_key(p): p for p in pred_files if "_pred_mask" in p.stem.lower()}

    common_keys = sorted(set(gt_by_key) & set(pred_by_key))

    if not common_keys:
        raise ValueError(
            "No matching ground-truth and predicted masks found. "
            "Check prediction filenames and mask filenames."
        )

    metric_rows = []
    visual_rows = []

    for key in common_keys:
        gt_path = gt_by_key[key]
        pred_path = pred_by_key[key]
        image_path = images_by_key.get(key)

        pred_img = Image.open(pred_path).convert("L")
        size = pred_img.size

        pred = np.asarray(pred_img) > 127
        gt = load_binary(gt_path, size=size, threshold=cfg["data"].get("mask_threshold", 127))

        metrics = compute_metrics(pred, gt)

        row = {
            "key": key,
            "ground_truth": str(gt_path),
            "prediction": str(pred_path),
            **metrics,
        }

        metric_rows.append(row)

        if image_path is not None and len(visual_rows) < args.max_show:
            image = load_rgb(image_path, size=size)

            visual_rows.append(
                {
                    "image": image,
                    "gt": gt,
                    "pred": pred,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                }
            )

    csv_path = out_dir / "segmentation_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "num_samples": len(metric_rows),
        "mean_dice": float(np.mean([r["dice"] for r in metric_rows])),
        "mean_iou": float(np.mean([r["iou"] for r in metric_rows])),
        "mean_precision": float(np.mean([r["precision"] for r in metric_rows])),
        "mean_recall": float(np.mean([r["recall"] for r in metric_rows])),
        "mean_pixel_accuracy": float(np.mean([r["pixel_accuracy"] for r in metric_rows])),
    }

    summary_path = out_dir / "summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    grid_path = out_dir / "comparison_grid.png"
    save_comparison_grid(visual_rows, grid_path, max_show=args.max_show)

    print("Evaluation complete.")
    print(f"Samples evaluated: {summary['num_samples']}")
    print(f"Mean Dice: {summary['mean_dice']:.4f}")
    print(f"Mean IoU: {summary['mean_iou']:.4f}")
    print(f"Mean Precision: {summary['mean_precision']:.4f}")
    print(f"Mean Recall: {summary['mean_recall']:.4f}")
    print(f"Mean Pixel Accuracy: {summary['mean_pixel_accuracy']:.4f}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved comparison grid: {grid_path}")


if __name__ == "__main__":
    main()