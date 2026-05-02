"""Run inference with a trained road segmentation model.

This version saves three outputs per image:
1. binary mask used for graph extraction,
2. probability map for debugging,
3. overlay image so you can visually check whether roads are detected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RoadSegmentationDataset
from .model import build_model
from .train import unpack_model_output
from .utils import ensure_dir, get_device, load_config


def otsu_threshold(prob_uint8: np.ndarray) -> float:
    """Return an Otsu threshold in probability scale [0, 1]."""
    threshold, _ = cv2.threshold(
        prob_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return float(threshold) / 255.0


def remove_tiny_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove very small false-positive blobs before graph extraction."""
    if min_area <= 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="outputs/baseline_run/best.pt")
    parser.add_argument(
        "--threshold",
        default="auto",
        help="Use a number like 0.5 or use 'auto' for Otsu threshold per image.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    torch.set_num_threads(cfg.get("train", {}).get("num_threads", 1))

    device = get_device()

    pred_dir = ensure_dir(cfg["output"].get("prediction_dir", "outputs/predictions"))
    min_area = int(cfg.get("predict", {}).get("min_component_area", 20))

    dataset = RoadSegmentationDataset(
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        img_size=cfg["data"].get("img_size", 256),
        strict_pairing=cfg["data"].get("strict_pairing", True),
        mask_threshold=cfg["data"].get("mask_threshold", 127),
        invert_mask=cfg["data"].get("invert_mask", False),
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_model(cfg).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)

            logits = unpack_model_output(model(images))
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prob_uint8 = np.clip(probs * 255.0, 0, 255).astype(np.uint8)

            if str(args.threshold).lower() == "auto":
                threshold = otsu_threshold(prob_uint8)

                # Keep threshold in a sane range. This avoids all-white masks when
                # probabilities are not yet well separated early in training.
                threshold = min(max(threshold, 0.20), 0.80)
            else:
                threshold = float(args.threshold)

            binary = (probs >= threshold).astype(np.uint8)
            binary = remove_tiny_components(binary, min_area=min_area)

            mask_uint8 = binary * 255

            image_np = (
                images[0]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                * 255
            ).astype(np.uint8)

            overlay = image_np.copy()
            overlay[binary > 0] = [255, 0, 0]
            overlay = (0.65 * image_np + 0.35 * overlay).astype(np.uint8)

            source_name = Path(batch["image_path"][0]).stem

            Image.fromarray(mask_uint8).save(pred_dir / f"{source_name}_pred_mask.png")
            Image.fromarray(prob_uint8).save(pred_dir / f"{source_name}_probability.png")
            Image.fromarray(overlay).save(pred_dir / f"{source_name}_overlay.png")

    print(f"Predicted masks, probability maps, and overlays saved in: {pred_dir}")


if __name__ == "__main__":
    main()