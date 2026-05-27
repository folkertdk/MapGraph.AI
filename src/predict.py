"""Run inference with a trained road segmentation model.

This script supports both:

1. Baseline U-Net
   - Saves predicted mask, probability map, and overlay.

2. MoE U-Net
   - Saves the same visual prediction outputs.
   - Also saves router/expert probabilities into a CSV file so we can analyse
     which expert was used for each satellite tile.

Prediction outputs per image:
    1. *_pred_mask.png
    2. *_probability.png
    3. *_overlay.png

Additional MoE output:
    moe_router_weights.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RoadSegmentationDataset
from .model import build_model
from .utils import ensure_dir, get_device, load_config


def parse_prediction_output(output):
    """Handle both baseline and MoE model outputs during prediction.

    Baseline returns:
        logits

    MoE returns:
        logits, router_probs, aux_loss, diagnostics

    We only need logits for the mask, but for MoE we also save router_probs.
    """
    if torch.is_tensor(output):
        return output, None

    if not isinstance(output, (tuple, list)):
        raise TypeError(f"Unexpected model output type: {type(output)}")

    logits = output[0]

    router_probs = None
    if len(output) >= 2 and torch.is_tensor(output[1]):
        router_probs = output[1]

    return logits, router_probs


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

    model_cfg = cfg.get("model", {})
    architecture = model_cfg.get("architecture", "baseline").lower()
    num_experts = int(model_cfg.get("num_experts", 0)) if architecture == "moe" else 0

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

    router_rows = []

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)

            output = model(images)
            logits, router_probs = parse_prediction_output(output)

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

            if router_probs is not None:
                probs_list = router_probs[0].detach().cpu().tolist()
                selected_expert = int(np.argmax(probs_list))

                row = {
                    "image": source_name,
                    "selected_expert": selected_expert,
                }

                for expert_idx, prob in enumerate(probs_list):
                    row[f"expert_{expert_idx}_prob"] = prob

                router_rows.append(row)

    if router_rows:
        router_csv_path = pred_dir / "moe_router_weights.csv"

        fieldnames = ["image", "selected_expert"]
        for expert_idx in range(num_experts):
            fieldnames.append(f"expert_{expert_idx}_prob")

        with open(router_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(router_rows)

        print(f"MoE router weights saved in: {router_csv_path}")

    print(f"Predicted masks, probability maps, and overlays saved in: {pred_dir}")


if __name__ == "__main__":
    main()