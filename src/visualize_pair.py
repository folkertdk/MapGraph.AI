"""Load and visualize one image/mask pair."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from .dataset import RoadSegmentationDataset
from .utils import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="outputs/example_pair.png")

    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    dataset = RoadSegmentationDataset(
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        img_size=data_cfg.get("img_size", 256),
        strict_pairing=data_cfg.get("strict_pairing", True),
        mask_threshold=data_cfg.get("mask_threshold", 127),
        invert_mask=data_cfg.get("invert_mask", False),
    )

    sample = dataset[args.index]

    image = sample["image"].permute(1, 2, 0).numpy()
    mask = sample["mask"].squeeze(0).numpy()

    overlay = image.copy()
    overlay[mask > 0] = [1.0, 0.0, 0.0]
    overlay = 0.65 * image + 0.35 * overlay

    ensure_dir("outputs")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Aerial image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Road mask ({mask.mean():.2%} road)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay sanity check")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=160)

    print(f"Saved visualization to {args.out}")
    print(f"Image path: {sample['image_path']}")
    print(f"Mask path:  {sample['mask_path']}")


if __name__ == "__main__":
    main()