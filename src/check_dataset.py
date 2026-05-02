"""Check image/mask pairing and mask quality before training.

Run:
    python -m src.check_dataset --config configs/baseline.yaml

This is the first command to run when predictions become all black/all white.
It confirms that strict pairing is finding the correct files and reports how
many road pixels each mask contains.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .dataset import RoadSegmentationDataset
from .utils import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--max_show", type=int, default=6)
    parser.add_argument("--out", default="outputs/dataset_check.png")

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

    print(f"Found {len(dataset)} valid image/mask pairs.\n")

    n = min(len(dataset), args.max_show)
    road_fracs = []

    for i in range(len(dataset)):
        sample = dataset[i]
        frac = float(sample["mask"].mean().item())
        road_fracs.append(frac)

        if i < 20:
            print(f"{i:03d}: road_pixels={frac:.4%}")
            print(f"     image: {sample['image_path']}")
            print(f"     mask : {sample['mask_path']}")

    print("\nSummary:")
    print(f"  min road fraction : {np.min(road_fracs):.4%}")
    print(f"  mean road fraction: {np.mean(road_fracs):.4%}")
    print(f"  max road fraction : {np.max(road_fracs):.4%}")

    if np.mean(road_fracs) == 0:
        print("ERROR: all masks are empty. Change mask_threshold or check mask_dir.")

    if np.mean(road_fracs) > 0.5:
        print("WARNING: masks look mostly white. You may need data.invert_mask: true.")

    ensure_dir("outputs")

    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))

    if n == 1:
        axes = np.array([axes])

    for row in range(n):
        sample = dataset[row]

        image = sample["image"].permute(1, 2, 0).numpy()
        mask = sample["mask"].squeeze(0).numpy()

        overlay = image.copy()
        overlay[mask > 0] = [1.0, 0.0, 0.0]
        overlay = 0.65 * image + 0.35 * overlay

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("image")

        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title(f"mask ({mask.mean():.2%} road)")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("overlay check")

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=160)

    print(f"\nSaved visual dataset check to: {args.out}")


if __name__ == "__main__":
    main()
    