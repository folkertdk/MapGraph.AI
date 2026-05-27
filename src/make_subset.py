"""Create a small paired subset of the road segmentation dataset.

This script is useful for quick smoke tests and small experiments.

Important:
The dataset uses strict image/mask pairing. Therefore, this script copies
already-matched image/mask pairs, not random images and random masks separately.

Example PowerShell usage:

python -m src.make_subset `
  --image_dir data/mixed/images `
  --mask_dir data/mixed/masks `
  --out_image_dir data/moe_test/images `
  --out_mask_dir data/moe_test/masks `
  --count 100
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from .dataset import make_pairs


def clear_folder(folder: Path) -> None:
    """Remove old files from a folder and recreate it."""
    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--out_image_dir", required=True)
    parser.add_argument("--out_mask_dir", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    out_image_dir = Path(args.out_image_dir)
    out_mask_dir = Path(args.out_mask_dir)

    pairs = make_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        strict_pairing=True,
    )

    if len(pairs) == 0:
        raise ValueError("No paired image/mask files found.")

    random.seed(args.seed)
    random.shuffle(pairs)

    selected_pairs = pairs[: min(args.count, len(pairs))]

    clear_folder(out_image_dir)
    clear_folder(out_mask_dir)

    for image_path, mask_path in selected_pairs:
        shutil.copy2(image_path, out_image_dir / image_path.name)
        shutil.copy2(mask_path, out_mask_dir / mask_path.name)

    print(f"Found total paired samples: {len(pairs)}")
    print(f"Copied paired subset samples: {len(selected_pairs)}")
    print(f"Images saved to: {out_image_dir}")
    print(f"Masks saved to: {out_mask_dir}")


if __name__ == "__main__":
    main()