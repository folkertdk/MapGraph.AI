"""Create a smaller paired subset from the full dataset.

Example:
    python -m src.make_subset \
        --image_dir data/full/images \
        --mask_dir data/full/masks \
        --out_image_dir data/subset/images \
        --out_mask_dir data/subset/masks \
        --count 100

This copies matching image-mask pairs into data/subset.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_files(folder: str | Path):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def pair_key(path: Path) -> str:
    """Extract img number key from filenames such as img21, img_21, img-21."""
    stem = path.stem.lower()

    match = re.search(r"img[_-]?(\d+)", stem)

    if match:
        return f"img{int(match.group(1))}"

    return stem


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--out_image_dir", default="data/subset/images")
    parser.add_argument("--out_mask_dir", default="data/subset/masks")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    image_files = list_files(args.image_dir)
    mask_files = list_files(args.mask_dir)

    masks_by_key = {pair_key(mask): mask for mask in mask_files}

    pairs = []

    for image in image_files:
        key = pair_key(image)

        if key in masks_by_key:
            pairs.append((image, masks_by_key[key]))

    if not pairs:
        raise ValueError("No matching image-mask pairs found. Check filenames.")

    random.seed(args.seed)
    random.shuffle(pairs)

    selected = pairs[: min(args.count, len(pairs))]

    out_image_dir = Path(args.out_image_dir)
    out_mask_dir = Path(args.out_mask_dir)

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    for image, mask in selected:
        shutil.copy2(image, out_image_dir / image.name)
        shutil.copy2(mask, out_mask_dir / mask.name)

    print(f"Found total paired samples: {len(pairs)}")
    print(f"Copied subset samples: {len(selected)}")
    print(f"Images saved to: {out_image_dir}")
    print(f"Masks saved to: {out_mask_dir}")


if __name__ == "__main__":
    main()