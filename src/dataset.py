"""Dataset loader for aerial road segmentation.

This loader is intentionally defensive because road datasets commonly fail for
simple reasons: image/mask names do not match perfectly, masks are accidentally
inverted, or masks are resized with the wrong interpolation. The functions below
make those cases visible instead of silently training a model that predicts all
black or all white masks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_image_files(folder: str | Path) -> List[Path]:
    """Return all image-like files in a folder, sorted for reproducibility."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )

    if not files:
        raise FileNotFoundError(f"No image files found in: {folder}")

    return files


def _stem_key(path: Path) -> str:
    """Return a robust pairing key for SpaceNet-style image/mask filenames.

    Examples that should match:
    - SN3_roads_train_AOI_5_Khartoum_PS-RGB_img21.tif
    - SN3_roads_train_AOI_5_Khartoum_mask_img21.png
    - SN3_roads_train_AOI_5_Khartoum_geojson_roads_img21.png

    The most reliable identifier is usually the trailing img number. If that is
    present, we combine it with the AOI/city string so different cities do not
    collide. If no img number exists, we fall back to a cleaned stem.
    """
    stem = path.stem.lower()

    img_match = re.search(r"img[_-]?(\d+)", stem)
    aoi_match = re.search(r"aoi[_-]?\d+[_-]?[a-z0-9]+", stem)

    if img_match:
        aoi = aoi_match.group(0) if aoi_match else ""
        return f"{aoi}_img{int(img_match.group(1))}"

    cleaned = stem
    for token in ["ps-rgb", "rgb", "image", "images", "mask", "masks", "roads", "road", "geojson"]:
        cleaned = cleaned.replace(token, "")

    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return cleaned


def make_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
    strict_pairing: bool = True,
) -> List[Tuple[Path, Path]]:
    """Create image/mask pairs.

    strict_pairing=True is recommended for training. It prevents accidentally
    training image A with mask B, which is one of the main reasons predictions
    become useless.
    """
    image_files = list_image_files(image_dir)
    mask_files = list_image_files(mask_dir)

    if strict_pairing:
        masks_by_key = {}
        duplicate_mask_keys = set()

        for mask in mask_files:
            key = _stem_key(mask)

            if key in masks_by_key:
                duplicate_mask_keys.add(key)

            masks_by_key[key] = mask

        pairs: List[Tuple[Path, Path]] = []
        unmatched_images = []

        for image in image_files:
            key = _stem_key(image)
            mask = masks_by_key.get(key)

            if mask is None:
                unmatched_images.append(image.name)
            else:
                pairs.append((image, mask))

        if duplicate_mask_keys:
            print(f"Warning: duplicate mask pairing keys found: {sorted(duplicate_mask_keys)[:10]}")

        if unmatched_images:
            print("Warning: some images did not find masks. First unmatched files:")
            for name in unmatched_images[:10]:
                print(f"  - {name}")

        if not pairs:
            raise ValueError(
                "strict_pairing=True but no image/mask pairs were found. "
                "Check that image and mask filenames share the same img number, e.g. img21."
            )

        return pairs

    n = min(len(image_files), len(mask_files))

    if n == 0:
        raise ValueError("Need at least one image and one mask.")

    print("Warning: strict_pairing=False. This is okay for demos, but not recommended for training.")
    return list(zip(image_files[:n], mask_files[:n]))


def load_rgb(path: str | Path, img_size: int) -> torch.Tensor:
    """Load an RGB image, resize it, normalize to [0, 1], and return CxHxW."""
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))

    return torch.from_numpy(arr)


def load_mask(
    path: str | Path,
    img_size: int,
    mask_threshold: int = 127,
    invert_mask: bool = False,
) -> torch.Tensor:
    """Load a binary road mask and return 1xHxW.

    Masks are resized with nearest-neighbor interpolation so thin road labels do
    not turn into blurry gray borders. Pixels above ``mask_threshold`` are road.

    Use ``invert_mask=True`` only if your dataset stores roads as black and
    background as white.
    """
    mask = Image.open(path).convert("L")
    mask = mask.resize((img_size, img_size), resample=Image.NEAREST)

    arr = np.asarray(mask, dtype=np.float32)
    arr = (arr > mask_threshold).astype(np.float32)

    if invert_mask:
        arr = 1.0 - arr

    return torch.from_numpy(arr[None, :, :])


class RoadSegmentationDataset(Dataset):
    """PyTorch dataset returning image, mask, and paths for debugging."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        img_size: int = 256,
        strict_pairing: bool = True,
        mask_threshold: int = 127,
        invert_mask: bool = False,
    ):
        self.img_size = int(img_size)
        self.mask_threshold = int(mask_threshold)
        self.invert_mask = bool(invert_mask)

        self.pairs = make_pairs(
            image_dir=image_dir,
            mask_dir=mask_dir,
            strict_pairing=strict_pairing,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]

        return {
            "image": load_rgb(image_path, self.img_size),
            "mask": load_mask(
                mask_path,
                self.img_size,
                self.mask_threshold,
                self.invert_mask,
            ),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def split_dataset(dataset: Dataset, val_split: float, seed: int) -> Tuple[Dataset, Dataset]:
    """Split into train/validation. Very tiny datasets reuse train as val."""
    total = len(dataset)

    if total < 4:
        return dataset, dataset

    val_size = max(1, int(round(total * val_split)))
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)

    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )