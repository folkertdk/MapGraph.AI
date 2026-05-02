"""Shared helper functions for MapGraph.AI.

The project is deliberately written with small, readable utilities instead of a
large framework. This makes it easier to explain in class and easier to modify
when the project evolves from a baseline segmentation model to graph extraction
and then Mixture-of-Experts.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file as a Python dictionary.

    Args:
        path: Location of the YAML file, for example ``configs/baseline.yaml``.

    Returns:
        Nested dictionary containing data, model, training, and output settings.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Make training and data splitting as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a folder if it does not already exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> torch.device:
    """Use GPU when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
