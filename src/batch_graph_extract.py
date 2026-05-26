"""Generate graphs for all predicted road masks.

Example:
    python -m src.batch_graph_extract --config configs/baseline.yaml

This searches outputs/predictions for *_pred_mask.png files and creates:
    outputs/graphs/*_graph.png
    outputs/graphs/*_graph.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .graph_extract import (
    load_binary_mask,
    mask_to_graph,
    save_graph_json,
    save_graph_visualization,
)
from .utils import ensure_dir, load_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--prediction_dir", default=None)
    parser.add_argument("--out_dir", default=None)

    parser.add_argument("--max_size", type=int, default=512)
    parser.add_argument("--min_component_area", type=int, default=30)
    parser.add_argument("--close_kernel_size", type=int, default=3)
    parser.add_argument("--spur_prune_iterations", type=int, default=8)
    parser.add_argument("--node_merge_radius", type=int, default=8)
    parser.add_argument("--node_expand_radius", type=int, default=3)
    parser.add_argument("--min_edge_length", type=int, default=15)
    parser.add_argument("--ignore_border", type=int, default=3)

    args = parser.parse_args()

    cfg = load_config(args.config)

    prediction_dir = Path(args.prediction_dir or cfg["output"]["prediction_dir"])
    out_dir = ensure_dir(args.out_dir or cfg["output"]["graph_dir"])

    mask_files = sorted(prediction_dir.glob("*_pred_mask.png"))

    if not mask_files:
        raise FileNotFoundError(
            f"No *_pred_mask.png files found in {prediction_dir}. "
            "Run predict.py first."
        )

    print(f"Found predicted masks: {len(mask_files)}")

    for mask_path in tqdm(mask_files):
        binary = load_binary_mask(
            mask_path,
            max_size=args.max_size if args.max_size > 0 else None,
        )

        skeleton, graph = mask_to_graph(
            binary=binary,
            min_component_area=args.min_component_area,
            close_kernel_size=args.close_kernel_size,
            spur_prune_iterations=args.spur_prune_iterations,
            node_merge_radius=args.node_merge_radius,
            node_expand_radius=args.node_expand_radius,
            min_edge_length=args.min_edge_length,
            ignore_border=args.ignore_border,
        )

        stem = mask_path.stem

        json_path = out_dir / f"{stem}_graph.json"
        png_path = out_dir / f"{stem}_graph.png"

        save_graph_json(graph, json_path)
        save_graph_visualization(skeleton, graph, png_path)

    print(f"Batch graph extraction complete. Saved to: {out_dir}")


if __name__ == "__main__":
    main()