from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import RoadSegmentationDataset
from .model import build_model
from .train import dice_score_from_logits, iou_score_from_logits, parse_model_output
from .utils import get_device, load_config


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def evaluate_city(model, city, image_dir, mask_dir, cfg, device, checkpoint_epoch):
    dataset = RoadSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_size=cfg["data"].get("img_size", 256),
        strict_pairing=cfg["data"].get("strict_pairing", True),
        mask_threshold=cfg["data"].get("mask_threshold", 127),
        invert_mask=cfg["data"].get("invert_mask", False),
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    stats = defaultdict(list)

    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            output = model(images)
            final_logits, _, diagnostics = parse_model_output(output)

            router_probs = output[1].detach().cpu()[0]

            stats["final_dice"].append(dice_score_from_logits(final_logits, masks))
            stats["final_iou"].append(iou_score_from_logits(final_logits, masks))

            for expert_idx, expert in enumerate(model.experts):
                expert_logits = expert(images)

                stats[f"expert_{expert_idx}_dice"].append(
                    dice_score_from_logits(expert_logits, masks)
                )
                stats[f"expert_{expert_idx}_iou"].append(
                    iou_score_from_logits(expert_logits, masks)
                )
                stats[f"expert_{expert_idx}_router_weight"].append(
                    float(router_probs[expert_idx])
                )

    row = {
        "city": city,
        "checkpoint_epoch": checkpoint_epoch,
        "num_samples": len(dataset),
        "final_dice": mean(stats["final_dice"]),
        "final_iou": mean(stats["final_iou"]),
    }

    num_experts = len(model.experts)

    for expert_idx in range(num_experts):
        row[f"expert_{expert_idx}_dice"] = mean(stats[f"expert_{expert_idx}_dice"])
        row[f"expert_{expert_idx}_iou"] = mean(stats[f"expert_{expert_idx}_iou"])
        row[f"expert_{expert_idx}_router_weight"] = mean(
            stats[f"expert_{expert_idx}_router_weight"]
        )

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/moe_mixed.yaml")
    parser.add_argument("--checkpoint", default="outputs/moe_mixed/best.pt")
    parser.add_argument("--out", default="outputs/moe_expert_analysis.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    checkpoint_epoch = checkpoint.get("epoch", -1)

    cities = {
        "Khartoum": {
            "image_dir": "data/spacenet3/8bit/Khartoum/images",
            "mask_dir": "data/spacenet3/8bit/Khartoum/mask",
        },
        "Paris": {
            "image_dir": "data/spacenet3/8bit/Paris/images",
            "mask_dir": "data/spacenet3/8bit/Paris/mask",
        },
        "Shanghai": {
            "image_dir": "data/spacenet3/8bit/Shanghai/images",
            "mask_dir": "data/spacenet3/8bit/Shanghai/mask",
        },
        "Vegas": {
            "image_dir": "data/spacenet3/8bit/Vegas/images",
            "mask_dir": "data/spacenet3/8bit/Vegas/mask",
        },
    }

    rows = []

    for city, paths in cities.items():
        print(f"Analyzing {city}...")
        rows.append(
            evaluate_city(
                model=model,
                city=city,
                image_dir=paths["image_dir"],
                mask_dir=paths["mask_dir"],
                cfg=checkpoint["config"],
                device=device,
                checkpoint_epoch=checkpoint_epoch,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved analysis to {out_path}")

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()