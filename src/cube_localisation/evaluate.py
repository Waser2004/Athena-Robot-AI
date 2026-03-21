"""Evaluation script for trained cube localisation models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cube_localisation.dataset import (
    CubeLocalisationDataset,
    SpatialRegion,
    build_split_from_regions,
    load_records,
    load_split_definition,
    resolve_dataset_dir,
)
from cube_localisation.model import build_localisation_model
from cube_localisation.train import run_epoch_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cube localisation checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Optional split json. If omitted, uses split regions stored in checkpoint.",
    )
    return parser.parse_args()


def _load_regions_from_checkpoint(ckpt: dict) -> tuple[SpatialRegion, SpatialRegion]:
    split_payload = ckpt.get("split")
    if not split_payload:
        raise RuntimeError("Checkpoint has no split metadata. Provide --split-file.")
    val_region = SpatialRegion.from_dict(split_payload["val_region"])
    test_region = SpatialRegion.from_dict(split_payload["test_region"])
    return val_region, test_region


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    dataset_dir = resolve_dataset_dir(args.dataset_dir or checkpoint.get("dataset_dir"))
    records = load_records(dataset_dir)

    if args.split_file:
        val_region, test_region = load_split_definition(args.split_file)
    else:
        val_region, test_region = _load_regions_from_checkpoint(checkpoint)

    split = build_split_from_regions(records, val_region=val_region, test_region=test_region)
    indices = split.test_indices if args.split == "test" else split.val_indices

    target_keys = checkpoint["target_keys"]
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    joint_mean_raw = checkpoint.get("joint_mean")
    joint_std_raw = checkpoint.get("joint_std")
    joint_mean = np.asarray(joint_mean_raw, dtype=np.float32) if joint_mean_raw is not None else None
    joint_std = np.asarray(joint_std_raw, dtype=np.float32) if joint_std_raw is not None else None
    image_size = int(checkpoint.get("image_size", 224))
    backbone = checkpoint["backbone"]
    fallback_joint_dim = len(records[indices[0]].joint_rotations_rad)
    joint_input_dim = int(checkpoint.get("joint_input_dim", fallback_joint_dim))
    joint_hidden_dim = int(checkpoint.get("joint_hidden_dim", 64))

    dataset = CubeLocalisationDataset(
        records=records,
        indices=indices,
        target_keys=target_keys,
        image_size=image_size,
        augment=False,
        joint_input_dim=joint_input_dim,
        joint_mean=joint_mean,
        joint_std=joint_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_localisation_model(
        output_dim=len(target_keys),
        backbone=backbone,
        pretrained=False,
        joint_input_dim=joint_input_dim,
        joint_hidden_dim=joint_hidden_dim,
        joint_mean=joint_mean,
        joint_std=joint_std,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss_fn = nn.MSELoss()
    eval_summary = run_epoch_eval(
        model,
        loader,
        loss_fn,
        device,
        target_mean=target_mean,
        target_std=target_std,
        target_keys=target_keys,
        collect_samples=False,
    )
    loss = eval_summary.loss
    mae_m = eval_summary.mae_m
    mae_dim_m = eval_summary.mae_dim_m

    print(f"split={args.split} samples={len(indices)}")
    print(f"loss={loss:.6f}")
    print(f"mae_m={mae_m:.6f}")
    print(f"mae_cm={mae_m * 100.0:.3f}")
    if eval_summary.mean_distance_offset_m is not None:
        print(f"mean_distance_offset_m={eval_summary.mean_distance_offset_m:.6f}")
        print(f"mean_distance_offset_cm={eval_summary.mean_distance_offset_m * 100.0:.3f}")
    if eval_summary.mean_rotation_offset_deg is not None:
        print(f"mean_rotation_offset_deg={eval_summary.mean_rotation_offset_deg:.3f}")
    for idx, key in enumerate(target_keys):
        print(f"mae_{key}_m={mae_dim_m[idx]:.6f}")
        print(f"mae_{key}_cm={mae_dim_m[idx] * 100.0:.3f}")


if __name__ == "__main__":
    main()
