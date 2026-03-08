"""Training entrypoint for cube localisation finetuning."""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cube_localisation.dataset import (
    CubeLocalisationDataset,
    SpatialSplitConfig,
    build_spatial_split,
    build_split_from_regions,
    compute_joint_stats,
    compute_target_stats,
    load_records,
    load_split_definition,
    resolve_dataset_dir,
    save_split_definition,
)
from cube_localisation.model import build_localisation_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _denormalize_targets(
    values: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, dtype=values.dtype, device=device)
    std_t = torch.as_tensor(std, dtype=values.dtype, device=device)
    return values * std_t + mean_t


def run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_items = 0

    for images, joint_inputs, targets in loader:
        images = images.to(device, non_blocking=True)
        joint_inputs = joint_inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images, joint_inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        running_loss += loss.item() * batch_size
        total_items += batch_size

    return running_loss / max(total_items, 1)


@torch.no_grad()
def run_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    running_loss = 0.0
    running_abs_error = 0.0
    running_abs_error_dim: np.ndarray | None = None
    total_items = 0

    for images, joint_inputs, targets in loader:
        images = images.to(device, non_blocking=True)
        joint_inputs = joint_inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(images, joint_inputs)
        loss = loss_fn(predictions, targets)

        predictions_m = _denormalize_targets(predictions, target_mean, target_std, device)
        targets_m = _denormalize_targets(targets, target_mean, target_std, device)
        abs_error = (predictions_m - targets_m).abs()

        batch_size = images.shape[0]
        running_loss += loss.item() * batch_size
        running_abs_error += abs_error.mean(dim=1).sum().item()

        per_dim = abs_error.sum(dim=0).detach().cpu().numpy()
        running_abs_error_dim = per_dim if running_abs_error_dim is None else running_abs_error_dim + per_dim
        total_items += batch_size

    avg_loss = running_loss / max(total_items, 1)
    mae_m = running_abs_error / max(total_items, 1)
    mae_dim_m = (
        np.zeros_like(target_mean, dtype=np.float64)
        if running_abs_error_dim is None
        else running_abs_error_dim / max(total_items, 1)
    )
    return avg_loss, mae_m, mae_dim_m


def main() -> None:
    # -----------------------
    # Editable runtime config
    # -----------------------
    DATASET_DIR = None  # Example: "docs/Cube_localisation_dataset"
    OUTPUT_DIR = "runs/cube_localisation"
    RUN_NAME = None

    TARGET_KEYS = ["cube_x_m", "cube_y_m", "cube_z_rotation_sin4", "cube_z_rotation_cos4"]
    BACKBONE = "resnet18"
    USE_PRETRAINED = True
    DROPOUT = 0.1
    JOINT_HIDDEN_DIM = 64

    EPOCHS = 100
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    IMAGE_SIZE = 224
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    SEED = 100
    VAL_REGION_RATIO = 0.25
    TEST_REGION_RATIO = 0.25
    ALLOW_REGION_OVERLAP = False
    SPLIT_FILE = None

    set_seed(SEED)

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("TensorBoard support requires `tensorboard`. Install with: pip install tensorboard") from exc

    dataset_dir = resolve_dataset_dir(DATASET_DIR)
    records = load_records(dataset_dir)

    if SPLIT_FILE:
        val_region, test_region = load_split_definition(SPLIT_FILE)
        split = build_split_from_regions(records, val_region=val_region, test_region=test_region)
        split_config = SpatialSplitConfig(
            seed=SEED,
            val_region_ratio=VAL_REGION_RATIO,
            test_region_ratio=TEST_REGION_RATIO,
            ensure_non_overlapping_regions=not ALLOW_REGION_OVERLAP,
        )
    else:
        split_config = SpatialSplitConfig(
            seed=SEED,
            val_region_ratio=VAL_REGION_RATIO,
            test_region_ratio=TEST_REGION_RATIO,
            ensure_non_overlapping_regions=not ALLOW_REGION_OVERLAP,
        )
        split = build_spatial_split(records, split_config)

    joint_dims = {len(records[idx].joint_rotations_rad) for idx in split.train_indices}
    if not joint_dims or len(joint_dims) != 1:
        raise RuntimeError(f"Inconsistent train-set joint vector lengths: {sorted(joint_dims)}")
    joint_input_dim = joint_dims.pop()
    if joint_input_dim <= 0:
        raise RuntimeError("joint_rotations_rad are missing in train split. They are required as model inputs.")

    target_mean, target_std = compute_target_stats(records, split.train_indices, TARGET_KEYS)
    joint_mean, joint_std = compute_joint_stats(records, split.train_indices)

    train_dataset = CubeLocalisationDataset(
        records=records,
        indices=split.train_indices,
        target_keys=TARGET_KEYS,
        image_size=IMAGE_SIZE,
        augment=True,
        joint_input_dim=joint_input_dim,
        joint_mean=joint_mean,
        joint_std=joint_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_dataset = CubeLocalisationDataset(
        records=records,
        indices=split.val_indices,
        target_keys=TARGET_KEYS,
        image_size=IMAGE_SIZE,
        augment=False,
        joint_input_dim=joint_input_dim,
        joint_mean=joint_mean,
        joint_std=joint_std,
        target_mean=target_mean,
        target_std=target_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(OUTPUT_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = run_dir / "tensorboard"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    split_path = save_split_definition(run_dir / "spatial_split.json", split, split_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_localisation_model(
        output_dim=len(TARGET_KEYS),
        backbone=BACKBONE,
        pretrained=USE_PRETRAINED,
        dropout=DROPOUT,
        joint_input_dim=joint_input_dim,
        joint_hidden_dim=JOINT_HIDDEN_DIM,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("data/dataset_dir", str(dataset_dir.resolve()))
    writer.add_text("data/targets", ", ".join(TARGET_KEYS))
    writer.add_scalar("data/joint_input_dim", joint_input_dim, 0)
    writer.add_text("split/val_region", json.dumps(split.val_region.to_dict()))
    writer.add_text("split/test_region", json.dumps(split.test_region.to_dict()))
    writer.add_text("split/split_file", str(split_path.resolve()))
    writer.add_scalar("split/train_size", len(split.train_indices), 0)
    writer.add_scalar("split/val_size", len(split.val_indices), 0)
    writer.add_scalar("split/test_size", len(split.test_indices), 0)

    best_val_mae_m = float("inf")
    best_ckpt_path = checkpoints_dir / "best.pt"
    last_ckpt_path = checkpoints_dir / "last.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch_train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_mae_m, val_mae_dim_m = run_epoch_eval(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        scheduler.step()

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/mae_m", val_mae_m, epoch)
        writer.add_scalar("val/mae_cm", val_mae_m * 100.0, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        for idx, key in enumerate(TARGET_KEYS):
            writer.add_scalar(f"val/mae_{key}_m", float(val_mae_dim_m[idx]), epoch)
            writer.add_scalar(f"val/mae_{key}_cm", float(val_mae_dim_m[idx] * 100.0), epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mae_m": best_val_mae_m,
            "backbone": BACKBONE,
            "use_pretrained": USE_PRETRAINED,
            "target_keys": list(TARGET_KEYS),
            "joint_input_dim": joint_input_dim,
            "joint_hidden_dim": JOINT_HIDDEN_DIM,
            "joint_mean": joint_mean.tolist(),
            "joint_std": joint_std.tolist(),
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "image_size": IMAGE_SIZE,
            "dataset_dir": str(dataset_dir.resolve()),
            "split": split.to_dict(),
        }
        torch.save(checkpoint, last_ckpt_path)

        if val_mae_m < best_val_mae_m:
            best_val_mae_m = val_mae_m
            checkpoint["best_val_mae_m"] = best_val_mae_m
            torch.save(checkpoint, best_ckpt_path)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_mae_cm={val_mae_m * 100.0:.3f}"
        )

    writer.close()
    print(f"Training complete. Best validation MAE: {best_val_mae_m * 100.0:.3f} cm")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Spatial split file: {split_path}")


if __name__ == "__main__":
    main()
