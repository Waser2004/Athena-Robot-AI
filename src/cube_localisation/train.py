"""Training entrypoint for cube localisation finetuning."""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Use a non-GUI matplotlib backend for training-time plot exports on desktop/worker threads.
os.environ.setdefault("MPLBACKEND", "Agg")

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
    SpatialRegion,
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
from cube_localisation.utils import suggest_lr


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


@dataclass
class EvalSummary:
    loss: float
    mae_m: float
    mae_dim_m: np.ndarray
    mean_distance_offset_m: float | None
    mean_rotation_offset_deg: float | None
    sample_target_xy_m: np.ndarray | None = None
    sample_distance_offsets_m: np.ndarray | None = None
    sample_rotation_offsets_deg: np.ndarray | None = None


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
    target_keys: Sequence[str],
    collect_samples: bool = False,
) -> EvalSummary:
    model.eval()
    running_loss = 0.0
    running_abs_error = 0.0
    running_abs_error_dim: np.ndarray | None = None
    running_distance_offset = 0.0
    running_rotation_offset_deg = 0.0
    distance_count = 0
    rotation_count = 0
    total_items = 0

    keys = list(target_keys)
    idx_x = keys.index("cube_x_m") if "cube_x_m" in keys else None
    idx_y = keys.index("cube_y_m") if "cube_y_m" in keys else None
    idx_rot_sin = keys.index("cube_z_rotation_sin4") if "cube_z_rotation_sin4" in keys else None
    idx_rot_cos = keys.index("cube_z_rotation_cos4") if "cube_z_rotation_cos4" in keys else None

    sample_target_xy_m: list[np.ndarray] = []
    sample_distance_offsets_m: list[np.ndarray] = []
    sample_rotation_offsets_deg: list[np.ndarray] = []

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

        predictions_np = predictions_m.detach().cpu().numpy()
        targets_np = targets_m.detach().cpu().numpy()

        if idx_x is not None and idx_y is not None:
            delta_xy = predictions_np[:, [idx_x, idx_y]] - targets_np[:, [idx_x, idx_y]]
            batch_distance_offsets = np.linalg.norm(delta_xy, axis=1)
            running_distance_offset += float(batch_distance_offsets.sum())
            distance_count += int(batch_distance_offsets.size)
            if collect_samples:
                sample_target_xy_m.append(targets_np[:, [idx_x, idx_y]])
                sample_distance_offsets_m.append(batch_distance_offsets)

        if idx_rot_sin is not None and idx_rot_cos is not None:
            pred_angle4 = np.arctan2(predictions_np[:, idx_rot_sin], predictions_np[:, idx_rot_cos])
            target_angle4 = np.arctan2(targets_np[:, idx_rot_sin], targets_np[:, idx_rot_cos])
            delta_angle4 = np.arctan2(np.sin(pred_angle4 - target_angle4), np.cos(pred_angle4 - target_angle4))
            batch_rotation_offsets_deg = np.degrees(np.abs(delta_angle4) / 4.0)
            running_rotation_offset_deg += float(batch_rotation_offsets_deg.sum())
            rotation_count += int(batch_rotation_offsets_deg.size)
            if collect_samples:
                sample_rotation_offsets_deg.append(batch_rotation_offsets_deg)

        total_items += batch_size

    avg_loss = running_loss / max(total_items, 1)
    mae_m = running_abs_error / max(total_items, 1)
    mae_dim_m = (
        np.zeros_like(target_mean, dtype=np.float64)
        if running_abs_error_dim is None
        else running_abs_error_dim / max(total_items, 1)
    )
    mean_distance_offset_m = running_distance_offset / distance_count if distance_count > 0 else None
    mean_rotation_offset_deg = running_rotation_offset_deg / rotation_count if rotation_count > 0 else None

    collected_target_xy = np.concatenate(sample_target_xy_m, axis=0) if sample_target_xy_m else None
    collected_distance_offsets = np.concatenate(sample_distance_offsets_m, axis=0) if sample_distance_offsets_m else None
    collected_rotation_offsets = (
        np.concatenate(sample_rotation_offsets_deg, axis=0) if sample_rotation_offsets_deg else None
    )

    return EvalSummary(
        loss=avg_loss,
        mae_m=mae_m,
        mae_dim_m=mae_dim_m,
        mean_distance_offset_m=mean_distance_offset_m,
        mean_rotation_offset_deg=mean_rotation_offset_deg,
        sample_target_xy_m=collected_target_xy,
        sample_distance_offsets_m=collected_distance_offsets,
        sample_rotation_offsets_deg=collected_rotation_offsets,
    )


def save_error_visualization(
    *,
    target_xy_m: np.ndarray,
    distance_offsets_m: np.ndarray,
    rotation_offsets_deg: np.ndarray,
    output_path: Path,
    epoch: int,
    val_region: SpatialRegion | None = None,
    test_region: SpatialRegion | None = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError(
            "Error visualization requires matplotlib. Install with: pip install matplotlib"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    distance_offsets_cm = distance_offsets_m * 100.0
    max_distance_cm = max(float(np.max(distance_offsets_cm)) if distance_offsets_cm.size else 0.0, 1e-6)
    marker_sizes = 24.0 + 240.0 * np.clip(distance_offsets_cm / max_distance_cm, 0.0, 1.0)

    fig, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        target_xy_m[:, 0],
        target_xy_m[:, 1],
        s=marker_sizes,
        c=rotation_offsets_deg,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=45.0,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.25,
    )
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Rotation offset (deg)")

    if val_region is not None:
        axis.add_patch(
            Rectangle(
                (val_region.x_min, val_region.y_min),
                val_region.x_max - val_region.x_min,
                val_region.y_max - val_region.y_min,
                linewidth=2,
                edgecolor="#ff7f0e",
                facecolor="none",
                linestyle="--",
                label="val region",
            )
        )
    if test_region is not None:
        axis.add_patch(
            Rectangle(
                (test_region.x_min, test_region.y_min),
                test_region.x_max - test_region.x_min,
                test_region.y_max - test_region.y_min,
                linewidth=2,
                edgecolor="#2ca02c",
                facecolor="none",
                linestyle="--",
                label="test region",
            )
        )

    axis.set_title(f"All-samples error map (epoch {epoch})")
    axis.set_xlabel("cube_x_m")
    axis.set_ylabel("cube_y_m")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    if val_region is not None or test_region is not None:
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    # -----------------------
    # Editable runtime config
    # -----------------------
    DATASET_DIR = None  # Example: "docs/Cube_localisation_dataset"
    OUTPUT_DIR = "runs/cube_localisation"
    RUN_NAME = None

    TARGET_KEYS = ["cube_x_m", "cube_y_m", "cube_z_rotation_sin4", "cube_z_rotation_cos4"]
    BACKBONE = "resnet34"
    USE_PRETRAINED = True
    DROPOUT = 0.1
    JOINT_HIDDEN_DIM = 64

    EPOCHS = 100
    ERROR_VIS_INTERVAL_EPOCHS = 5
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    IMAGE_SIZE = 224
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    USE_LR_FINDER = True
    APPLY_SUGGESTED_LR = True
    LR_FINDER_START_LR = 1e-4
    LR_FINDER_END_LR = 1e-1
    LR_FINDER_NUM_ITERS = 120

    AUTO_START_TENSORBOARD = True
    TENSORBOARD_HOST = "0.0.0.0"
    TENSORBOARD_PORT = 6006

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
    all_samples_dataset = CubeLocalisationDataset(
        records=records,
        indices=tuple(range(len(records))),
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
    all_samples_loader = DataLoader(
        all_samples_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(OUTPUT_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = run_dir / "tensorboard"
    error_maps_dir = run_dir / "error_maps"
    checkpoints_dir = run_dir / "checkpoints"
    error_maps_dir.mkdir(parents=True, exist_ok=True)
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
        joint_mean=joint_mean,
        joint_std=joint_std,
    ).to(device)

    loss_fn = nn.MSELoss()
    if USE_LR_FINDER:
        lr_finder_plot_path = run_dir / "lr_finder.png"
        lr_finder_result = suggest_lr(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            start_lr=LR_FINDER_START_LR,
            end_lr=LR_FINDER_END_LR,
            num_iters=LR_FINDER_NUM_ITERS,
            weight_decay=WEIGHT_DECAY,
            output_path=lr_finder_plot_path,
        )
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        writer.add_text("train/lr_finder_plot_path", str(lr_finder_result.plot_path.resolve()) if lr_finder_result.plot_path else "")
        writer.add_scalar("train/lr_finder_suggested_lr", lr_finder_result.suggested_lr, 0)
        writer.flush()
        writer.close()
        print(f"LR finder suggested learning rate: {lr_finder_result.suggested_lr:.6g}")
        if lr_finder_result.plot_path is not None:
            print(f"LR finder plot: {lr_finder_result.plot_path}")
        if APPLY_SUGGESTED_LR:
            LEARNING_RATE = lr_finder_result.suggested_lr
            print(f"Using suggested learning rate: {LEARNING_RATE:.6g}")

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
    writer.flush()

    tensorboard_url: str | None = None
    if AUTO_START_TENSORBOARD:
        try:
            from tensorboard import program as tensorboard_program

            tensorboard_server = tensorboard_program.TensorBoard()
            tensorboard_server.configure(
                argv=[
                    None,
                    "--logdir",
                    str(tensorboard_dir.resolve()),
                    "--host",
                    TENSORBOARD_HOST,
                    "--port",
                    str(TENSORBOARD_PORT),
                ]
            )
            tensorboard_url = tensorboard_server.launch()
            print(f"TensorBoard available at: {tensorboard_url}")
        except Exception as exc:
            print(f"Warning: Failed to launch TensorBoard automatically: {exc}")
            print(f"You can start it manually with: tensorboard --logdir \"{tensorboard_dir.resolve()}\"")

    best_val_mae_m = float("inf")
    best_ckpt_path = checkpoints_dir / "best.pt"
    last_ckpt_path = checkpoints_dir / "last.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch_train(model, train_loader, loss_fn, optimizer, device)
        collect_error_samples = ERROR_VIS_INTERVAL_EPOCHS > 0 and (
            (epoch % ERROR_VIS_INTERVAL_EPOCHS == 0) or (epoch == EPOCHS)
        )
        eval_summary = run_epoch_eval(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            target_keys=TARGET_KEYS,
            collect_samples=collect_error_samples,
        )
        val_loss = eval_summary.loss
        val_mae_m = eval_summary.mae_m
        val_mae_dim_m = eval_summary.mae_dim_m
        scheduler.step()

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/mae_m", val_mae_m, epoch)
        writer.add_scalar("val/mae_cm", val_mae_m * 100.0, epoch)
        if eval_summary.mean_distance_offset_m is not None:
            writer.add_scalar("val/mean_distance_offset_m", eval_summary.mean_distance_offset_m, epoch)
            writer.add_scalar("val/mean_distance_offset_cm", eval_summary.mean_distance_offset_m * 100.0, epoch)
        if eval_summary.mean_rotation_offset_deg is not None:
            writer.add_scalar("val/mean_rotation_offset_deg", eval_summary.mean_rotation_offset_deg, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        for idx, key in enumerate(TARGET_KEYS):
            writer.add_scalar(f"val/mae_{key}_m", float(val_mae_dim_m[idx]), epoch)
            writer.add_scalar(f"val/mae_{key}_cm", float(val_mae_dim_m[idx] * 100.0), epoch)
        writer.flush()

        if collect_error_samples:
            all_samples_summary = run_epoch_eval(
                model=model,
                loader=all_samples_loader,
                loss_fn=loss_fn,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
                target_keys=TARGET_KEYS,
                collect_samples=True,
            )
            if (
                all_samples_summary.sample_target_xy_m is not None
                and all_samples_summary.sample_distance_offsets_m is not None
                and all_samples_summary.sample_rotation_offsets_deg is not None
            ):
                error_map_path = error_maps_dir / f"all_samples_error_epoch_{epoch:03d}.png"
                save_error_visualization(
                    target_xy_m=all_samples_summary.sample_target_xy_m,
                    distance_offsets_m=all_samples_summary.sample_distance_offsets_m,
                    rotation_offsets_deg=all_samples_summary.sample_rotation_offsets_deg,
                    output_path=error_map_path,
                    epoch=epoch,
                    val_region=split.val_region,
                    test_region=split.test_region,
                )
                writer.add_text("all_samples/error_map_path", str(error_map_path.resolve()), epoch)
                print(f"Saved all-samples error map: {error_map_path}")
            else:
                print(
                    "Skipped error map: required targets are missing "
                    "(need cube_x_m, cube_y_m, cube_z_rotation_sin4, cube_z_rotation_cos4)."
                )

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

        distance_offset_str = (
            f"mean_distance_offset_cm={eval_summary.mean_distance_offset_m * 100.0:.3f}"
            if eval_summary.mean_distance_offset_m is not None
            else "mean_distance_offset_cm=n/a"
        )
        rotation_offset_str = (
            f"mean_rotation_offset_deg={eval_summary.mean_rotation_offset_deg:.3f}"
            if eval_summary.mean_rotation_offset_deg is not None
            else "mean_rotation_offset_deg=n/a"
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_mae_cm={val_mae_m * 100.0:.3f} "
            f"{distance_offset_str} "
            f"{rotation_offset_str}"
        )

    writer.close()
    print(f"Training complete. Best validation MAE: {best_val_mae_m * 100.0:.3f} cm")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    if tensorboard_url is not None:
        print(f"TensorBoard URL: {tensorboard_url}")
    print(f"Error maps: {error_maps_dir}")
    print(f"Spatial split file: {split_path}")


if __name__ == "__main__":
    main()
