"""Training entrypoint for cube visibility regression with ResNet18."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cube_detection.dataset import CLASS_NAMES, REGRESSION_TARGET_KEYS, create_dataloaders
from cube_detection.model import build_resnet18_regressor
from cube_detection.utils import suggest_lr


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    mae: float
    rmse: float
    visibility_acc: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_prediction_vector(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        return outputs.squeeze(1)
    if outputs.ndim == 1:
        return outputs
    raise ValueError(f"Expected model output shape [N] or [N,1], got {tuple(outputs.shape)}")


def run_train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    visibility_threshold: float,
) -> EpochMetrics:
    model.train()
    running_loss = 0.0
    running_abs_error = 0.0
    running_sq_error = 0.0
    running_visibility_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        predictions = _as_prediction_vector(outputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        diff = predictions - targets
        abs_error = torch.abs(diff)
        sq_error = diff * diff
        pred_visible = predictions >= visibility_threshold
        target_visible = targets >= visibility_threshold

        running_loss += float(loss.item()) * batch_size
        running_abs_error += float(abs_error.sum().item())
        running_sq_error += float(sq_error.sum().item())
        running_visibility_correct += int((pred_visible == target_visible).sum().item())
        total_samples += batch_size

    mean_loss = running_loss / max(total_samples, 1)
    mean_mae = running_abs_error / max(total_samples, 1)
    mean_rmse = float(np.sqrt(running_sq_error / max(total_samples, 1)))
    visibility_acc = running_visibility_correct / max(total_samples, 1)
    return EpochMetrics(loss=mean_loss, mae=mean_mae, rmse=mean_rmse, visibility_acc=visibility_acc)


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    visibility_threshold: float,
) -> EpochMetrics:
    model.eval()
    running_loss = 0.0
    running_abs_error = 0.0
    running_sq_error = 0.0
    running_visibility_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        predictions = _as_prediction_vector(outputs)
        loss = loss_fn(predictions, targets)

        batch_size = targets.shape[0]
        diff = predictions - targets
        abs_error = torch.abs(diff)
        sq_error = diff * diff
        pred_visible = predictions >= visibility_threshold
        target_visible = targets >= visibility_threshold

        running_loss += float(loss.item()) * batch_size
        running_abs_error += float(abs_error.sum().item())
        running_sq_error += float(sq_error.sum().item())
        running_visibility_correct += int((pred_visible == target_visible).sum().item())
        total_samples += batch_size

    mean_loss = running_loss / max(total_samples, 1)
    mean_mae = running_abs_error / max(total_samples, 1)
    mean_rmse = float(np.sqrt(running_sq_error / max(total_samples, 1)))
    visibility_acc = running_visibility_correct / max(total_samples, 1)
    return EpochMetrics(loss=mean_loss, mae=mean_mae, rmse=mean_rmse, visibility_acc=visibility_acc)


def main() -> None:
    # -----------------------
    # Editable runtime config
    # -----------------------
    DETECTION_DIR = None  # Example: "docs/Cube_Detection_dataset"
    OUTPUT_DIR = "runs/cube_detection"
    RUN_NAME = None

    TARGET_KEY = "visible_image_ratio"  # one of REGRESSION_TARGET_KEYS
    STRATIFY_BINS = 10

    EPOCHS = 20
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    IMAGE_SIZE = 224
    VAL_RATIO = 0.2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    LOSS_TYPE = "smooth_l1"  # options: "smooth_l1", "mse", "l1"
    USE_LR_FINDER = True
    APPLY_SUGGESTED_LR = True
    LR_FINDER_START_LR = 1e-6
    LR_FINDER_END_LR = 1e-1
    LR_FINDER_NUM_ITERS = 120
    AUTO_START_TENSORBOARD = True
    TENSORBOARD_HOST = "localhost"
    TENSORBOARD_PORT = 6006
    SEED = 42
    USE_PRETRAINED = True
    FREEZE_BACKBONE = False
    BOUNDED_OUTPUT = True
    VISIBILITY_THRESHOLD = 0.01

    if TARGET_KEY not in REGRESSION_TARGET_KEYS:
        raise ValueError(
            f"TARGET_KEY must be one of {REGRESSION_TARGET_KEYS}, got {TARGET_KEY!r}"
        )
    if TARGET_KEY == "edge_margin" and BOUNDED_OUTPUT:
        raise ValueError(
            "TARGET_KEY='edge_margin' requires BOUNDED_OUTPUT=False because edge_margin can be negative."
        )

    set_seed(SEED)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("TensorBoard support requires `tensorboard`. Install with: pip install tensorboard") from exc

    dataloaders = create_dataloaders(
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        detection_dir=DETECTION_DIR,
        target_key=TARGET_KEY,
        stratify_bins=STRATIFY_BINS,
    )

    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(OUTPUT_DIR) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_regressor(
        pretrained=USE_PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
        bounded_output=BOUNDED_OUTPUT,
    ).to(device)

    if LOSS_TYPE == "smooth_l1":
        loss_fn = nn.SmoothL1Loss()
    elif LOSS_TYPE == "mse":
        loss_fn = nn.MSELoss()
    elif LOSS_TYPE == "l1":
        loss_fn = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported LOSS_TYPE: {LOSS_TYPE!r}")

    if USE_LR_FINDER:
        lr_finder_plot_path = run_dir / "lr_finder.png"
        lr_finder_result = suggest_lr(
            model=model,
            loader=dataloaders.train_loader,
            loss_fn=loss_fn,
            device=device,
            start_lr=LR_FINDER_START_LR,
            end_lr=LR_FINDER_END_LR,
            num_iters=LR_FINDER_NUM_ITERS,
            weight_decay=WEIGHT_DECAY,
            output_path=lr_finder_plot_path,
        )
        print(f"LR finder suggested learning rate: {lr_finder_result.suggested_lr:.6g}")
        if lr_finder_result.plot_path is not None:
            print(f"LR finder plot: {lr_finder_result.plot_path.resolve()}")
        if APPLY_SUGGESTED_LR:
            LEARNING_RATE = lr_finder_result.suggested_lr
            print(f"Using suggested learning rate: {LEARNING_RATE:.6g}")

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("data/detection_dir", str(dataloaders.detection_dir.resolve()))
    writer.add_text("data/target_key", TARGET_KEY)
    writer.add_scalar(f"class_count/{CLASS_NAMES[0]}", dataloaders.class_counts[CLASS_NAMES[0]], 0)
    writer.add_scalar(f"class_count/{CLASS_NAMES[1]}", dataloaders.class_counts[CLASS_NAMES[1]], 0)
    writer.add_scalar("split/train_size", dataloaders.train_size, 0)
    writer.add_scalar("split/val_size", dataloaders.val_size, 0)
    writer.add_scalar("target/min", dataloaders.target_stats["min"], 0)
    writer.add_scalar("target/max", dataloaders.target_stats["max"], 0)
    writer.add_scalar("target/mean", dataloaders.target_stats["mean"], 0)
    writer.add_scalar("target/std", dataloaders.target_stats["std"], 0)
    if USE_LR_FINDER:
        writer.add_scalar("train/lr_finder_suggested_lr", LEARNING_RATE, 0)
        writer.add_text("train/lr_finder_plot_path", str((run_dir / "lr_finder.png").resolve()))
    writer.flush()

    print(f"Using device: {device}")
    print(f"Train samples: {dataloaders.train_size} | Val samples: {dataloaders.val_size}")
    print(f"Class counts (derived from filename label): {dataloaders.class_counts}")
    print(f"Target key: {TARGET_KEY} | stats: {dataloaders.target_stats}")
    print(f"TensorBoard logdir: {tensorboard_dir.resolve()}")

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
            print(
                f"You can start it manually with: "
                f"tensorboard --logdir \"{tensorboard_dir.resolve()}\" --host {TENSORBOARD_HOST} --port {TENSORBOARD_PORT}"
            )

    best_val_rmse = float("inf")
    best_ckpt_path = checkpoint_dir / "best.pt"
    last_ckpt_path = checkpoint_dir / "last.pt"

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_train_epoch(
            model=model,
            loader=dataloaders.train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            visibility_threshold=VISIBILITY_THRESHOLD,
        )
        val_metrics = run_eval_epoch(
            model=model,
            loader=dataloaders.val_loader,
            loss_fn=loss_fn,
            device=device,
            visibility_threshold=VISIBILITY_THRESHOLD,
        )
        scheduler.step()

        writer.add_scalar("train/loss", train_metrics.loss, epoch)
        writer.add_scalar("val/loss", val_metrics.loss, epoch)
        writer.add_scalar("train/mae", train_metrics.mae, epoch)
        writer.add_scalar("val/mae", val_metrics.mae, epoch)
        writer.add_scalar("train/rmse", train_metrics.rmse, epoch)
        writer.add_scalar("val/rmse", val_metrics.rmse, epoch)
        writer.add_scalar("train/visibility_acc", train_metrics.visibility_acc, epoch)
        writer.add_scalar("val/visibility_acc", val_metrics.visibility_acc, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_metrics.loss,
            "val_loss": val_metrics.loss,
            "train_mae": train_metrics.mae,
            "val_mae": val_metrics.mae,
            "train_rmse": train_metrics.rmse,
            "val_rmse": val_metrics.rmse,
            "train_visibility_acc": train_metrics.visibility_acc,
            "val_visibility_acc": val_metrics.visibility_acc,
            "best_val_rmse": min(best_val_rmse, val_metrics.rmse),
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "target_key": TARGET_KEY,
            "loss_type": LOSS_TYPE,
            "bounded_output": BOUNDED_OUTPUT,
            "visibility_threshold": VISIBILITY_THRESHOLD,
            "detection_dir": str(dataloaders.detection_dir.resolve()),
            "class_counts": dataloaders.class_counts,
            "target_stats": dataloaders.target_stats,
        }
        torch.save(checkpoint, last_ckpt_path)

        if val_metrics.rmse < best_val_rmse:
            best_val_rmse = val_metrics.rmse
            checkpoint["best_val_rmse"] = best_val_rmse
            torch.save(checkpoint, best_ckpt_path)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.6f} "
            f"val_loss={val_metrics.loss:.6f} "
            f"train_mae={train_metrics.mae:.6f} "
            f"val_mae={val_metrics.mae:.6f} "
            f"train_rmse={train_metrics.rmse:.6f} "
            f"val_rmse={val_metrics.rmse:.6f} "
            f"train_vis_acc={train_metrics.visibility_acc:.4f} "
            f"val_vis_acc={val_metrics.visibility_acc:.4f}"
        )

    writer.close()
    print(f"Training complete. Best val_rmse={best_val_rmse:.6f}")
    print(f"Best checkpoint: {best_ckpt_path.resolve()}")
    print(f"TensorBoard logs: {tensorboard_dir.resolve()}")
    if tensorboard_url is not None:
        print(f"TensorBoard URL: {tensorboard_url}")


if __name__ == "__main__":
    main()
