"""Simple training entrypoint for cube visibility regression."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cube_detection.dataset import IMAGENET_MEAN, IMAGENET_STD, REGRESSION_TARGET_KEYS, create_dataloaders
from cube_detection.model import build_simple_cnn_regressor
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


def _log_random_train_grid(
    loader: torch.utils.data.DataLoader,
    run_dir: Path,
    writer: object,
    image_size: int,
    grid_size: int = 8,
) -> None:
    """Create and log an NxN grid of random training samples."""
    try:
        from PIL import ImageDraw
        from torchvision.transforms import functional as tvf
        from torchvision.utils import make_grid
    except ImportError as exc:
        raise ImportError(
            "Grid visualization requires torchvision. Install with: pip install torchvision"
        ) from exc

    dataset = loader.dataset
    dataset_size = len(dataset)
    if dataset_size <= 0:
        print("Warning: Train dataset is empty; skipping train sample grid.")
        return

    total_tiles = grid_size * grid_size
    if dataset_size >= total_tiles:
        selected_indices = random.sample(range(dataset_size), k=total_tiles)
    else:
        selected_indices = [random.randrange(dataset_size) for _ in range(total_tiles)]

    images: list[torch.Tensor] = []
    source_labels: list[str] = []
    for sample_index in selected_indices:
        sample_image, _ = dataset[sample_index]
        images.append(sample_image.detach().cpu())
        record_index = int(dataset.indices[sample_index]) if hasattr(dataset, "indices") else sample_index
        source_name = "unknown_dataset"
        if hasattr(dataset, "records"):
            record = dataset.records[record_index]
            source_name = str(getattr(record, "source_dataset", source_name))
        source_labels.append(source_name)

    image_batch = torch.stack(images, dim=0)
    mean = image_batch.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image_batch.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    image_batch = (image_batch * std + mean).clamp(0.0, 1.0)

    annotated_images: list[torch.Tensor] = []
    for image_tensor, source_label in zip(image_batch, source_labels):
        pil_image = tvf.to_pil_image(image_tensor)
        draw = ImageDraw.Draw(pil_image)
        text = source_label
        text_height = 14
        draw.rectangle((0, 0, pil_image.width, text_height), fill=(0, 0, 0))
        draw.text((2, 1), text, fill=(255, 255, 255))
        annotated_images.append(tvf.to_tensor(pil_image))

    image_batch = torch.stack(annotated_images, dim=0)
    grid_tensor = make_grid(image_batch, nrow=grid_size, padding=2)
    grid_path = run_dir / f"train_samples_grid_{grid_size}x{grid_size}.png"
    tvf.to_pil_image(grid_tensor).save(grid_path)

    writer.add_image(f"samples/train_grid_{grid_size}x{grid_size}", grid_tensor, 0)
    writer.add_text(f"samples/train_grid_{grid_size}x{grid_size}_path", str(grid_path.resolve()), 0)
    writer.add_scalar("samples/train_grid_tile_count", float(total_tiles), 0)
    writer.add_scalar("samples/train_grid_image_size", float(image_size), 0)
    print(f"Saved random train sample grid: {grid_path.resolve()}")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    visibility_threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_visibility_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        predictions = _as_prediction_vector(outputs)
        loss = loss_fn(predictions, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        errors = predictions - targets
        abs_error = torch.abs(errors)
        sq_error = errors * errors
        pred_visible = predictions >= visibility_threshold
        target_visible = targets >= visibility_threshold

        total_loss += float(loss.item()) * batch_size
        total_abs_error += float(abs_error.sum().item())
        total_sq_error += float(sq_error.sum().item())
        total_visibility_correct += int((pred_visible == target_visible).sum().item())
        total_samples += batch_size

    denom = max(total_samples, 1)
    return EpochMetrics(
        loss=total_loss / denom,
        mae=total_abs_error / denom,
        rmse=float(np.sqrt(total_sq_error / denom)),
        visibility_acc=total_visibility_correct / denom,
    )


def main() -> None:
    # -----------------------
    # Editable runtime config
    # -----------------------
    DETECTION_DIR = None  # Example: "docs/Cube_Detection_dataset"
    INCLUDE_EXTRA_NEGATIVES = False
    EXTRA_NEGATIVE_DIR = None  # Example: r"C:\\Datasets\\CubeDetection"
    DROP_NEAR_BLANK_EXTRA_NEGATIVES = False

    OUTPUT_DIR = "runs/cube_detection"
    RUN_NAME = None

    TARGET_KEY = "inframe_fraction"  # one of REGRESSION_TARGET_KEYS
    STRATIFY_BINS = 10
    BALANCE_TRAIN_CLASSES = True
    VISIBILITY_THRESHOLD = 0.1

    EPOCHS = 60
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    VAL_RATIO = 0.2
    NUM_WORKERS = 0
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    USE_LR_FINDER = True
    APPLY_SUGGESTED_LR = True
    LR_FINDER_START_LR = 1e-4
    LR_FINDER_END_LR = 1e-1
    LR_FINDER_NUM_ITERS = 120
    AUTO_START_TENSORBOARD = True
    TENSORBOARD_HOST = "localhost"
    TENSORBOARD_PORT = 6006
    SEED = 42
    BOUNDED_OUTPUT = True

    if TARGET_KEY not in REGRESSION_TARGET_KEYS:
        raise ValueError(
            f"TARGET_KEY must be one of {REGRESSION_TARGET_KEYS}, got {TARGET_KEY!r}"
        )
    if TARGET_KEY == "edge_margin":
        raise ValueError(
            "TARGET_KEY='edge_margin' is not supported with final sigmoid output. "
            "Use visible_image_ratio or inframe_fraction."
        )

    set_seed(SEED)

    dataloaders = create_dataloaders(
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        detection_dir=DETECTION_DIR,
        include_extra_negatives=INCLUDE_EXTRA_NEGATIVES,
        extra_negative_dir=EXTRA_NEGATIVE_DIR,
        drop_near_blank_extra_negatives=DROP_NEAR_BLANK_EXTRA_NEGATIVES,
        target_key=TARGET_KEY,
        stratify_bins=STRATIFY_BINS,
        balance_train_classes=BALANCE_TRAIN_CLASSES,
    )

    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(OUTPUT_DIR) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_simple_cnn_regressor(
        bounded_output=BOUNDED_OUTPUT,
        freeze_backbone=False,
    ).to(device)
    loss_fn = nn.MSELoss()

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

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard support requires `tensorboard`. Install with: pip install tensorboard"
        ) from exc

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("data/detection_dir", str(dataloaders.detection_dir.resolve()))
    writer.add_text("data/target_key", TARGET_KEY)
    writer.add_scalar("split/train_size", dataloaders.train_size, 0)
    writer.add_scalar("split/val_size", dataloaders.val_size, 0)
    writer.add_scalar("target/min", dataloaders.target_stats["min"], 0)
    writer.add_scalar("target/max", dataloaders.target_stats["max"], 0)
    writer.add_scalar("target/mean", dataloaders.target_stats["mean"], 0)
    writer.add_scalar("target/std", dataloaders.target_stats["std"], 0)
    _log_random_train_grid(
        loader=dataloaders.train_loader,
        run_dir=run_dir,
        writer=writer,
        image_size=IMAGE_SIZE,
        grid_size=8,
    )
    if USE_LR_FINDER:
        writer.add_scalar("train/lr_finder_suggested_lr", LEARNING_RATE, 0)
        writer.add_text("train/lr_finder_plot_path", str((run_dir / "lr_finder.png").resolve()), 0)
    writer.flush()

    print(f"Using device: {device}")
    print(f"Train samples: {dataloaders.train_size} | Val samples: {dataloaders.val_size}")
    print(f"Dataset class counts (derived from filename label): {dataloaders.class_counts}")
    print(f"Train class counts: {dataloaders.train_class_counts}")
    print(f"Val class counts: {dataloaders.val_class_counts}")
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
        train_metrics = run_epoch(
            model=model,
            loader=dataloaders.train_loader,
            loss_fn=loss_fn,
            device=device,
            visibility_threshold=VISIBILITY_THRESHOLD,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=dataloaders.val_loader,
                loss_fn=loss_fn,
                device=device,
                visibility_threshold=VISIBILITY_THRESHOLD,
                optimizer=None,
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
            "loss_type": "mse",
            "bounded_output": BOUNDED_OUTPUT,
            "visibility_threshold": VISIBILITY_THRESHOLD,
            "detection_dir": str(dataloaders.detection_dir.resolve()),
            "class_counts": dataloaders.class_counts,
            "train_class_counts": dataloaders.train_class_counts,
            "val_class_counts": dataloaders.val_class_counts,
            "target_stats": dataloaders.target_stats,
        }
        torch.save(checkpoint, last_ckpt_path)

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
