"""Dataset and dataloader utilities for cube visibility regression."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTION_DIR = PROJECT_ROOT / "docs" / "Cube_Detection_dataset"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLASS_NAMES = ("no_cube_visible", "cube_visible")
REGRESSION_TARGET_KEYS = ("visible_image_ratio", "inframe_fraction", "edge_margin")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    label: str
    visible_image_ratio: float
    inframe_fraction: float
    edge_margin: float


@dataclass(frozen=True)
class SplitIndices:
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]


@dataclass(frozen=True)
class CubeDetectionDataLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    train_size: int
    val_size: int
    target_key: str
    target_stats: dict[str, float]
    class_counts: dict[str, int]
    detection_dir: Path


class CubeDetectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Regression dataset for cube visibility metrics."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        indices: Sequence[int],
        transform: object,
        target_key: str,
    ) -> None:
        if target_key not in REGRESSION_TARGET_KEYS:
            raise ValueError(
                f"target_key must be one of {REGRESSION_TARGET_KEYS}, got {target_key!r}"
            )
        self.records = records
        self.indices = list(indices)
        self.transform = transform
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[self.indices[index]]
        with Image.open(record.image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        target_value = float(getattr(record, self.target_key))
        target_tensor = torch.tensor(target_value, dtype=torch.float32)
        return image_tensor, target_tensor


def resolve_dataset_dir(detection_dir: str | Path | None = None) -> Path:
    resolved_detection_dir = Path(detection_dir) if detection_dir is not None else DEFAULT_DETECTION_DIR
    if not resolved_detection_dir.exists():
        raise FileNotFoundError(f"Cube_Detection_dataset directory not found: {resolved_detection_dir}")
    return resolved_detection_dir


def _collect_image_paths(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _parse_filename_labels(image_path: Path) -> SampleRecord:
    """
    Parse labels from generator naming convention:
    s_000000__f_0000__label_cube_visible__vis_0.123456__infrm_0.654321__edge_-0.100000.png
    """
    tokens = image_path.stem.split("__")
    values: dict[str, str] = {}
    for token in tokens:
        if "_" not in token:
            continue
        key, raw_value = token.split("_", 1)
        values[key] = raw_value

    missing_keys = [key for key in ("label", "vis", "infrm", "edge") if key not in values]
    if missing_keys:
        raise ValueError(
            f"Filename does not contain required encoded labels {missing_keys}: {image_path.name}"
        )

    label = values["label"]
    if label not in CLASS_NAMES:
        raise ValueError(
            f"Unknown label in filename {image_path.name!r}: {label!r}. "
            f"Expected one of {CLASS_NAMES}."
        )

    return SampleRecord(
        image_path=image_path,
        label=label,
        visible_image_ratio=float(values["vis"]),
        inframe_fraction=float(values["infrm"]),
        edge_margin=float(values["edge"]),
    )


def load_records(detection_dir: str | Path | None = None) -> tuple[list[SampleRecord], Path]:
    resolved_detection_dir = resolve_dataset_dir(detection_dir=detection_dir)
    image_paths = _collect_image_paths(resolved_detection_dir)
    if not image_paths:
        raise RuntimeError(f"Cube_Detection_dataset is empty: {resolved_detection_dir}")

    records: list[SampleRecord] = []
    parse_errors: list[str] = []
    for image_path in image_paths:
        try:
            records.append(_parse_filename_labels(image_path))
        except Exception as exc:
            parse_errors.append(f"{image_path.name}: {exc}")

    if parse_errors:
        preview = "\n".join(parse_errors[:10])
        raise RuntimeError(
            "Failed parsing filename labels from dataset. "
            "Ensure data was generated with the updated cube_detection_generator naming format.\n"
            f"First errors:\n{preview}"
        )

    return records, resolved_detection_dir


def _target_values(records: Sequence[SampleRecord], target_key: str) -> np.ndarray:
    return np.asarray([float(getattr(record, target_key)) for record in records], dtype=np.float64)


def build_stratified_split_indices(
    records: Sequence[SampleRecord],
    val_ratio: float,
    seed: int,
    target_key: str,
    n_bins: int = 10,
) -> SplitIndices:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")
    if not records:
        raise RuntimeError("No samples available for splitting.")
    if n_bins <= 1:
        raise ValueError("n_bins must be > 1")

    rng = random.Random(seed)
    targets = _target_values(records=records, target_key=target_key)

    # If target has no spread, fallback to plain random split.
    if float(targets.max()) == float(targets.min()):
        all_indices = list(range(len(records)))
        rng.shuffle(all_indices)
        val_count = max(1, min(len(all_indices) - 1, int(round(len(all_indices) * val_ratio))))
        val_indices = all_indices[:val_count]
        train_indices = all_indices[val_count:]
        return SplitIndices(train_indices=tuple(train_indices), val_indices=tuple(val_indices))

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(targets, quantiles)
    edges = np.unique(edges)

    if edges.size <= 2:
        all_indices = list(range(len(records)))
        rng.shuffle(all_indices)
        val_count = max(1, min(len(all_indices) - 1, int(round(len(all_indices) * val_ratio))))
        val_indices = all_indices[:val_count]
        train_indices = all_indices[val_count:]
        return SplitIndices(train_indices=tuple(train_indices), val_indices=tuple(val_indices))

    # Assign each sample to a quantile bin.
    bin_ids = np.digitize(targets, edges[1:-1], right=False)
    bin_to_indices: dict[int, list[int]] = {}
    for idx, bin_id in enumerate(bin_ids):
        bin_to_indices.setdefault(int(bin_id), []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for bin_id in sorted(bin_to_indices.keys()):
        indices = bin_to_indices[bin_id]
        rng.shuffle(indices)

        val_count = int(round(len(indices) * val_ratio))
        if len(indices) >= 2:
            val_count = max(1, min(len(indices) - 1, val_count))
        else:
            val_count = 0

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    if not train_indices:
        raise RuntimeError("Train split is empty.")
    if not val_indices:
        raise RuntimeError("Validation split is empty.")

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return SplitIndices(train_indices=tuple(train_indices), val_indices=tuple(val_indices))


def _build_image_transform(image_size: int, augment: bool) -> object:
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for image transforms. Install with: pip install torchvision"
        ) from exc

    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.35),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_dataloaders(
    batch_size: int = 32,
    val_ratio: float = 0.2,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
    detection_dir: str | Path | None = None,
    target_key: str = "visible_image_ratio",
    stratify_bins: int = 10,
) -> CubeDetectionDataLoaders:
    if target_key not in REGRESSION_TARGET_KEYS:
        raise ValueError(
            f"target_key must be one of {REGRESSION_TARGET_KEYS}, got {target_key!r}"
        )

    records, resolved_detection_dir = load_records(detection_dir=detection_dir)

    class_counts = {
        CLASS_NAMES[0]: sum(record.label == CLASS_NAMES[0] for record in records),
        CLASS_NAMES[1]: sum(record.label == CLASS_NAMES[1] for record in records),
    }

    targets = _target_values(records=records, target_key=target_key)
    target_stats = {
        "min": float(np.min(targets)),
        "max": float(np.max(targets)),
        "mean": float(np.mean(targets)),
        "std": float(np.std(targets)),
    }

    split = build_stratified_split_indices(
        records=records,
        val_ratio=val_ratio,
        seed=seed,
        target_key=target_key,
        n_bins=stratify_bins,
    )

    train_dataset = CubeDetectionDataset(
        records=records,
        indices=split.train_indices,
        transform=_build_image_transform(image_size=image_size, augment=True),
        target_key=target_key,
    )
    val_dataset = CubeDetectionDataset(
        records=records,
        indices=split.val_indices,
        transform=_build_image_transform(image_size=image_size, augment=False),
        target_key=target_key,
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return CubeDetectionDataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=len(split.train_indices),
        val_size=len(split.val_indices),
        target_key=target_key,
        target_stats=target_stats,
        class_counts=class_counts,
        detection_dir=resolved_detection_dir,
    )

