"""Dataset and dataloader utilities for cube visibility regression."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps, ImageStat
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTION_DIR = PROJECT_ROOT / "docs" / "Cube_Detection_dataset"
DEFAULT_EXTRA_NEGATIVE_DIR = Path(r"C:\Datasets\CubeDetection")
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
    source_dataset: str


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
    train_class_counts: dict[str, int]
    val_class_counts: dict[str, int]
    detection_dir: Path


class CubeDetectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Regression dataset for cube visibility metrics."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        indices: Sequence[int],
        transform: object,
        extra_negative_transform: object,
        target_key: str,
    ) -> None:
        if target_key not in REGRESSION_TARGET_KEYS:
            raise ValueError(
                f"target_key must be one of {REGRESSION_TARGET_KEYS}, got {target_key!r}"
            )
        self.records = records
        self.indices = list(indices)
        self.transform = transform
        self.extra_negative_transform = extra_negative_transform
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[self.indices[index]]
        image_transform = (
            self.extra_negative_transform
            if record.source_dataset == "extra_negative_dataset"
            else self.transform
        )
        with Image.open(record.image_path) as image:
            image_tensor = image_transform(image)
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
        source_dataset="cube_detection_dataset",
    )


def _is_near_blank_image(
    image_path: Path,
    mean_threshold: float = 250.0,
    std_threshold: float = 2.0,
    dark_pixel_threshold: float = 8.0,
    bright_pixel_threshold: float = 247.0,
    extreme_ratio_threshold: float = 0.9,
) -> bool:
    """
    Detect low-information images that add little training signal.

    This catches:
    - near-uniform white images,
    - near-uniform black images,
    - and highly extreme images that are mostly black/white with tiny structure.
    """
    with Image.open(image_path) as image:
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        mean_value = float(stat.mean[0])
        std_value = float(stat.stddev[0])
        gray_np = np.asarray(gray, dtype=np.float32)

    black_mean_threshold = max(0.0, 255.0 - mean_threshold)
    is_near_uniform_extreme = (
        std_value <= std_threshold
        and (mean_value >= mean_threshold or mean_value <= black_mean_threshold)
    )

    dark_ratio = float(np.mean(gray_np <= dark_pixel_threshold))
    bright_ratio = float(np.mean(gray_np >= bright_pixel_threshold))
    is_mostly_extreme = dark_ratio >= extreme_ratio_threshold or bright_ratio >= extreme_ratio_threshold

    return is_near_uniform_extreme or is_mostly_extreme


def load_records(
    detection_dir: str | Path | None = None,
    include_extra_negatives: bool = False,
    extra_negative_dir: str | Path | None = None,
    drop_near_blank_extra_negatives: bool = True,
    near_blank_mean_threshold: float = 250.0,
    near_blank_std_threshold: float = 2.0,
) -> tuple[list[SampleRecord], Path]:
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

    if include_extra_negatives:
        resolved_extra_negative_dir = (
            Path(extra_negative_dir)
            if extra_negative_dir is not None
            else DEFAULT_EXTRA_NEGATIVE_DIR
        )
        counter = 0
        if resolved_extra_negative_dir.exists():
            for image_path in _collect_image_paths(resolved_extra_negative_dir):
                if drop_near_blank_extra_negatives and _is_near_blank_image(
                    image_path=image_path,
                    mean_threshold=near_blank_mean_threshold,
                    std_threshold=near_blank_std_threshold,
                ):
                    continue
                records.append(
                    SampleRecord(
                        image_path=image_path,
                        label=CLASS_NAMES[0],
                        visible_image_ratio=0.0,
                        inframe_fraction=0.0,
                        edge_margin=0.0,
                        source_dataset="extra_negative_dataset",
                    )
                )

                counter += 1
                if counter > 5000:
                    break

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


def _count_labels_for_indices(
    records: Sequence[SampleRecord],
    indices: Sequence[int],
) -> dict[str, int]:
    return {
        CLASS_NAMES[0]: sum(records[index].label == CLASS_NAMES[0] for index in indices),
        CLASS_NAMES[1]: sum(records[index].label == CLASS_NAMES[1] for index in indices),
    }


def _balance_train_indices(
    records: Sequence[SampleRecord],
    train_indices: Sequence[int],
    seed: int,
) -> tuple[int, ...]:
    """
    Downsample majority `no_cube_visible` samples to match `cube_visible` count.

    Validation indices are intentionally untouched.
    """
    no_cube_indices = [idx for idx in train_indices if records[idx].label == CLASS_NAMES[0]]
    cube_visible_indices = [idx for idx in train_indices if records[idx].label == CLASS_NAMES[1]]

    if not no_cube_indices or not cube_visible_indices:
        return tuple(train_indices)

    if len(no_cube_indices) <= len(cube_visible_indices):
        return tuple(train_indices)

    rng = random.Random(seed)
    rng.shuffle(no_cube_indices)
    no_cube_indices = no_cube_indices[: len(cube_visible_indices)]

    balanced_indices = no_cube_indices + cube_visible_indices
    rng.shuffle(balanced_indices)
    return tuple(balanced_indices)


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
                transforms.Lambda(flatten_alpha_to_white),
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomApply(
                    [transforms.Lambda(lambda img: ImageOps.autocontrast(img, cutoff=1))],
                    p=0.15,
                ),
                transforms.RandomApply(
                    [transforms.Lambda(lambda img: jpeg_compress(img, quality=85))],
                    p=0.2,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.35))],
                    p=0.1,
                ),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype=torch.float32),
                transforms.Lambda(lambda x: add_sensor_noise(x, std=0.002, p=0.15)),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose([
        # 1. Resolve alpha correctly first
        transforms.Lambda(flatten_alpha_to_white),

        # 2. Resize to model input
        transforms.Resize((image_size, image_size), antialias=True),

        # 3. Convert to grayscale replicated to 3 channels
        transforms.Grayscale(num_output_channels=3),

        # 4. Keep validation/eval deterministic and structure-preserving.
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float32),

        # 5. ResNet normalization
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_extra_negative_transform(image_size: int) -> object:
    """Simple transform for extra negatives."""
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for image transforms. Install with: pip install torchvision"
        ) from exc

    return transforms.Compose(
        [
            transforms.Lambda(_prepare_extra_negative_image),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_inference_transform(image_size: int, augment: bool = False) -> object:
    """Build deterministic preprocessing used for validation/inference."""
    return _build_image_transform(image_size=image_size, augment=augment)


def add_sensor_noise(tensor: torch.Tensor, std: float = 0.003, p: float = 0.25) -> torch.Tensor:
    """Add low-amplitude Gaussian sensor noise with probability p."""
    if std <= 0.0 or p <= 0.0:
        return tensor
    if torch.rand(1).item() >= p:
        return tensor
    return (tensor + std * torch.randn_like(tensor)).clamp(0.0, 1.0)


def jpeg_compress(img: Image.Image, quality: int = 50) -> Image.Image:
    """Simulate JPEG compression."""
    with io.BytesIO() as buffer:
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()


def _prepare_extra_negative_image(img: Image.Image) -> Image.Image:
    """
    Decode extra-negative images robustly across odd modes/bit-depths.

    Some extra-negative files can be 16-bit or float-like and look nearly black if
    naively converted. This normalizes dynamic range to 8-bit before the regular
    resize/grayscale/tensor/normalize pipeline.
    """
    # Common case: regular 8-bit modes (with optional alpha).
    if img.mode in ("RGB", "RGBA", "L", "LA", "P", "CMYK", "YCbCr"):
        return flatten_alpha_to_white(img)

    # Fallback for high-bit-depth / float-like modes.
    array = np.asarray(img)
    if array.size == 0:
        return flatten_alpha_to_white(img)

    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.percentile(array, 1.0))
    high = float(np.percentile(array, 99.0))
    if high <= low:
        low = float(np.min(array))
        high = float(np.max(array))

    if high <= low:
        scaled = np.zeros_like(array, dtype=np.uint8)
    else:
        scaled = ((np.clip(array, low, high) - low) / (high - low) * 255.0).astype(np.uint8)

    if scaled.ndim == 2:
        prepared = Image.fromarray(scaled, mode="L")
    elif scaled.ndim == 3 and scaled.shape[2] >= 3:
        prepared = Image.fromarray(scaled[:, :, :3], mode="RGB")
    elif scaled.ndim == 3 and scaled.shape[2] == 1:
        prepared = Image.fromarray(scaled[:, :, 0], mode="L")
    else:
        return flatten_alpha_to_white(img)

    return flatten_alpha_to_white(prepared)


def flatten_alpha_to_white(img: Image.Image) -> Image.Image:
    """Composite RGBA / LA images onto a white background."""
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")

def create_dataloaders(
    batch_size: int = 32,
    val_ratio: float = 0.2,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
    detection_dir: str | Path | None = None,
    include_extra_negatives: bool = False,
    extra_negative_dir: str | Path | None = None,
    drop_near_blank_extra_negatives: bool = True,
    near_blank_mean_threshold: float = 250.0,
    near_blank_std_threshold: float = 2.0,
    target_key: str = "visible_image_ratio",
    stratify_bins: int = 10,
    balance_train_classes: bool = True,
) -> CubeDetectionDataLoaders:
    if target_key not in REGRESSION_TARGET_KEYS:
        raise ValueError(
            f"target_key must be one of {REGRESSION_TARGET_KEYS}, got {target_key!r}"
        )

    records, resolved_detection_dir = load_records(
        detection_dir=detection_dir,
        include_extra_negatives=include_extra_negatives,
        extra_negative_dir=extra_negative_dir,
        drop_near_blank_extra_negatives=drop_near_blank_extra_negatives,
        near_blank_mean_threshold=near_blank_mean_threshold,
        near_blank_std_threshold=near_blank_std_threshold,
    )

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
    if balance_train_classes:
        split = SplitIndices(
            train_indices=_balance_train_indices(
                records=records,
                train_indices=split.train_indices,
                seed=seed,
            ),
            val_indices=split.val_indices,
        )

    train_class_counts = _count_labels_for_indices(records=records, indices=split.train_indices)
    val_class_counts = _count_labels_for_indices(records=records, indices=split.val_indices)

    train_dataset = CubeDetectionDataset(
        records=records,
        indices=split.train_indices,
        transform=_build_image_transform(image_size=image_size, augment=True),
        extra_negative_transform=_build_extra_negative_transform(image_size=image_size),
        target_key=target_key,
    )
    val_dataset = CubeDetectionDataset(
        records=records,
        indices=split.val_indices,
        transform=_build_image_transform(image_size=image_size, augment=True),
        extra_negative_transform=_build_extra_negative_transform(image_size=image_size),
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
        train_class_counts=train_class_counts,
        val_class_counts=val_class_counts,
        detection_dir=resolved_detection_dir,
    )
