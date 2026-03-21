"""Dataset utilities for cube localisation finetuning."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_CANDIDATE = PROJECT_ROOT / "docs" / "Cube_Localisation_dataset"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_joint_angles(
    joint_angles: np.ndarray,
    joint_mean: np.ndarray,
    joint_std: np.ndarray,
) -> np.ndarray:
    """Normalizes joint angles with provided per-dimension statistics."""
    return (joint_angles - joint_mean) / joint_std


def denormalize_joint_angles(
    normalized_joint_angles: np.ndarray,
    joint_mean: np.ndarray,
    joint_std: np.ndarray,
) -> np.ndarray:
    """Restores joint angles from normalized values."""
    return normalized_joint_angles * joint_std + joint_mean


def normalize_cube_position(
    cube_position: np.ndarray,
    cube_pos_mean: np.ndarray,
    cube_pos_std: np.ndarray,
) -> np.ndarray:
    """Normalizes cube xyz position with provided per-dimension statistics."""
    return (cube_position - cube_pos_mean) / cube_pos_std


def denormalize_cube_position(
    normalized_cube_position: np.ndarray,
    cube_pos_mean: np.ndarray,
    cube_pos_std: np.ndarray,
) -> np.ndarray:
    """Restores cube xyz position from normalized values."""
    return normalized_cube_position * cube_pos_std + cube_pos_mean


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    sample_index: int | None
    waypoint_index: int | None
    joint_rotations_rad: tuple[float, ...]
    cube_x_m: float
    cube_y_m: float
    cube_z_m: float
    cube_z_rotation_rad: float
    cube_z_rotation_sin4: float
    cube_z_rotation_cos4: float

    def target_vector(self, target_keys: Sequence[str]) -> np.ndarray:
        return np.asarray([getattr(self, key) for key in target_keys], dtype=np.float32)


def encode_cube_z_rotation_fourfold(cube_z_rotation_rad: float) -> tuple[float, float]:
    """
    Encode cube z-rotation with 90-degree symmetry handling.

    Maps theta -> 4*theta and returns (sin(4*theta), cos(4*theta)).
    Example:
    -45 deg -> -180 deg, +45 deg -> +180 deg, both map to cos=-1 and sin≈0.
    """
    angle_4 = 4.0 * float(cube_z_rotation_rad)
    return float(math.sin(angle_4)), float(math.cos(angle_4))


def decode_cube_z_rotation_fourfold(cube_z_rotation_sin4: float, cube_z_rotation_cos4: float) -> float:
    """
    Decode (sin(4*theta), cos(4*theta)) back to a canonical theta in [-pi/4, pi/4].
    """
    return float(math.atan2(float(cube_z_rotation_sin4), float(cube_z_rotation_cos4)) / 4.0)


@dataclass(frozen=True)
class SpatialRegion:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def overlaps(self, other: "SpatialRegion") -> bool:
        x_overlap = self.x_min < other.x_max and self.x_max > other.x_min
        y_overlap = self.y_min < other.y_max and self.y_max > other.y_min
        return x_overlap and y_overlap

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SpatialRegion":
        return SpatialRegion(
            x_min=float(payload["x_min"]),
            x_max=float(payload["x_max"]),
            y_min=float(payload["y_min"]),
            y_max=float(payload["y_max"]),
        )


@dataclass(frozen=True)
class SpatialSplitConfig:
    seed: int = 42
    val_region_ratio: float = 0.25
    test_region_ratio: float = 0.25
    ensure_non_overlapping_regions: bool = True
    max_sampling_attempts: int = 200

    def validate(self) -> None:
        for field_name in ("val_region_ratio", "test_region_ratio"):
            ratio = getattr(self, field_name)
            if not 0.0 <= ratio < 1.0:
                raise ValueError(f"{field_name} must be in [0, 1). Got {ratio}.")
        if self.max_sampling_attempts <= 0:
            raise ValueError("max_sampling_attempts must be > 0.")


@dataclass(frozen=True)
class SpatialSplit:
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    val_region: SpatialRegion
    test_region: SpatialRegion

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_indices": list(self.train_indices),
            "val_indices": list(self.val_indices),
            "test_indices": list(self.test_indices),
            "val_region": self.val_region.to_dict(),
            "test_region": self.test_region.to_dict(),
        }


def resolve_dataset_dir(dataset_dir: str | Path | None = None) -> Path:
    # Custom dataset directory provided by caller takes precedence.
    if dataset_dir is not None:
        candidate = Path(dataset_dir)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset directory not found: {candidate}")


    if DEFAULT_DATASET_CANDIDATE.exists():
        return DEFAULT_DATASET_CANDIDATE

    raise FileNotFoundError(f"No dataset directory found. Expected: {DEFAULT_DATASET_CANDIDATE}")


def _parse_dataset_filename(file_name: str | Path) -> dict[str, Any]:
    """ 
    Parses a dataset filename to extract metadata values.
    Expected filename format: "wp_{waypoint_index}__s_{sample_index}__j{joint_index}_{joint_rotation_rad}__cx_{cube_x_m}__cy_{cube_y_m}__cz_{cube_z_m}__cyaw_{cube_z_rotation_rad}.png"
    """
    stem = Path(file_name).stem
    values: dict[str, Any] = {}
    joints: dict[int, float] = {}

    for token in stem.split("__"):
        if "_" not in token:
            continue
        key, raw_value = token.split("_", 1)
        if key == "wp":
            values["waypoint_index"] = int(raw_value)
            continue
        if key == "s":
            values["sample_index"] = int(raw_value)
            continue
        if key.startswith("j") and key[1:].isdigit():
            joints[int(key[1:])] = float(raw_value)
            continue
        if key == "cx":
            values["cube_x_m"] = float(raw_value)
            continue
        if key == "cy":
            values["cube_y_m"] = float(raw_value)
            continue
        if key == "cz":
            values["cube_z_m"] = float(raw_value)
            continue
        if key == "cyaw":
            values["cube_z_rotation_rad"] = float(raw_value)

    if joints:
        values["joint_rotations_rad"] = tuple(joints[idx] for idx in sorted(joints.keys()))
    else:
        values["joint_rotations_rad"] = tuple()

    return values


def load_records(dataset_dir: str | Path | None = None) -> list[SampleRecord]:
    """Loads dataset records by parsing PNG filenames in the specified directory."""
    root = resolve_dataset_dir(dataset_dir)
    records: list[SampleRecord] = []

    for image_path in sorted(root.glob("*.png")):
        # parse and validate metadata from filename, skip if required keys are missing
        parsed = _parse_dataset_filename(image_path.name)
        required = ("cube_x_m", "cube_y_m", "cube_z_m", "cube_z_rotation_rad")
        if any(key not in parsed for key in required):
            continue
        
        # create a SampleRecord for this image and add to the list
        cube_z_rotation_rad = float(parsed["cube_z_rotation_rad"])
        cube_z_rotation_sin4, cube_z_rotation_cos4 = encode_cube_z_rotation_fourfold(cube_z_rotation_rad)

        records.append(
            SampleRecord(
                image_path=image_path,
                sample_index=parsed.get("sample_index"),
                waypoint_index=parsed.get("waypoint_index"),
                joint_rotations_rad=tuple(parsed.get("joint_rotations_rad", tuple())),
                cube_x_m=float(parsed["cube_x_m"]),
                cube_y_m=float(parsed["cube_y_m"]),
                cube_z_m=float(parsed["cube_z_m"]),
                cube_z_rotation_rad=cube_z_rotation_rad,
                cube_z_rotation_sin4=cube_z_rotation_sin4,
                cube_z_rotation_cos4=cube_z_rotation_cos4,
            )
        )

    if not records:
        raise RuntimeError(f"No valid PNG records found in {root}")

    return records


def _sample_region(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ratio: float,
    rng: random.Random,
) -> SpatialRegion:
    """Samples a random spatial region within the given bounds with the specified size ratio."""
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)
    width = x_span * ratio
    height = y_span * ratio

    x0_max = max(x_min, x_max - width)
    y0_max = max(y_min, y_max - height)
    x0 = rng.uniform(x_min, x0_max)
    y0 = rng.uniform(y_min, y0_max)

    return SpatialRegion(
        x_min=x0,
        x_max=x0 + width,
        y_min=y0,
        y_max=y0 + height,
    )


def _assign_indices_to_regions(
    records: Sequence[SampleRecord],
    val_region: SpatialRegion,
    test_region: SpatialRegion,
) -> tuple[list[int], list[int], list[int]]:
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for idx, record in enumerate(records):
        x, y = record.cube_x_m, record.cube_y_m
        if test_region.contains(x, y):
            test_indices.append(idx)
        elif val_region.contains(x, y):
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    return train_indices, val_indices, test_indices


def plot_spatial_split(
    records: Sequence[SampleRecord],
    split: SpatialSplit,
    title: str = "Cube Localisation Spatial Split",
    show: bool = True,
    output_path: str | Path | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib") from exc

    def _coords(indices: Sequence[int]) -> tuple[list[float], list[float]]:
        x_vals = [records[index].cube_x_m for index in indices]
        y_vals = [records[index].cube_y_m for index in indices]
        return x_vals, y_vals

    all_indices = set(range(len(records)))
    plotted_indices = set(split.train_indices) | set(split.val_indices) | set(split.test_indices)
    missing_indices = all_indices - plotted_indices
    if missing_indices:
        raise ValueError(
            f"Split is missing {len(missing_indices)} records. "
            "All records must be assigned to train/val/test for full-coverage plotting."
        )

    x_all = [record.cube_x_m for record in records]
    y_all = [record.cube_y_m for record in records]

    train_x, train_y = _coords(split.train_indices)
    val_x, val_y = _coords(split.val_indices)
    test_x, test_y = _coords(split.test_indices)

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(train_x, train_y, s=14, c="#1f77b4", alpha=0.55, label=f"train ({len(train_x)})")
    axis.scatter(val_x, val_y, s=18, c="#ff7f0e", alpha=0.8, label=f"val ({len(val_x)})")
    axis.scatter(test_x, test_y, s=18, c="#2ca02c", alpha=0.8, label=f"test ({len(test_x)})")

    val_rect = Rectangle(
        (split.val_region.x_min, split.val_region.y_min),
        split.val_region.x_max - split.val_region.x_min,
        split.val_region.y_max - split.val_region.y_min,
        linewidth=2,
        edgecolor="#ff7f0e",
        facecolor="none",
        linestyle="--",
        label="val region",
    )
    test_rect = Rectangle(
        (split.test_region.x_min, split.test_region.y_min),
        split.test_region.x_max - split.test_region.x_min,
        split.test_region.y_max - split.test_region.y_min,
        linewidth=2,
        edgecolor="#2ca02c",
        facecolor="none",
        linestyle="--",
        label="test region",
    )
    axis.add_patch(val_rect)
    axis.add_patch(test_rect)

    axis.set_title(title)
    axis.set_xlabel("cube_x_m")
    axis.set_ylabel("cube_y_m")
    axis.set_aspect("equal", adjustable="box")

    x_span = max(x_all) - min(x_all)
    y_span = max(y_all) - min(y_all)
    x_margin = max(0.02 * x_span, 1e-4)
    y_margin = max(0.02 * y_span, 1e-4)
    axis.set_xlim(min(x_all) - x_margin, max(x_all) + x_margin)
    axis.set_ylim(min(y_all) - y_margin, max(y_all) + y_margin)

    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)

    if show:
        plt.show()

    plt.close(fig)


def build_spatial_split(
    records: Sequence[SampleRecord],
    config: SpatialSplitConfig,
    plot: bool = False,
    plot_output_path: str | Path | None = None,
    show_plot: bool = True,
) -> SpatialSplit:
    """Builds a spatial train/val/test split by sampling random regions and assigning records based on cube positions."""
    config.validate()

    x_values = [record.cube_x_m for record in records]
    y_values = [record.cube_y_m for record in records]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    rng = random.Random(config.seed)

    for _ in range(config.max_sampling_attempts):
        val_region = _sample_region(x_min, x_max, y_min, y_max, config.val_region_ratio, rng)
        test_region = _sample_region(x_min, x_max, y_min, y_max, config.test_region_ratio, rng)

        if config.ensure_non_overlapping_regions and val_region.overlaps(test_region):
            continue

        train_indices, val_indices, test_indices = _assign_indices_to_regions(records, val_region, test_region)
        if train_indices and val_indices and test_indices:
            split = SpatialSplit(
                train_indices=tuple(train_indices),
                val_indices=tuple(val_indices),
                test_indices=tuple(test_indices),
                val_region=val_region,
                test_region=test_region,
            )
            if plot:
                plot_spatial_split(
                    records=records,
                    split=split,
                    title="Cube Localisation Spatial Split (Random Regions)",
                    show=show_plot,
                    output_path=plot_output_path,
                )
            return split

    raise RuntimeError(
        "Failed to sample non-empty train/val/test splits from spatial regions. "
        "Try lower region ratios or increase max_sampling_attempts."
    )


def build_split_from_regions(
    records: Sequence[SampleRecord],
    val_region: SpatialRegion,
    test_region: SpatialRegion,
    plot: bool = False,
    plot_output_path: str | Path | None = None,
    show_plot: bool = True,
) -> SpatialSplit:
    train_indices, val_indices, test_indices = _assign_indices_to_regions(records, val_region, test_region)
    if not train_indices:
        raise RuntimeError("Train split is empty for the provided regions.")
    if not val_indices:
        raise RuntimeError("Validation split is empty for the provided regions.")
    if not test_indices:
        raise RuntimeError("Test split is empty for the provided regions.")

    split = SpatialSplit(
        train_indices=tuple(train_indices),
        val_indices=tuple(val_indices),
        test_indices=tuple(test_indices),
        val_region=val_region,
        test_region=test_region,
    )
    if plot:
        plot_spatial_split(
            records=records,
            split=split,
            title="Cube Localisation Spatial Split (Provided Regions)",
            show=show_plot,
            output_path=plot_output_path,
        )
    return split


def compute_target_stats(
    records: Sequence[SampleRecord],
    indices: Iterable[int],
    target_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.stack([records[idx].target_vector(target_keys) for idx in indices], axis=0)
    mean = targets.mean(axis=0).astype(np.float32)
    std = targets.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def compute_joint_stats(
    records: Sequence[SampleRecord],
    indices: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.stack([np.asarray(records[idx].joint_rotations_rad, dtype=np.float32) for idx in indices], axis=0)
    mean = joints.mean(axis=0).astype(np.float32)
    std = joints.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def compute_cube_position_stats(
    records: Sequence[SampleRecord],
    indices: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    cube_positions = np.stack(
        [
            np.asarray([records[idx].cube_x_m, records[idx].cube_y_m, records[idx].cube_z_m], dtype=np.float32)
            for idx in indices
        ],
        axis=0,
    )
    mean = cube_positions.mean(axis=0).astype(np.float32)
    std = cube_positions.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


class CubeLocalisationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        indices: Sequence[int],
        target_keys: Sequence[str],
        image_size: int = 224,
        augment: bool = False,
        joint_input_dim: int | None = None,
        joint_mean: np.ndarray | None = None,
        joint_std: np.ndarray | None = None,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ) -> None:
        self.records = records
        self.indices = list(indices)
        self.target_keys = list(target_keys)
        self.joint_input_dim = int(joint_input_dim) if joint_input_dim is not None else self._infer_joint_input_dim()
        if self.joint_input_dim <= 0:
            raise ValueError("joint_input_dim must be > 0. Joint inputs are required for this model.")
        self.joint_mean = joint_mean.astype(np.float32) if joint_mean is not None else None
        self.joint_std = joint_std.astype(np.float32) if joint_std is not None else None
        self.target_mean = target_mean.astype(np.float32) if target_mean is not None else None
        self.target_std = target_std.astype(np.float32) if target_std is not None else None

        try:
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for image transforms. Install with: pip install torchvision"
            ) from exc

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.12, contrast=0.12)], p=0.35),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _infer_joint_input_dim(self) -> int:
        joint_lengths = {len(self.records[idx].joint_rotations_rad) for idx in self.indices}
        if not joint_lengths:
            raise ValueError("Cannot infer joint_input_dim from an empty index set.")
        if len(joint_lengths) != 1:
            raise ValueError(f"Inconsistent joint vector lengths across samples: {sorted(joint_lengths)}")
        inferred_dim = joint_lengths.pop()
        if inferred_dim <= 0:
            raise ValueError(
                "Dataset records contain empty joint_rotations_rad values, but joint inputs are required."
            )
        return inferred_dim

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[self.indices[item]]
        with Image.open(record.image_path) as img:
            image = self.transform(img.convert("RGB"))

        joint_rotations = np.asarray(record.joint_rotations_rad, dtype=np.float32)
        if joint_rotations.shape[0] != self.joint_input_dim:
            raise ValueError(
                f"Expected {self.joint_input_dim} joint values but got {joint_rotations.shape[0]} "
                f"for sample: {record.image_path.name}"
            )
        if self.joint_mean is not None and self.joint_std is not None:
            joint_rotations = normalize_joint_angles(joint_rotations, self.joint_mean, self.joint_std)

        target = record.target_vector(self.target_keys)
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std

        joint_tensor = torch.from_numpy(joint_rotations)
        target_tensor = torch.from_numpy(target)
        return image, joint_tensor, target_tensor


def save_split_definition(
    output_path: str | Path,
    split: SpatialSplit,
    split_config: SpatialSplitConfig,
) -> Path:
    payload = {
        "split_config": {
            "seed": split_config.seed,
            "val_region_ratio": split_config.val_region_ratio,
            "test_region_ratio": split_config.test_region_ratio,
            "ensure_non_overlapping_regions": split_config.ensure_non_overlapping_regions,
            "max_sampling_attempts": split_config.max_sampling_attempts,
        },
        "split": split.to_dict(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return output_path


def load_split_definition(split_file: str | Path) -> tuple[SpatialRegion, SpatialRegion]:
    with Path(split_file).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    split_payload = payload["split"]
    return (
        SpatialRegion.from_dict(split_payload["val_region"]),
        SpatialRegion.from_dict(split_payload["test_region"]),
    )


# Dataset example usage
if __name__ == "__main__":
    dataset_dir = resolve_dataset_dir()
    records = load_records(dataset_dir)
    split_config = SpatialSplitConfig(
        seed=None,
        val_region_ratio=0.25,
        test_region_ratio=0.25,
        ensure_non_overlapping_regions=True,
        max_sampling_attempts=200,
    )

    split = build_spatial_split(
        records=records,
        config=split_config,
        plot=True,
        show_plot=True,
    )

    print(f"dataset_dir={dataset_dir}")
    print(f"records={len(records)}")
    print(f"train={len(split.train_indices)} val={len(split.val_indices)} test={len(split.test_indices)}")
    print(f"val_region={split.val_region}")
    print(f"test_region={split.test_region}")
