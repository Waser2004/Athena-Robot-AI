"""Inference script for cube localisation model."""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Add parent directory to path for imports
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cube_localisation.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    JOINT_SIGN_CORRECTIONS_ON_LOAD,
    denormalize_cube_position,
    decode_cube_z_rotation_fourfold,
)
from cube_localisation.model import build_localisation_model


class CubeLocalisationInference:
    """Inference wrapper for cube localisation model."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the inference model.

        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract model configuration from checkpoint
        self.backbone = checkpoint.get("backbone", "resnet34")
        self.use_pretrained = checkpoint.get("use_pretrained", True)
        self.image_size = checkpoint.get("image_size", 224)
        self.target_keys = checkpoint.get("target_keys", ["cube_x_m", "cube_y_m", "cube_z_rotation_sin4", "cube_z_rotation_cos4"])
        self.joint_input_dim = checkpoint.get("joint_input_dim", 6)
        self.joint_hidden_dim = checkpoint.get("joint_hidden_dim", 64)

        # Load normalization statistics
        self.joint_mean = np.array(checkpoint["joint_mean"], dtype=np.float32)
        self.joint_std = np.array(checkpoint["joint_std"], dtype=np.float32)
        self.target_mean = np.array(checkpoint["target_mean"], dtype=np.float32)
        self.target_std = np.array(checkpoint["target_std"], dtype=np.float32)

        # Build model
        self.model = build_localisation_model(
            output_dim=len(self.target_keys),
            backbone=self.backbone,
            pretrained=False,  # We'll load the state dict
            joint_input_dim=self.joint_input_dim,
            joint_hidden_dim=self.joint_hidden_dim,
            joint_mean=self.joint_mean,
            joint_std=self.joint_std,
        )

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        checkpoint_sha = _file_sha256_hex(self.checkpoint_path)
        print(f"[OrpheusDebug] checkpoint={self.checkpoint_path} checkpoint_sha256={checkpoint_sha}")
        print(f"✓ Model loaded from: {self.checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Backbone: {self.backbone}")
        print(f"  Image size: {self.image_size}")
        print(f"[OrpheusDebug] checkpoint_image_size={self.image_size} source=checkpoint_memory")
        print(f"  Target keys: {self.target_keys}")

    def _load_and_preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        """
        Load and preprocess an image.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image tensor of shape (3, image_size, image_size)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to tensor and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Apply ImageNet normalization (ensure float32 to avoid type promotion to float64)
        imagenet_mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        imagenet_std = np.array(IMAGENET_STD, dtype=np.float32)
        image_array = (image_array - imagenet_mean) / imagenet_std

        # Convert to tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()

        return image_tensor

    def _normalize_joint_angles(self, joint_angles: np.ndarray) -> torch.Tensor:
        """
        Normalize joint angles using stored statistics.

        Args:
            joint_angles: Joint angles in radians, shape (joint_input_dim,)

        Returns:
            Normalized joint angles as tensor
        """
        if len(joint_angles) != self.joint_input_dim:
            raise ValueError(
                f"Expected {self.joint_input_dim} joint angles but got {len(joint_angles)}"
            )

        # Normalize
        normalized = (joint_angles - self.joint_mean) / self.joint_std
        return torch.from_numpy(normalized.astype(np.float32))

    @staticmethod
    def _parse_joint_angles_from_filename(image_path: str | Path) -> np.ndarray:
        """
        Extract joint angles encoded in the image filename.

        Expected tokens like ``j0_...__j1_...`` in the filename stem. The parsing
        matches the dataset convention, including the sign correction applied to joint 3.
        """
        stem = Path(image_path).stem
        joints: dict[int, float] = {}

        for token in stem.split("__"):
            if "_" not in token:
                continue
            key, raw_value = token.split("_", 1)
            if key.startswith("j") and key[1:].isdigit():
                joints[int(key[1:])] = float(raw_value)

        if not joints:
            raise ValueError(
                f"Could not extract joint angles from filename: {Path(image_path).name}"
            )

        for joint_index, sign in JOINT_SIGN_CORRECTIONS_ON_LOAD.items():
            if joint_index in joints:
                joints[joint_index] = float(joints[joint_index]) * float(sign)

        ordered_joint_indices = sorted(joints.keys())
        joint_angles = np.asarray([joints[index] for index in ordered_joint_indices], dtype=np.float32)
        return joint_angles

    @torch.no_grad()
    def infer(
        self,
        image_path: str | Path,
        joint_angles: np.ndarray | list | None = None,
    ) -> dict:
        """
        Run inference on a single sample.

        Args:
            image_path: Path to the RGB image
            joint_angles: Robot joint angles in radians (array of length joint_input_dim).
                If omitted, the angles are parsed from the image filename.

        Returns:
            Dictionary containing:
                - "raw_prediction": Raw model output (normalized values)
                - "cube_x_m": Predicted cube X coordinate in meters
                - "cube_y_m": Predicted cube Y coordinate in meters
                - "cube_z_rotation_rad": Predicted cube Z rotation in radians
                - "cube_z_rotation_sin4": Predicted sin(4*rotation)
                - "cube_z_rotation_cos4": Predicted cos(4*rotation)
        """
        # Load and preprocess image
        image_tensor = self._load_and_preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        image_sha = _file_sha256_hex(Path(image_path))
        print(f"[OrpheusDebug] image_path={Path(image_path)} image_sha256={image_sha}")
        print(
            "[OrpheusDebug] image_tensor "
            f"shape={tuple(image_tensor.shape)} "
            f"min={float(image_tensor.min().item()):.6f} "
            f"max={float(image_tensor.max().item()):.6f} "
            f"mean={float(image_tensor.mean().item()):.6f}"
        )

        # Normalize joint angles
        if joint_angles is None:
            joint_angles = self._parse_joint_angles_from_filename(image_path)
            print(f"[OrpheusDebug] joints_source=filename")
        else:
            joint_angles = np.array(joint_angles, dtype=np.float32)
            print(f"[OrpheusDebug] joints_source=manual")
        joint_tensor = self._normalize_joint_angles(joint_angles)
        joint_tensor = joint_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        print(f"[OrpheusDebug] joints_rad={joint_angles.tolist()}")
        print(f"[OrpheusDebug] joints_input={joint_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()}")

        # Run inference
        predictions = self.model(image_tensor, joint_tensor)
        predictions = predictions.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
        print(f"[OrpheusDebug] raw_prediction={predictions.astype(np.float32).tolist()}")

        # Denormalize predictions
        denormalized = predictions * self.target_std + self.target_mean
        print(f"[OrpheusDebug] denormalized_prediction={denormalized.astype(np.float32).tolist()}")

        # Parse results
        results = {
            "raw_prediction": predictions,
            "denormalized_prediction": denormalized,
        }

        # Map to target keys
        for i, key in enumerate(self.target_keys):
            results[key] = float(denormalized[i])

        # Decode rotation if present
        if "cube_z_rotation_sin4" in self.target_keys and "cube_z_rotation_cos4" in self.target_keys:
            idx_sin = self.target_keys.index("cube_z_rotation_sin4")
            idx_cos = self.target_keys.index("cube_z_rotation_cos4")
            results["cube_z_rotation_rad"] = decode_cube_z_rotation_fourfold(
                denormalized[idx_sin],
                denormalized[idx_cos],
            )

        return results

    def infer_batch(
        self,
        image_paths: list[str | Path],
        joint_angles_batch: list[np.ndarray] | np.ndarray | None = None,
    ) -> list[dict]:
        """
        Run inference on multiple samples.

        Args:
            image_paths: List of paths to RGB images
            joint_angles_batch: List of joint angle arrays, numpy array of shape
                (batch_size, joint_input_dim), or None to parse angles from filenames

        Returns:
            List of result dictionaries (one per sample)
        """
        results = []
        if joint_angles_batch is None:
            for image_path in image_paths:
                result = self.infer(image_path, None)
                results.append(result)
            return results

        for image_path, joint_angles in zip(image_paths, joint_angles_batch):
            result = self.infer(image_path, joint_angles)
            results.append(result)
        return results


# ============================================================================
# INFERENCE CONFIGURATION - MODIFY THESE VARIABLES
# ============================================================================

# Path to the checkpoint (.pt file)
CHECKPOINT_PATH = r"D:\OneDrive - Venusnet\Dokumente\4. Robot V2\Orpheus (RobotAI)\runs\cube_localisation\20260331-071029\checkpoints\best.pt"

# Path to the image for inference
IMAGE_PATH = r"C:\Users\nicow\Downloads\s_000000__wp_0000__j0_0.350314__j1_0.000000__j2_0.820305__j3_0.911869__j4_-2.383215__j5_-1.412130__cx_0.000000__cy_0.000000__cz_0.000000__cyaw_0.000000.png"

# Device: "cuda" or "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================


def _file_sha256_hex(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return "unavailable"


def main():
    """Main inference function."""
    # Initialize inference model
    inference = CubeLocalisationInference(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)

    # Run inference
    print(f"\n--- Running Inference ---")
    print(f"Image: {IMAGE_PATH}")
    result = inference.infer(image_path=IMAGE_PATH)

    # Print results
    print(f"\n--- Prediction Results ---")
    print(f"Raw prediction (normalized): {result['raw_prediction']}")
    print(f"Denormalized prediction: {result['denormalized_prediction']}")

    for key, value in result.items():
        if key not in ["raw_prediction", "denormalized_prediction"]:
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
