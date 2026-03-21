"""Model factory for cube localisation regression."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import torch
from torch import nn

from cube_localisation.forward_kinematics import RobotFKModel


class CubeLocalisationRegressor(nn.Module):
    """Fuses image features and FK-derived end-effector pose for cube localisation regression."""

    def __init__(
        self,
        image_encoder: nn.Module,
        image_feature_dim: int,
        output_dim: int,
        joint_input_dim: int,
        dropout: float = 0.1,
        joint_hidden_dim: int = 64,
        joint_mean: np.ndarray | torch.Tensor | None = None,
        joint_std: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.joint_input_dim = int(joint_input_dim)
        if self.joint_input_dim <= 0:
            raise ValueError("joint_input_dim must be > 0.")

        self._fk_model = RobotFKModel()
        self._fk_joint_count = len(self._fk_model.joints)
        if self.joint_input_dim != self._fk_joint_count:
            raise ValueError(
                f"RobotFKModel expects {self._fk_joint_count} joints, got joint_input_dim={self.joint_input_dim}."
            )
        self._fk_ee_joint_index = self._fk_joint_count - 1

        if (joint_mean is None) != (joint_std is None):
            raise ValueError("joint_mean and joint_std must either both be set or both be None.")

        if joint_mean is not None and joint_std is not None:
            joint_mean_tensor = torch.as_tensor(joint_mean, dtype=torch.float32).view(-1)
            joint_std_tensor = torch.as_tensor(joint_std, dtype=torch.float32).view(-1)
            if joint_mean_tensor.numel() != self.joint_input_dim or joint_std_tensor.numel() != self.joint_input_dim:
                raise ValueError(
                    "joint_mean and joint_std must match joint_input_dim. "
                    f"Expected {self.joint_input_dim}, got {joint_mean_tensor.numel()} and {joint_std_tensor.numel()}."
                )
            self.register_buffer("_joint_mean", joint_mean_tensor, persistent=False)
            self.register_buffer("_joint_std", joint_std_tensor, persistent=False)
        else:
            self._joint_mean = None
            self._joint_std = None

        self._fk_translation_dim = 3
        self._fk_rotation_dim = 3
        self._fk_feature_dim = self._fk_translation_dim + (2 * self._fk_rotation_dim)
        self.joint_encoder = nn.Sequential(
            nn.Linear(self._fk_feature_dim, joint_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        fusion_dim = image_feature_dim + joint_hidden_dim
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, output_dim),
        )

    def _joint_inputs_to_ee_features(self, joint_inputs: torch.Tensor) -> torch.Tensor:
        if self._joint_mean is not None and self._joint_std is not None:
            joint_inputs = joint_inputs * self._joint_std + self._joint_mean

        joint_inputs_np = joint_inputs.detach().to(dtype=torch.float32, device="cpu").numpy()
        ee_features = np.zeros((joint_inputs_np.shape[0], self._fk_feature_dim), dtype=np.float32)

        for sample_index, joint_radians in enumerate(joint_inputs_np):
            joint_degrees = np.rad2deg(joint_radians)
            self._fk_model.set_joint_angles(*joint_degrees.tolist())
            rotation_deg, translation = self._fk_model.get_joint_rot_trans(self._fk_ee_joint_index)
            rotation_rad = np.deg2rad(np.asarray(rotation_deg, dtype=np.float32))

            ee_features[sample_index, 0:3] = np.asarray(translation, dtype=np.float32)
            ee_features[sample_index, 3:6] = np.sin(rotation_rad)
            ee_features[sample_index, 6:9] = np.cos(rotation_rad)

        return torch.from_numpy(ee_features).to(device=joint_inputs.device, dtype=joint_inputs.dtype)

    def forward(self, images: torch.Tensor, joint_inputs: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images)
        if image_features.ndim > 2:
            image_features = torch.flatten(image_features, start_dim=1)

        if joint_inputs.ndim != 2:
            raise ValueError(
                f"Expected joint_inputs shape [batch, joints], got shape {tuple(joint_inputs.shape)}."
            )
        if joint_inputs.shape[1] != self.joint_input_dim:
            raise ValueError(
                f"Expected {self.joint_input_dim} joint inputs, got {joint_inputs.shape[1]}."
            )

        ee_features = self._joint_inputs_to_ee_features(joint_inputs)
        joint_features = self.joint_encoder(ee_features)
        fused_features = torch.cat([image_features, joint_features], dim=1)
        return self.regressor(fused_features)


def _build_resnet_backbone(
    models: object,
    backbone: str,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    resnet_factories: dict[str, tuple[Callable[..., nn.Module], object]] = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    }

    if backbone not in resnet_factories:
        supported = ", ".join(sorted(list(resnet_factories.keys()) + ["efficientnet_b0"]))
        raise ValueError(f"Unsupported backbone: {backbone}. Supported: {supported}")

    factory, default_weights = resnet_factories[backbone]
    weights = default_weights if pretrained else None
    try:
        model = factory(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        warnings.warn(
            f"Could not load pretrained weights for {backbone} ({exc}). Falling back to random initialization.",
            stacklevel=2,
        )
        model = factory(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, in_features


def _build_efficientnet_backbone(
    models: object,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    try:
        model = models.efficientnet_b0(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        warnings.warn(
            f"Could not load pretrained weights for efficientnet_b0 ({exc}). Falling back to random initialization.",
            stacklevel=2,
        )
        model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Identity()
    return model, in_features


def build_localisation_model(
    output_dim: int = 2,
    backbone: str = "resnet34",
    pretrained: bool = True,
    dropout: float = 0.1,
    joint_input_dim: int = 6,
    joint_hidden_dim: int = 64,
    joint_mean: np.ndarray | torch.Tensor | None = None,
    joint_std: np.ndarray | torch.Tensor | None = None,
) -> nn.Module:
    """
    Build an image+joint regression model.

    `resnet34` is the default for more capacity while keeping ResNet-style
    training behavior.
    """
    if output_dim <= 0:
        raise ValueError("output_dim must be > 0.")
    if joint_input_dim <= 0:
        raise ValueError("joint_input_dim must be > 0.")

    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "Model creation requires torchvision pretrained backbones. Install with: pip install torchvision"
        ) from exc

    if backbone == "efficientnet_b0":
        image_encoder, image_feature_dim = _build_efficientnet_backbone(models=models, pretrained=pretrained)
    else:
        image_encoder, image_feature_dim = _build_resnet_backbone(
            models=models,
            backbone=backbone,
            pretrained=pretrained,
        )

    return CubeLocalisationRegressor(
        image_encoder=image_encoder,
        image_feature_dim=image_feature_dim,
        output_dim=output_dim,
        joint_input_dim=joint_input_dim,
        dropout=dropout,
        joint_hidden_dim=joint_hidden_dim,
        joint_mean=joint_mean,
        joint_std=joint_std,
    )
