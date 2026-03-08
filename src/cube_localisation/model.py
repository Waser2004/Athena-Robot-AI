"""Model factory for cube localisation regression."""

from __future__ import annotations

import warnings
from typing import Callable

import torch
from torch import nn


class CubeLocalisationRegressor(nn.Module):
    """Fuses image features and joint rotations for cube localisation regression."""

    def __init__(
        self,
        image_encoder: nn.Module,
        image_feature_dim: int,
        output_dim: int,
        joint_input_dim: int,
        dropout: float = 0.1,
        joint_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.joint_input_dim = int(joint_input_dim)
        if self.joint_input_dim <= 0:
            raise ValueError("joint_input_dim must be > 0.")

        self.joint_encoder = nn.Sequential(
            nn.Linear(self.joint_input_dim, joint_hidden_dim),
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

        joint_features = self.joint_encoder(joint_inputs)
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
    backbone: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.1,
    joint_input_dim: int = 6,
    joint_hidden_dim: int = 64,
) -> nn.Module:
    """
    Build an image+joint regression model.

    `resnet18` is the default because it fine-tunes quickly and works well on
    small to medium image datasets.
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
    )
