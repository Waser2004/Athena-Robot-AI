"""ResNet18 model factory for cube visibility regression."""

from __future__ import annotations

import warnings

from torch import nn


def build_resnet18_regressor(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    bounded_output: bool = True,
) -> nn.Module:
    """
    Build a ResNet18 regressor for cube-visibility prediction.

    If `bounded_output=True`, model output is constrained to [0, 1] via sigmoid.
    This matches visibility-ratio targets.
    """
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "Model creation requires torchvision. Install with: pip install torchvision"
        ) from exc

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        warnings.warn(
            f"Could not load pretrained ResNet18 weights ({exc}). Falling back to random initialization.",
            stacklevel=2,
        )
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    if bounded_output:
        model.fc = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
    else:
        model.fc = nn.Linear(in_features, 1)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return model


def build_resnet18_classifier(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Backward-compatible helper. Prefer `build_resnet18_regressor`.
    """
    if num_classes <= 1:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}.")
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "Model creation requires torchvision. Install with: pip install torchvision"
        ) from exc

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return model

