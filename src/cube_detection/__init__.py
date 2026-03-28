"""Cube detection training package."""

from .dataset import CLASS_NAMES, REGRESSION_TARGET_KEYS, create_dataloaders
from .model import build_resnet18_classifier, build_resnet18_regressor
from .utils import LRSuggestion, suggest_lr

__all__ = [
    "CLASS_NAMES",
    "REGRESSION_TARGET_KEYS",
    "create_dataloaders",
    "build_resnet18_classifier",
    "build_resnet18_regressor",
    "LRSuggestion",
    "suggest_lr",
]
