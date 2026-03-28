"""Utility helpers for cube detection training workflows."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


@dataclass
class LRSuggestion:
    """Container for LR finder outputs."""

    suggested_lr: float
    learning_rates: list[float]
    losses: list[float]
    plot_path: Path | None = None


def suggest_lr(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    *,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iters: int = 100,
    beta: float = 0.98,
    stop_divergence_factor: float = 4.0,
    weight_decay: float = 1e-4,
    output_path: str | Path | None = None,
) -> LRSuggestion:
    """
    Sweep learning rates from low to high and suggest a stable training LR.

    The model weights are restored after the sweep, so calling this function
    does not alter training state.
    """
    if start_lr <= 0 or end_lr <= 0:
        raise ValueError("start_lr and end_lr must be positive.")
    if end_lr <= start_lr:
        raise ValueError("end_lr must be greater than start_lr.")
    if num_iters < 10:
        raise ValueError("num_iters must be >= 10 for a meaningful LR sweep.")

    was_training = model.training
    model.train()

    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=weight_decay)
    lr_mult = math.exp(math.log(end_lr / start_lr) / max(num_iters - 1, 1))

    avg_loss = 0.0
    best_loss = float("inf")
    learning_rates: list[float] = []
    losses: list[float] = []

    iter_loader = itertools.cycle(loader)

    try:
        for iteration in range(1, num_iters + 1):
            images, labels = next(iter_loader)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            raw_loss = float(loss.item())
            avg_loss = beta * avg_loss + (1.0 - beta) * raw_loss
            smooth_loss = avg_loss / (1.0 - beta**iteration)

            current_lr = float(optimizer.param_groups[0]["lr"])
            learning_rates.append(current_lr)
            losses.append(smooth_loss)

            if smooth_loss < best_loss:
                best_loss = smooth_loss

            if iteration > 10 and smooth_loss > stop_divergence_factor * best_loss:
                break

            for param_group in optimizer.param_groups:
                param_group["lr"] = float(param_group["lr"]) * lr_mult
    finally:
        model.load_state_dict(initial_state)
        if not was_training:
            model.eval()

    if len(learning_rates) < 3:
        raise RuntimeError("LR sweep finished too early. Increase num_iters or check data/loss stability.")

    log_lrs = np.log10(np.asarray(learning_rates, dtype=np.float64))
    loss_values = np.asarray(losses, dtype=np.float64)
    gradients = np.gradient(loss_values, log_lrs)

    trim = max(1, len(learning_rates) // 20)
    candidate_start = trim
    candidate_end = len(learning_rates) - trim
    if candidate_end <= candidate_start:
        candidate_start = 0
        candidate_end = len(learning_rates)

    best_idx = int(np.argmin(gradients[candidate_start:candidate_end])) + candidate_start
    suggested_lr = float(learning_rates[best_idx])

    plot_file: Path | None = None
    if output_path is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Plotting LR suggestions requires matplotlib. Install with: pip install matplotlib"
            ) from exc

        plot_file = Path(output_path)
        plot_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(learning_rates, losses, color="#1f77b4", linewidth=2)
        axis.scatter([suggested_lr], [losses[best_idx]], color="#d62728", s=60, label="Suggested LR")
        axis.set_xscale("log")
        axis.set_xlabel("Learning rate")
        axis.set_ylabel("Smoothed loss")
        axis.set_title("Learning rate range test")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_file, dpi=180)
        plt.close(fig)

    return LRSuggestion(
        suggested_lr=suggested_lr,
        learning_rates=learning_rates,
        losses=losses,
        plot_path=plot_file,
    )

