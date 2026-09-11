"""Differentiable long-window objective primitives.

The rollout driver supplies a tensor of per-sample, per-window errors. These
functions deliberately operate only on tensors, so gradients can flow from
post-window-5 objectives into accepted endpoints and recurrent writes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def deterministic_recurrent_carry(
    inputs: torch.Tensor,
    write_weight: torch.Tensor,
    read_weight: torch.Tensor,
    *,
    detach_after_window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small functional recurrent rollout used by the mandatory CUDA preflight."""
    if inputs.ndim != 3:
        raise ValueError("inputs must be [batch, windows, features]")
    state = torch.zeros_like(inputs[:, 0])
    endpoints = []
    for window in range(inputs.shape[1]):
        state = torch.tanh(inputs[:, window] @ write_weight + state)
        endpoint = state @ read_weight
        endpoints.append(endpoint)
        if detach_after_window is not None and window + 1 == detach_after_window:
            state = state.detach()
    return torch.stack(endpoints, dim=1), state


def validate_window_errors(errors: torch.Tensor, *, min_windows: int = 6) -> None:
    if errors.ndim != 2:
        raise ValueError(f"errors must be [batch, windows], got {tuple(errors.shape)}")
    if errors.shape[1] < min_windows:
        raise ValueError(f"need at least {min_windows} windows, got {errors.shape[1]}")
    if not torch.is_floating_point(errors):
        raise TypeError("errors must be floating point")


def least_squares_slope(values: torch.Tensor, *, start_window: int = 5) -> torch.Tensor:
    """Return one slope per batch row, using zero-based window indices."""
    validate_window_errors(values, min_windows=start_window + 1)
    tail = values[:, start_window:]
    x = torch.arange(tail.shape[1], device=values.device, dtype=values.dtype)
    x = x - x.mean()
    return (tail * x).sum(dim=1) / x.square().sum().clamp_min(torch.finfo(values.dtype).eps)


def post_window_auc(values: torch.Tensor, *, start_window: int = 5) -> torch.Tensor:
    validate_window_errors(values, min_windows=start_window + 1)
    return torch.trapezoid(values[:, start_window:], dim=1)


def differentiable_rollout_loss(
    errors: torch.Tensor,
    *,
    slope_weight: float = 0.0,
    auc_weight: float = 0.0,
    final_weight: float = 0.0,
    margin_cm_per_window: float = 0.0,
    start_window: int = 5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Smooth one-sided slope plus AUC/final penalties with explicit units."""
    validate_window_errors(errors, min_windows=start_window + 1)
    slopes = least_squares_slope(errors, start_window=start_window)
    auc = post_window_auc(errors, start_window=start_window)
    final = errors[:, -1]
    slope_loss = F.softplus(slopes - float(margin_cm_per_window)).mean()
    auc_loss = auc.mean()
    final_loss = final.mean()
    total = slope_weight * slope_loss + auc_weight * auc_loss + final_weight * final_loss
    return total, {
        "slope_cm_per_window": slopes,
        "auc_cm_windows": auc,
        "final_error_cm": final,
        "slope_loss": slope_loss,
        "auc_loss": auc_loss,
        "final_loss": final_loss,
    }
