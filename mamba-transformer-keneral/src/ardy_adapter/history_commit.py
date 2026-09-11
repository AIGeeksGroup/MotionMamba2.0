"""Canonical accepted-history materialization shared by training and runtime.

An accepted chunk is always presented to ARDY as history, with a synthetic
next-generation region only to establish the next-window token origin.  The
raw two-stage denoiser remains the sole owner of root/body projections,
position indices, masks, and root-to-local conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any

import torch


@dataclass(frozen=True)
class CanonicalHistoryLayout:
    history_len: torch.Tensor
    generation_len: torch.Tensor
    future_len: torch.Tensor
    history_mask: torch.Tensor
    generation_mask: torch.Tensor
    future_mask: torch.Tensor
    history_token_mask: torch.Tensor
    generation_token_mask: torch.Tensor
    future_token_mask: torch.Tensor
    x: torch.Tensor
    motion_mask: torch.Tensor
    observed_motion: torch.Tensor


def build_canonical_history_layout(
    raw_denoiser: Any,
    history_hybrid: torch.Tensor,
    *,
    history_frames: int,
    next_generation_frames: int,
) -> CanonicalHistoryLayout:
    """Build ``[history | synthetic generation]`` masks for a prime call."""
    if history_hybrid.ndim != 3:
        raise ValueError("history_hybrid must have shape [B,T,D]")
    nfpt = int(raw_denoiser.num_frames_per_token)
    if history_frames <= 0 or next_generation_frames <= 0:
        raise ValueError("history and next generation lengths must be positive")
    if history_frames % nfpt or next_generation_frames % nfpt:
        raise ValueError("history/generation frames must be divisible by token patch size")
    history_tokens = history_frames // nfpt
    generation_tokens = next_generation_frames // nfpt
    if history_hybrid.shape[1] != history_tokens:
        raise ValueError(
            f"history_hybrid has {history_hybrid.shape[1]} tokens; expected {history_tokens}"
        )
    batch, _, token_dim = history_hybrid.shape
    total_frames = history_frames + next_generation_frames
    total_tokens = history_tokens + generation_tokens
    device = history_hybrid.device
    x = torch.zeros(batch, total_tokens, token_dim, device=device, dtype=history_hybrid.dtype)
    x[:, :history_tokens] = history_hybrid
    history_len = torch.full((batch,), history_frames, dtype=torch.long, device=device)
    generation_len = torch.full((batch,), next_generation_frames, dtype=torch.long, device=device)
    future_len = torch.zeros_like(history_len)
    from ardy.model.ardy_model import get_three_mask_from_len

    history_mask, generation_mask, future_mask = get_three_mask_from_len(
        history_len, generation_len, future_len, total_frames, device
    )
    history_token_mask, generation_token_mask, future_token_mask = (
        raw_denoiser.hybrid.convert_frame_mask_to_token_mask(
            history_mask,
            generation_mask,
            future_mask,
            torch.zeros(batch, total_frames, raw_denoiser.motion_rep.motion_rep_dim,
                        dtype=torch.float32, device=device),
        )
    )
    motion_mask = torch.zeros(
        batch, total_frames, raw_denoiser.motion_rep.motion_rep_dim,
        dtype=torch.float32, device=device,
    )
    observed_motion = torch.zeros_like(motion_mask)
    return CanonicalHistoryLayout(
        history_len=history_len,
        generation_len=generation_len,
        future_len=future_len,
        history_mask=history_mask,
        generation_mask=generation_mask,
        future_mask=future_mask,
        history_token_mask=history_token_mask,
        generation_token_mask=generation_token_mask,
        future_token_mask=future_token_mask,
        x=x,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )


def canonical_history_prime(
    raw_denoiser: Any,
    history_hybrid: torch.Tensor,
    *,
    text_feat: torch.Tensor,
    text_pad_mask: torch.Tensor,
    first_heading_angle: torch.Tensor,
    next_generation_frames: int,
    chunk_id: int,
    detach: bool = True,
) -> CanonicalHistoryLayout:
    """Materialize one canonical history chunk and commit it atomically."""
    history_frames = int(history_hybrid.shape[1] * raw_denoiser.num_frames_per_token)
    layout = build_canonical_history_layout(
        raw_denoiser,
        history_hybrid,
        history_frames=history_frames,
        next_generation_frames=next_generation_frames,
    )
    root = raw_denoiser.root_model
    body = raw_denoiser.body_model
    root_before = None if root._sequence_state is None else root._sequence_state.clone()
    body_before = None if body._sequence_state is None else body._sequence_state.clone()
    root_commits = root.accepted_commits
    body_commits = body.accepted_commits
    root.prepare_accepted_commit(layout.history_token_mask, chunk_id=chunk_id, detach=detach)
    body.prepare_accepted_commit(layout.history_token_mask, chunk_id=chunk_id, detach=detach)
    try:
        grad_context = torch.no_grad() if detach else nullcontext()
        with grad_context, torch.autocast(
            device_type=layout.x.device.type, enabled=False
        ):
            raw_denoiser(
                layout.x,
                layout.history_len,
                layout.generation_len,
                layout.future_len,
                layout.history_mask,
                layout.generation_mask,
                layout.future_mask,
                layout.history_token_mask,
                layout.generation_token_mask,
                layout.future_token_mask,
                text_feat,
                text_pad_mask,
                torch.zeros_like(layout.history_len),
                first_heading_angle,
                layout.motion_mask,
                layout.observed_motion,
            )
        if root.accepted_commits != root_commits + 1:
            raise RuntimeError("canonical root history prime must commit exactly once")
        if body.accepted_commits != body_commits + 1:
            raise RuntimeError("canonical body history prime must commit exactly once")
    except Exception:
        root._sequence_state = root_before
        body._sequence_state = body_before
        root.accepted_commits = root_commits
        body.accepted_commits = body_commits
        raise
    finally:
        root.cancel_accepted_commit()
        body.cancel_accepted_commit()
    return layout
