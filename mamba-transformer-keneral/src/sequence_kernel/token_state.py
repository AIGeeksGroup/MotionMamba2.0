"""Small token/state interactions for MotionMamba Contribution-1 pilots."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from minimal_attention_kernel import ChunkSlidingWindowKVCache


TokenStateVariant = Literal[
    "naive_gated_add",
    "state_query",
    "state_kv_modulation",
]


class TokenStateInteraction(nn.Module):
    """Use committed prior Mamba summaries only as token-memory addresses.

    Values are never produced from recurrent state. ``scale=0`` makes both
    proposed variants exactly neutral with respect to the Minimal attention
    path, which is required for Minimal-checkpoint initialization.
    """

    def __init__(
        self,
        *,
        variant: TokenStateVariant,
        d_model: int,
        num_heads: int,
        d_state_summary: int,
        use_state_delta: bool,
        scale_init: float,
    ) -> None:
        super().__init__()
        if variant not in {"state_query", "state_kv_modulation"}:
            raise ValueError(f"unsupported token/state interaction: {variant!r}")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if d_state_summary <= 0:
            raise ValueError("d_state_summary must be positive")
        self.variant = variant
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_state_summary = d_state_summary
        self.use_state_delta = use_state_delta
        state_dim = d_state_summary * (2 if use_state_delta else 1)
        self.state_to_address = nn.Linear(state_dim, d_model, bias=False)
        nn.init.xavier_uniform_(self.state_to_address.weight)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def _context(
        self, state_summary: torch.Tensor, state_delta: torch.Tensor
    ) -> torch.Tensor:
        if state_summary.shape != state_delta.shape:
            raise ValueError("state summary and delta must have identical shapes")
        if state_summary.ndim != 2 or state_summary.shape[1] != self.d_state_summary:
            raise ValueError(
                "state summary must be [B,d_state_summary], got "
                f"{tuple(state_summary.shape)}"
            )
        pieces = [state_summary]
        if self.use_state_delta:
            pieces.append(state_delta)
        context = torch.cat(pieces, dim=-1).to(self.state_to_address.weight.dtype)
        return self.state_to_address(context)

    @property
    def effective_scale(self) -> torch.Tensor:
        return torch.tanh(self.scale)

    def query_bias(
        self, state_summary: torch.Tensor, state_delta: torch.Tensor
    ) -> torch.Tensor:
        if self.variant != "state_query":
            raise RuntimeError("query_bias is only defined for state_query")
        return self.effective_scale * self._context(state_summary, state_delta)

    def chunk_key_scales(
        self,
        state_summary: torch.Tensor,
        state_delta: torch.Tensor,
        cache: ChunkSlidingWindowKVCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return differentiable ``[B,num_chunks]`` key scales and raw scores."""
        if self.variant != "state_kv_modulation":
            raise RuntimeError(
                "chunk_key_scales is only defined for state_kv_modulation"
            )
        address = self._context(state_summary, state_delta).view(
            state_summary.shape[0], self.num_heads, self.head_dim
        )
        address = F.normalize(address.float(), dim=-1)
        scores = []
        for chunk in cache.chunks:
            valid = chunk.token_valid[:, None, :, None].to(chunk.key.dtype)
            denominator = valid.sum(dim=2).clamp_min(1.0)
            key_summary = (chunk.key * valid).sum(dim=2) / denominator
            key_summary = F.normalize(key_summary.float(), dim=-1)
            scores.append((address * key_summary).sum(dim=-1).mean(dim=1))
        if scores:
            raw_scores = torch.stack(scores, dim=1)
        else:
            raw_scores = torch.empty(
                state_summary.shape[0],
                0,
                device=state_summary.device,
                dtype=torch.float32,
            )
        scales = 1.0 + self.effective_scale.float() * torch.tanh(raw_scores)
        return scales, raw_scores


__all__ = ["TokenStateInteraction", "TokenStateVariant"]
