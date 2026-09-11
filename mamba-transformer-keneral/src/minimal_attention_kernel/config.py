"""Configuration for the standalone minimal attention kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MinimalAttentionKernelConfig:
    """Validated construction options for :class:`MinimalAttentionKernel`.

    ``head_dim`` is optional because it is fully determined by
    ``d_model // num_heads``. Supplying it is useful at config boundaries and
    is checked rather than silently ignored.
    """

    d_model: int
    num_heads: int
    window_size_chunks: int
    head_dim: Optional[int] = None
    use_attention_sinks: bool = True
    num_sink_tokens: int = 4
    attention_backend: str = "flash2"
    allow_backend_fallback: bool = True
    causal_within_chunk: bool = False
    attention_dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        derived_head_dim = self.d_model // self.num_heads
        if self.head_dim is not None and self.head_dim != derived_head_dim:
            raise ValueError(
                f"head_dim must equal d_model // num_heads ({derived_head_dim}), "
                f"got {self.head_dim}"
            )
        if self.window_size_chunks < 0:
            raise ValueError(
                f"window_size_chunks must be non-negative, got {self.window_size_chunks}"
            )
        if self.num_sink_tokens < 0:
            raise ValueError(f"num_sink_tokens must be non-negative, got {self.num_sink_tokens}")
        if self.use_attention_sinks and self.num_sink_tokens == 0:
            raise ValueError("use_attention_sinks=True requires num_sink_tokens > 0")
        if not self.use_attention_sinks and self.num_sink_tokens != 0:
            raise ValueError(
                "use_attention_sinks=False requires num_sink_tokens=0 so sink semantics are unambiguous"
            )
        if self.attention_backend not in {"sdpa", "flash2", "flash4"}:
            raise ValueError(
                "attention_backend must be 'sdpa', 'flash2', or the reserved "
                f"'flash4', got {self.attention_backend!r}"
            )
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(
                f"attention_dropout must be in [0, 1), got {self.attention_dropout}"
            )

    @property
    def resolved_head_dim(self) -> int:
        return self.d_model // self.num_heads
