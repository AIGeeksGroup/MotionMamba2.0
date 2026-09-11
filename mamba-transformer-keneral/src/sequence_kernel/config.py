"""Configuration for the generic minimal/hybrid sequence kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from minimal_attention_kernel import MinimalAttentionKernelConfig

from .token_state import TokenStateVariant


InteractionVariant = TokenStateVariant | Literal["h0_parallel", "h1_innovation", "h2_delta_rule"]


@dataclass(frozen=True)
class SequenceKernelConfig:
    """Select the Stage-1 attention path or its explicit Mamba hybrid."""

    mode: Literal["minimal", "hybrid"]
    attention: MinimalAttentionKernelConfig
    fusion: Literal["gated_add"] = "gated_add"
    fusion_gate_logit_init: float = 0.0
    interaction: InteractionVariant = "naive_gated_add"
    state_summary_dim: int = 128
    use_state_delta: bool = True
    state_conditioning_scale_init: float = 0.0
    state_aware_output_mode: Literal[
        "attention_only", "native_gated_add"
    ] = "attention_only"
    content_memory_enabled: bool = False
    content_memory_slots: int = 1
    content_memory_read_enabled: bool = True
    content_memory_use_state_delta: bool = True
    content_memory_write_gate_context: Literal[
        "previous_memory", "proposal"
    ] = "previous_memory"
    content_memory_decoder_init_scale: float = 0.02
    content_memory_write_gate_init_scale: float = 0.02
    content_memory_write_gate_bias_init: float = 0.0
    content_memory_read_scale_init: float = 0.0
    content_memory_conditioning: Literal["none", "text"] = "none"
    content_memory_normalize_delta: bool = False
    content_memory_normalize_slots: bool = False
    content_memory_max_write_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"minimal", "hybrid"}:
            raise ValueError(f"mode must be 'minimal' or 'hybrid', got {self.mode!r}")
        if self.fusion != "gated_add":
            raise ValueError(
                "the v1 hybrid baseline supports only explicit 'gated_add' fusion"
            )
        if self.interaction not in {
            "naive_gated_add",
            "state_query",
            "state_kv_modulation",
            "h0_parallel",
            "h1_innovation",
            "h2_delta_rule",
        }:
            raise ValueError(f"invalid token/state interaction {self.interaction!r}")
        if self.mode == "minimal" and self.interaction != "naive_gated_add":
            raise ValueError("minimal mode cannot enable token/state interaction")
        if self.state_summary_dim <= 0:
            raise ValueError("state_summary_dim must be positive")
        if self.content_memory_slots <= 0:
            raise ValueError("content_memory_slots must be positive")
        if self.content_memory_enabled and self.mode != "hybrid":
            raise ValueError("content memory requires hybrid mode")
        if self.state_aware_output_mode not in {
            "attention_only", "native_gated_add"
        }:
            raise ValueError("invalid state-aware output mode")
        if (
            self.state_aware_output_mode != "attention_only"
            and self.interaction not in {"state_query", "state_kv_modulation"}
        ):
            raise ValueError(
                "state-aware output mode applies only to state-query/state-KV interactions"
            )
        if self.content_memory_write_gate_context not in {
            "previous_memory", "proposal"
        }:
            raise ValueError("invalid content-memory write-gate context")
        if (
            self.content_memory_decoder_init_scale <= 0
            or self.content_memory_write_gate_init_scale <= 0
        ):
            raise ValueError("content-memory initialization scales must be positive")
        if self.content_memory_conditioning not in {"none", "text"}:
            raise ValueError("invalid content-memory conditioning")
        if not 0.0 < self.content_memory_max_write_rate <= 1.0:
            raise ValueError("content-memory max write rate must be in (0, 1]")

    @property
    def d_model(self) -> int:
        return self.attention.d_model
