"""Install original/DiT/minimal/hybrid sequence configurations into ARDY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from .ardy_dit_block import ArdyDiTSequenceBlock
from .diffusion_memory import ArdyDiffusionMemoryController, ArdySequenceKernelClient


@dataclass(frozen=True)
class ArdySequenceKernelConfig:
    mode: Literal["original", "dit", "minimal", "hybrid"] = "minimal"
    attention_backend: str = "flash2"
    window_size_chunks: int = 4
    num_sink_tokens: int = 4
    allow_backend_fallback: bool = True
    mamba_headdim: int = 64
    mamba_d_state: int = 64
    mamba_expand: int = 2
    mamba_chunk_size: int = 16
    mamba_dtype: torch.dtype = torch.bfloat16
    token_state_interaction: Literal[
        "naive_gated_add", "state_query", "state_kv_modulation",
        "h0_parallel", "h1_innovation", "h2_delta_rule"
    ] = "naive_gated_add"
    state_summary_dim: int = 128
    use_state_delta: bool = True
    state_conditioning_scale_init: float = 0.0
    state_aware_output_mode: Literal[
        "attention_only", "native_gated_add"
    ] = "attention_only"
    fusion_gate_logit_init: float = 0.0
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
    ff_size: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"original", "dit", "minimal", "hybrid"}:
            raise ValueError("invalid ARDY sequence mode")
        if self.window_size_chunks < 0:
            raise ValueError("window_size_chunks must be non-negative")
        if self.num_sink_tokens <= 0 and self.mode in {"minimal", "hybrid"}:
            raise ValueError("minimal/hybrid modes require attention sink tokens")
        if self.ff_size is not None and self.ff_size <= 0:
            raise ValueError("ff_size must be positive")
        if self.mode != "hybrid" and self.token_state_interaction != "naive_gated_add":
            raise ValueError("token/state interaction is available only in hybrid mode")
        if self.content_memory_enabled and self.mode != "hybrid":
            raise ValueError("content memory requires hybrid mode")
        if self.content_memory_slots <= 0:
            raise ValueError("content_memory_slots must be positive")
        if self.state_aware_output_mode not in {
            "attention_only", "native_gated_add"
        }:
            raise ValueError("invalid state-aware output mode")
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


def _copy_compatible_frontend(source: nn.Module, target: nn.Module) -> None:
    """Reuse shape-compatible ARDY embeddings/projections, never mixer weights."""
    names = (
        "embed_text",
        "sequence_pos_encoder",
        "embed_timestep",
        "input_linear",
        "output_linear",
        "linear_first_heading_angle",
        "motion_token_embedding",
    )
    for name in names:
        old = getattr(source, name, None)
        new = getattr(target, name, None)
        if old is None or new is None or not isinstance(old, nn.Module):
            continue
        try:
            new.load_state_dict(old.state_dict(), strict=True)
        except (RuntimeError, ValueError):
            # A new DiT positional scheme may intentionally differ. Only exact
            # matches are reused; no partial reshaping is hidden here.
            continue


def _block_kwargs(raw_denoiser: nn.Module, old_block: nn.Module) -> dict:
    old_layer = old_block.seqTransEncoder.layers[0]
    return {
        "input_dim": -1,
        "skeleton": raw_denoiser.motion_rep.skeleton,
        "llm_shape": raw_denoiser.llm_shape,
        "use_text_mask": old_block.use_text_mask,
        "latent_dim": raw_denoiser.latent_dim,
        "ff_size": old_layer.linear1.out_features,
        "num_layers": len(old_block.seqTransEncoder.layers),
        "num_heads": old_layer.nhead,
        "activation": "gelu",
        "dropout": old_layer.attn_dropout_p,
        "pe_dropout": old_block.sequence_pos_encoder.dropout.p,
        "norm_first": False,
        "input_first_heading_angle": hasattr(
            old_block, "linear_first_heading_angle"
        ),
        "add_input_proj": not isinstance(old_block.input_linear, nn.Identity),
        "positional_encoding_mode": raw_denoiser.positional_encoding_mode,
        "device": next(raw_denoiser.parameters()).device,
    }


def install_ardy_sequence_kernel(
    ardy_model: Any,
    config: ArdySequenceKernelConfig,
) -> ArdySequenceKernelClient:
    """Replace only ARDY's root/body mixers and return a lifecycle client."""
    if config.mode == "original":
        return ArdySequenceKernelClient(ardy_model, controller=None)

    raw = ardy_model.denoiser.model
    if getattr(raw, "sparsify_token_seq", False):
        raise ValueError(
            "the v1 accepted-token commit adapter requires dense ARDY token sequences"
        )
    old_root = raw.root_model
    old_body = raw.body_model

    common = {
        "sequence_mode": config.mode,
        "attention_backend": config.attention_backend,
        "window_size_chunks": config.window_size_chunks,
        "num_sink_tokens": config.num_sink_tokens,
        "allow_backend_fallback": config.allow_backend_fallback,
        "mamba_headdim": config.mamba_headdim,
        "mamba_d_state": config.mamba_d_state,
        "mamba_expand": config.mamba_expand,
        "mamba_chunk_size": config.mamba_chunk_size,
        "mamba_dtype": config.mamba_dtype,
        "token_state_interaction": config.token_state_interaction,
        "state_summary_dim": config.state_summary_dim,
        "use_state_delta": config.use_state_delta,
        "state_conditioning_scale_init": config.state_conditioning_scale_init,
        "state_aware_output_mode": config.state_aware_output_mode,
        "fusion_gate_logit_init": config.fusion_gate_logit_init,
        "content_memory_enabled": config.content_memory_enabled,
        "content_memory_slots": config.content_memory_slots,
        "content_memory_read_enabled": config.content_memory_read_enabled,
        "content_memory_use_state_delta": config.content_memory_use_state_delta,
        "content_memory_write_gate_context": config.content_memory_write_gate_context,
        "content_memory_decoder_init_scale": config.content_memory_decoder_init_scale,
        "content_memory_write_gate_init_scale": config.content_memory_write_gate_init_scale,
        "content_memory_write_gate_bias_init": config.content_memory_write_gate_bias_init,
        "content_memory_read_scale_init": config.content_memory_read_scale_init,
        "content_memory_conditioning": config.content_memory_conditioning,
        "content_memory_normalize_delta": config.content_memory_normalize_delta,
        "content_memory_normalize_slots": config.content_memory_normalize_slots,
        "content_memory_max_write_rate": config.content_memory_max_write_rate,
    }
    root_kwargs = _block_kwargs(raw, old_root)
    body_kwargs = _block_kwargs(raw, old_body)
    if config.ff_size is not None:
        root_kwargs["ff_size"] = config.ff_size
        body_kwargs["ff_size"] = config.ff_size
    root = ArdyDiTSequenceBlock(
        output_dim=old_root.output_linear.out_features,
        **root_kwargs,
        **common,
    )
    body = ArdyDiTSequenceBlock(
        output_dim=old_body.output_linear.out_features,
        **body_kwargs,
        **common,
    )
    _copy_compatible_frontend(old_root, root)
    _copy_compatible_frontend(old_body, body)
    # Constructors place ordinary frontend modules on CPU by default. The
    # mixer stack knows the requested device, but the whole replacement block
    # must follow the already-loaded ARDY denoiser or its first real CUDA
    # forward will mix CPU text/time projections with CUDA motion tensors.
    target_device = next(raw.parameters()).device
    root.to(device=target_device)
    body.to(device=target_device)
    raw.root_model = root
    raw.body_model = body

    controller = ArdyDiffusionMemoryController(ardy_model).install()
    return ArdySequenceKernelClient(ardy_model, controller)
