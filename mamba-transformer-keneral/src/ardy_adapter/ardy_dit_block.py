"""ARDY-compatible DiT block backed by the generic SequenceKernel stack."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Optional

import torch
from torch import Tensor, nn

from dit_sequence_kernel import DiTSequenceStack, DiTSequenceState
from minimal_attention_kernel import MinimalAttentionKernelConfig
from sequence_kernel import SequenceKernel, SequenceKernelConfig


def _import_ardy_backbone():
    # ARDY is a downstream dependency and is intentionally imported lazily.
    from ardy.model.backbone import (
        PositionalEncoding,
        PositionalEncodingNegativeIndex,
        TimestepEmbedder,
        pad_x_and_mask_to_fixed_size,
    )

    return (
        PositionalEncoding,
        PositionalEncodingNegativeIndex,
        TimestepEmbedder,
        pad_x_and_mask_to_fixed_size,
    )


MambaFactory = Callable[[int, int, torch.device | str | None, torch.dtype], tuple]


class ArdyDiTSequenceBlock(nn.Module):
    """Drop-in ARDY root/body block with DiT conditioning and generic mixers.

    Text, diffusion timestep, and optional first-heading information become a
    DiT conditioning vector. Only motion tokens enter the sequence mixer, so
    cross-window cache commits never persist noisy timestep/text prefix tokens.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        skeleton=None,
        llm_shape,
        use_text_mask: bool,
        latent_dim: int,
        ff_size: int,
        num_layers: int,
        num_heads: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        pe_dropout: float = 0.0,
        norm_first: bool = False,
        input_first_heading_angle: bool = False,
        add_input_proj: bool = True,
        positional_encoding_mode: str = "default",
        sequence_mode: Literal["dit", "minimal", "hybrid"] = "minimal",
        attention_backend: str = "flash2",
        window_size_chunks: int = 4,
        num_sink_tokens: int = 4,
        allow_backend_fallback: bool = True,
        mamba_headdim: int = 64,
        mamba_d_state: int = 64,
        mamba_expand: int = 2,
        mamba_chunk_size: int = 16,
        mamba_dtype: torch.dtype = torch.bfloat16,
        token_state_interaction: Literal[
            "naive_gated_add", "state_query", "state_kv_modulation",
            "h0_parallel", "h1_innovation", "h2_delta_rule"
        ] = "naive_gated_add",
        state_summary_dim: int = 128,
        use_state_delta: bool = True,
        state_conditioning_scale_init: float = 0.0,
        state_aware_output_mode: Literal[
            "attention_only", "native_gated_add"
        ] = "attention_only",
        fusion_gate_logit_init: float = 0.0,
        content_memory_enabled: bool = False,
        content_memory_slots: int = 1,
        content_memory_read_enabled: bool = True,
        content_memory_use_state_delta: bool = True,
        content_memory_write_gate_context: Literal[
            "previous_memory", "proposal"
        ] = "previous_memory",
        content_memory_decoder_init_scale: float = 0.02,
        content_memory_write_gate_init_scale: float = 0.02,
        content_memory_write_gate_bias_init: float = 0.0,
        content_memory_read_scale_init: float = 0.0,
        content_memory_conditioning: Literal["none", "text"] = "none",
        content_memory_normalize_delta: bool = False,
        content_memory_normalize_slots: bool = False,
        content_memory_max_write_rate: float = 1.0,
        mamba_factory: Optional[MambaFactory] = None,
        device=None,
        **_ignored,
    ) -> None:
        super().__init__()
        if sequence_mode not in {"dit", "minimal", "hybrid"}:
            raise ValueError(
                "sequence_mode must be 'dit', 'minimal', or 'hybrid'"
            )
        if activation != "gelu":
            raise ValueError("DiT v1 supports activation='gelu'")
        if norm_first:
            raise ValueError("DiT owns its AdaLN ordering; norm_first must be False")

        (
            PositionalEncoding,
            PositionalEncodingNegativeIndex,
            TimestepEmbedder,
            pad_x_and_mask_to_fixed_size,
        ) = _import_ardy_backbone()
        self._pad_x_and_mask_to_fixed_size = pad_x_and_mask_to_fixed_size
        self.sequence_mode = sequence_mode
        self.latent_dim = latent_dim
        self.num_text_tokens = llm_shape[0]
        self.use_text_mask = use_text_mask
        self.input_first_heading_angle = input_first_heading_angle
        self.positional_encoding_mode = positional_encoding_mode
        self.nbjoints = getattr(skeleton, "nbjoints", None)

        self.embed_text = nn.Linear(llm_shape[-1], latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, pe_dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        self.input_linear = (
            nn.Linear(input_dim, latent_dim) if add_input_proj else nn.Identity()
        )
        self.output_linear = nn.Linear(latent_dim, output_dim)
        if input_first_heading_angle:
            self.linear_first_heading_angle = nn.Linear(2, latent_dim)

        if positional_encoding_mode == "learned_prefix_zero_at_first_generation":
            self.motion_token_embedding = PositionalEncodingNegativeIndex(
                latent_dim, pe_dropout
            )
        elif positional_encoding_mode != "default":
            raise ValueError(
                f"invalid positional_encoding_mode: {positional_encoding_mode!r}"
            )
        self.condition_norm = nn.LayerNorm(latent_dim)

        use_sinks = sequence_mode != "dit"
        effective_window = 0 if sequence_mode == "dit" else window_size_chunks
        effective_sinks = num_sink_tokens if use_sinks else 0
        kernel_mode = "hybrid" if sequence_mode == "hybrid" else "minimal"

        def mixer_factory(layer_index: int) -> SequenceKernel:
            attention = MinimalAttentionKernelConfig(
                d_model=latent_dim,
                num_heads=num_heads,
                window_size_chunks=effective_window,
                use_attention_sinks=use_sinks,
                num_sink_tokens=effective_sinks,
                attention_backend=attention_backend,
                allow_backend_fallback=allow_backend_fallback,
                attention_dropout=dropout,
            )
            config = SequenceKernelConfig(
                mode=kernel_mode,
                attention=attention,
                interaction=token_state_interaction,
                state_summary_dim=state_summary_dim,
                use_state_delta=use_state_delta,
                state_conditioning_scale_init=state_conditioning_scale_init,
                state_aware_output_mode=state_aware_output_mode,
                fusion_gate_logit_init=fusion_gate_logit_init,
                content_memory_enabled=content_memory_enabled,
                content_memory_slots=content_memory_slots,
                content_memory_read_enabled=content_memory_read_enabled,
                content_memory_use_state_delta=content_memory_use_state_delta,
                content_memory_write_gate_context=content_memory_write_gate_context,
                content_memory_decoder_init_scale=content_memory_decoder_init_scale,
                content_memory_write_gate_init_scale=content_memory_write_gate_init_scale,
                content_memory_write_gate_bias_init=content_memory_write_gate_bias_init,
                content_memory_read_scale_init=content_memory_read_scale_init,
                content_memory_conditioning=content_memory_conditioning,
                content_memory_normalize_delta=content_memory_normalize_delta,
                content_memory_normalize_slots=content_memory_normalize_slots,
                content_memory_max_write_rate=content_memory_max_write_rate,
            )
            if kernel_mode == "minimal":
                return SequenceKernel(config)

            if mamba_factory is None:
                from mamba_ssm.modules.mamba3 import Mamba3

                mamba = Mamba3(
                    d_model=latent_dim,
                    d_state=mamba_d_state,
                    expand=mamba_expand,
                    headdim=mamba_headdim,
                    ngroups=1,
                    is_mimo=False,
                    chunk_size=mamba_chunk_size,
                    layer_idx=layer_index,
                    device=device,
                    dtype=mamba_dtype,
                )
                return SequenceKernel(config, mamba3=mamba)

            built = mamba_factory(layer_index, latent_dim, device, mamba_dtype)
            if len(built) == 2:
                mamba, wrapper = built
                reset_fn = None
            elif len(built) == 3:
                mamba, wrapper, reset_fn = built
            else:
                raise ValueError("mamba_factory must return 2 or 3 items")
            return SequenceKernel(
                config,
                mamba3=mamba,
                mamba_wrapper=wrapper,
                mamba_reset_fn=reset_fn,
            )

        self.sequence_stack = DiTSequenceStack(
            d_model=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            mixer_factory=mixer_factory,
            dropout=dropout,
        )
        # Preserve the trained Minimal/DiT stack in fp32. Mamba is constructed
        # in its requested compute dtype and SequenceKernel owns the explicit
        # cast at that path boundary; recurrent state and attention K/V need
        # not share a dtype.
        if device is not None:
            self.sequence_stack.to(device=device)
        if sequence_mode == "hybrid":
            # Optimizer-sized updates around zero (C1 beta) or a negative
            # logit (naive control) are below bf16 resolution. Keep only
            # these scalar coefficients in fp32; tensor paths stay bf16.
            for layer in self.sequence_stack.layers:
                mixer = layer.mixer
                if mixer.uses_token_state_interaction:
                    mixer.token_state_interaction.scale.data = (
                        mixer.token_state_interaction.scale.data.float()
                    )
                    if mixer.fusion_gate_logit is not None:
                        mixer.fusion_gate_logit.data = (
                            mixer.fusion_gate_logit.data.float()
                        )
                    if mixer.content_memory is not None:
                        mixer.content_memory.read_gate.data = (
                            mixer.content_memory.read_gate.data.float()
                        )
                elif mixer.uses_output_interaction:
                    mixer.output_interaction.scale.data = (
                        mixer.output_interaction.scale.data.float()
                    )
                elif mixer.uses_delta_rule:
                    mixer.delta_rule.beta_logit.data = (
                        mixer.delta_rule.beta_logit.data.float()
                    )
                    mixer.delta_rule.output_scale.data = (
                        mixer.delta_rule.output_scale.data.float()
                    )
                else:
                    mixer.fusion_gate_logit.data = mixer.fusion_gate_logit.data.float()

        self._sequence_state: Optional[DiTSequenceState] = None
        self._accepted_commit: Optional[tuple[torch.Tensor, int, bool]] = None
        self.content_memory_read_enabled_runtime: bool | None = None
        self.content_memory_write_enabled_runtime = True
        self.content_memory_delta_mode_runtime = "normal"
        self.content_memory_read_view_runtime: list[torch.Tensor] | None = None
        self.content_memory_read_view_level_runtime = "slots"
        self.read_only_forwards = 0
        self.accepted_commits = 0

    @property
    def has_cross_window_memory(self) -> bool:
        return self.sequence_mode in {"minimal", "hybrid"}

    def _stack_parameter(self) -> torch.Tensor:
        # A scalar C1/fusion coefficient may intentionally remain fp32 while
        # the tensor mixer runs bf16. Anchor ARDY tensor conversion to the
        # actual attention projection, not arbitrary module traversal order.
        return self.sequence_stack.layers[0].mixer.attention.q_proj.weight

    def _ensure_state(self, batch_size: int) -> DiTSequenceState:
        parameter = self._stack_parameter()
        if self._sequence_state is None:
            self._sequence_state = self.sequence_stack.init_state(
                batch_size=batch_size,
                device=parameter.device,
                dtype=parameter.dtype,
            )
        first_cache = self._sequence_state.layers[0].attention_cache
        if (
            first_cache.batch_size != batch_size
            or first_cache.device != parameter.device
            or first_cache.dtype != parameter.dtype
        ):
            raise RuntimeError(
                "ARDY sequence memory batch/device/dtype changed; call reset_memory() "
                "before starting a new generation"
            )
        return self._sequence_state

    def reset_memory(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        if self._sequence_state is None:
            return
        if reset_mask is None:
            # Recreate lazily. This also safely handles optimizer/device changes.
            self._sequence_state = None
        else:
            self.sequence_stack.reset_state(self._sequence_state, reset_mask)
        self._accepted_commit = None

    def prepare_accepted_commit(
        self,
        generation_token_mask: torch.Tensor,
        *,
        chunk_id: int,
        detach: bool = True,
    ) -> None:
        if not self.has_cross_window_memory:
            return
        if generation_token_mask.dtype != torch.bool or generation_token_mask.ndim != 2:
            raise ValueError("generation_token_mask must be bool [B,T]")
        self._accepted_commit = (generation_token_mask, chunk_id, detach)

    def cancel_accepted_commit(self) -> None:
        self._accepted_commit = None

    def _encode(
        self,
        x: Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: Tensor,
        first_heading_angle: Optional[Tensor],
        token_index: Optional[Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        motion = self.input_linear(x)
        if self.num_text_tokens is not None:
            text_feat, text_feat_pad_mask = self._pad_x_and_mask_to_fixed_size(
                text_feat, text_feat_pad_mask, self.num_text_tokens
            )
        embedded_text = self.embed_text(text_feat.to(self.embed_text.weight.dtype))
        if not self.use_text_mask:
            text_feat_pad_mask = torch.ones(
                embedded_text.shape[:2], dtype=torch.bool, device=x.device
            )
        text_weights = text_feat_pad_mask[:, :, None].to(embedded_text.dtype)
        pooled_text = (embedded_text * text_weights).sum(dim=1) / text_weights.sum(
            dim=1
        ).clamp_min(1.0)
        condition = pooled_text + self.embed_timestep(timesteps).squeeze(1)
        if self.input_first_heading_angle:
            if first_heading_angle is None:
                raise ValueError("first_heading_angle is required")
            heading = torch.stack(
                [torch.cos(first_heading_angle), torch.sin(first_heading_angle)], dim=-1
            )
            condition = condition + self.linear_first_heading_angle(heading)
        condition = self.condition_norm(condition)
        # Persistent writes receive prompt semantics explicitly. This excludes
        # diffusion timestep and heading so identical state transitions can be
        # interpreted differently under different text without noisy-step data.
        memory_condition = torch.nn.functional.layer_norm(
            pooled_text.float(), (self.latent_dim,)
        ).to(motion.dtype)

        if self.positional_encoding_mode == "default":
            motion = self.sequence_pos_encoder(motion)
        else:
            if token_index is None:
                raise ValueError("token_index is required for negative-index encoding")
            motion = self.motion_token_embedding(motion, token_index)
        valid = x_pad_mask.to(device=motion.device, dtype=torch.bool)
        motion = motion * valid[:, :, None].to(motion.dtype)
        return motion, condition, valid, memory_condition

    @staticmethod
    def _select_equal_count(
        tensor: torch.Tensor, selection: torch.Tensor
    ) -> torch.Tensor:
        counts = selection.sum(dim=1)
        if counts.numel() == 0 or int(counts.min()) == 0:
            raise ValueError("accepted commit must select at least one token per row")
        if not torch.equal(counts, counts[:1].expand_as(counts)):
            raise ValueError("accepted commit requires the same token count in every batch row")
        count = int(counts[0])
        return tensor[selection].reshape(tensor.shape[0], count, *tensor.shape[2:])

    def forward(
        self,
        x: Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: Tensor,
        first_heading_angle: Optional[Tensor] = None,
        token_index: Optional[Tensor] = None,
    ) -> Tensor:
        motion, condition, valid, memory_condition = self._encode(
            x,
            x_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
            token_index,
        )
        parameter = self._stack_parameter()
        motion = motion.to(device=parameter.device, dtype=parameter.dtype)
        condition = condition.to(device=parameter.device, dtype=parameter.dtype)
        memory_condition = memory_condition.to(
            device=parameter.device, dtype=parameter.dtype
        )
        valid = valid.to(parameter.device)
        if self._sequence_state is None:
            state = self._ensure_state(x.shape[0])
        else:
            state = self._sequence_state
        state_batch = state.layers[0].attention_cache.batch_size
        if x.shape[0] % state_batch != 0:
            raise RuntimeError(
                "ARDY CFG batch must be an integer multiple of persistent memory "
                f"batch size; got input={x.shape[0]}, memory={state_batch}"
            )
        if self._accepted_commit is not None and x.shape[0] != state_batch:
            raise RuntimeError("accepted commits must use the unexpanded ARDY batch")

        # ARDY classifier-free guidance evaluates 2 or 3 branches in one
        # expanded batch. Persistent memory remains at the user batch B; each
        # guidance branch reads the same B-row history independently. Forward
        # is read-only, so sharing the committed state cannot mutate it.
        hidden_parts = []
        for start in range(0, x.shape[0], state_batch):
            stop = start + state_batch
            part_valid = valid[start:stop]
            # The existing Mamba kernel is dense recurrence. Invalid ARDY
            # tokens are zeroed, while minimal attention gets an exact mask.
            stack_mask = part_valid if self.sequence_mode != "hybrid" else None
            part_hidden, _, _ = self.sequence_stack(
                motion[start:stop],
                condition[start:stop],
                state=state,
                commit=False,
                current_mask=stack_mask,
                content_memory_read_enabled=self.content_memory_read_enabled_runtime,
                content_memory_write_enabled=self.content_memory_write_enabled_runtime,
                content_memory_delta_mode=self.content_memory_delta_mode_runtime,
                content_memory_condition=memory_condition[start:stop],
                content_memory_read_view=(
                    None
                    if self.content_memory_read_view_runtime is None
                    else self.content_memory_read_view_runtime
                ),
                content_memory_read_view_level=self.content_memory_read_view_level_runtime,
            )
            hidden_parts.append(part_hidden)
        hidden = torch.cat(hidden_parts, dim=0)
        self.read_only_forwards += 1

        if self._accepted_commit is not None:
            selection, chunk_id, detach = self._accepted_commit
            selection = selection.to(valid.device)
            if selection.shape != valid.shape:
                raise ValueError(
                    "accepted generation mask shape differs from ARDY block input; "
                    "sparsified-token mode is not supported by this adapter"
                )
            if bool((selection & ~valid).any()):
                raise ValueError("accepted generation mask selects invalid tokens")
            accepted_motion = self._select_equal_count(motion, selection)
            self.sequence_stack(
                accepted_motion,
                condition,
                state=state,
                commit=True,
                chunk_id=chunk_id,
                current_mask=None,
                detach_on_commit=detach,
                content_memory_read_enabled=self.content_memory_read_enabled_runtime,
                content_memory_write_enabled=self.content_memory_write_enabled_runtime,
                content_memory_delta_mode=self.content_memory_delta_mode_runtime,
                content_memory_condition=memory_condition,
                # Accepted writes use the trajectory's durable memory as the
                # write context; the read-view intervention only affects the
                # generated read path.
                content_memory_read_view=None,
                content_memory_read_view_level="slots",
            )
            self.accepted_commits += 1
            self._accepted_commit = None

        output = self.output_linear(hidden.to(self.output_linear.weight.dtype))
        return output * valid[:, :, None].to(output.dtype)
