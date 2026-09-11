"""Thin integration layer over the closed attention and existing Mamba paths."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

from minimal_attention_kernel import MinimalAttentionKernel

from .config import SequenceKernelConfig
from .delta_rule import DeltaRuleStateCorrection
from .innovation import OutputInteraction
from .state import PendingSequenceKernelState, SequenceKernelState
from .token_state import TokenStateInteraction


ResetFn = Callable[[Any, Optional[torch.Tensor]], Any]


def _reset_tensor_state(state: Any, reset_mask: Optional[torch.Tensor]) -> Any:
    """Fallback used by lightweight injected wrappers in CPU interface tests."""
    if not isinstance(state, torch.Tensor):
        raise TypeError(
            "an injected non-tensor recurrent state requires mamba_reset_fn"
        )
    if reset_mask is None:
        return torch.zeros_like(state)
    if reset_mask.shape != (state.shape[0],) or reset_mask.dtype != torch.bool:
        raise ValueError(
            f"reset_mask must be bool with shape {(state.shape[0],)}, "
            f"got shape={tuple(reset_mask.shape)} dtype={reset_mask.dtype}"
        )
    shape = [state.shape[0]] + [1] * (state.ndim - 1)
    return torch.where(reset_mask.to(state.device).view(*shape), torch.zeros_like(state), state)


class SequenceKernel(nn.Module):
    """Generic sequence operator with exact minimal and opt-in hybrid modes.

    Minimal mode registers only :class:`MinimalAttentionKernel`. Hybrid mode
    additionally wraps the repository's existing ``Mamba3ChunkWrapper`` and
    fuses both outputs as ``attention + sigmoid(gate) * mamba``.

    ``forward_chunk(commit=False)`` is read-only with respect to committed
    memory. Call :meth:`commit` explicitly, or pass ``commit=True``.
    """

    def __init__(
        self,
        config: SequenceKernelConfig,
        *,
        mamba3: Optional[nn.Module] = None,
        mamba_wrapper: Optional[Any] = None,
        mamba_reset_fn: Optional[ResetFn] = None,
        state_projector: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.attention = MinimalAttentionKernel(config.attention)
        # Keep the minimal import/registration contract intact.  Hybrid-only
        # content memory is attached below, after the mode checks.
        self.content_memory = None
        self.last_content_memory_diagnostics: dict[str, float | bool | str | None] = {}
        self.record_memory_read_debug = False
        self.memory_read_debug_trace: list[dict] = []
        self.memory_read_debug_epoch = 0
        self.memory_read_debug_phase = "noisy_read"
        self.memory_read_debug_level = "slots"

        if config.mode == "minimal":
            if (
                mamba3 is not None
                or mamba_wrapper is not None
                or mamba_reset_fn is not None
                or state_projector is not None
            ):
                raise ValueError("minimal mode accepts no Mamba objects")
            self._mamba_wrapper = None
            self._mamba_reset_fn = None
            return

        if mamba3 is None:
            raise ValueError("hybrid mode requires the existing Mamba3 module")
        if not isinstance(mamba3, nn.Module):
            raise TypeError("mamba3 must be a torch.nn.Module")
        if (
            config.interaction
            in {"naive_gated_add", "h0_parallel", "h1_innovation", "h2_delta_rule"}
            and state_projector is not None
            and not config.content_memory_enabled
        ):
            raise ValueError("naive hybrid does not use a state_projector")

        # The import is intentionally deferred: constructing/importing minimal
        # mode must not import either state_memory_operator or mamba_ssm.
        if mamba_wrapper is None:
            from state_memory_operator.mamba3_chunk_wrapper import (
                Mamba3ChunkWrapper,
                apply_reset_mask,
            )

            mamba_wrapper = Mamba3ChunkWrapper(mamba3)
            mamba_reset_fn = apply_reset_mask
        elif mamba_reset_fn is None:
            mamba_reset_fn = _reset_tensor_state

        wrapped_module = getattr(mamba_wrapper, "mamba3", mamba3)
        if wrapped_module is not mamba3:
            raise ValueError("mamba_wrapper must wrap the supplied mamba3 module")

        self.mamba_path = mamba3
        self._mamba_wrapper = mamba_wrapper
        self._mamba_reset_fn = mamba_reset_fn
        self.runtime_mamba_output_scale = 1.0
        if config.interaction == "naive_gated_add":
            self.fusion_gate_logit = nn.Parameter(
                torch.tensor(float(config.fusion_gate_logit_init))
            )
            self.state_projector = state_projector if config.content_memory_enabled else None
            self.token_state_interaction = None
            self.output_interaction = None
            self.delta_rule = None
        elif config.interaction in {"h0_parallel", "h1_innovation"}:
            self.fusion_gate_logit = None
            self.state_projector = None
            self.token_state_interaction = None
            self.output_interaction = OutputInteraction(
                config.interaction,
                config.d_model,
                scale_init=config.state_conditioning_scale_init,
            )
            self.delta_rule = None
        elif config.interaction == "h2_delta_rule":
            self.fusion_gate_logit = None
            self.state_projector = None
            self.token_state_interaction = None
            self.output_interaction = None
            self.delta_rule = DeltaRuleStateCorrection(
                config.d_model,
                nheads=mamba3.nheads,
                headdim=mamba3.headdim,
                d_state=mamba3.d_state,
                scale_init=config.state_conditioning_scale_init,
            )
        else:
            if state_projector is None:
                from state_memory_operator.state_projector import StateProjector

                nheads = getattr(mamba3, "nheads", None)
                headdim = getattr(mamba3, "headdim", None)
                if nheads is None or headdim is None:
                    raise ValueError(
                        "C1 interactions require Mamba nheads/headdim or an "
                        "explicit state_projector"
                    )
                state_projector = StateProjector(
                    nheads=nheads,
                    headdim=headdim,
                    d_summary=config.state_summary_dim,
                )
            if not isinstance(state_projector, nn.Module):
                raise TypeError("state_projector must be a torch.nn.Module")
            self.state_projector = state_projector
            self.token_state_interaction = TokenStateInteraction(
                variant=config.interaction,
                d_model=config.d_model,
                num_heads=config.attention.num_heads,
                d_state_summary=config.state_summary_dim,
                use_state_delta=config.use_state_delta,
                scale_init=config.state_conditioning_scale_init,
            )
            self.output_interaction = None
            self.delta_rule = None
            self.fusion_gate_logit = (
                nn.Parameter(torch.tensor(float(config.fusion_gate_logit_init)))
                if config.state_aware_output_mode == "native_gated_add"
                else None
            )

        if config.content_memory_enabled:
            if self.mode != "hybrid":
                raise ValueError("content memory requires hybrid mode")
            # Memory-only ablations still project Mamba state for writes, but
            # do not inject that summary into the Attention query/key path.
            if self.state_projector is None:
                from state_memory_operator.state_projector import StateProjector

                nheads = getattr(mamba3, "nheads", None)
                headdim = getattr(mamba3, "headdim", None)
                if nheads is None or headdim is None:
                    raise ValueError(
                        "content memory requires Mamba nheads/headdim or an "
                        "explicit state_projector"
                    )
                self.state_projector = StateProjector(
                    nheads=nheads,
                    headdim=headdim,
                    d_summary=config.state_summary_dim,
                )
            from state_memory_operator.content_memory import PersistentContentMemory

            self.content_memory = PersistentContentMemory(
                state_dim=config.state_summary_dim,
                d_model=config.d_model,
                slots=config.content_memory_slots,
                write_gate_context=config.content_memory_write_gate_context,
                decoder_init_scale=config.content_memory_decoder_init_scale,
                write_gate_init_scale=config.content_memory_write_gate_init_scale,
                write_gate_bias_init=config.content_memory_write_gate_bias_init,
                read_scale_init=config.content_memory_read_scale_init,
                condition_dim=(
                    config.d_model
                    if config.content_memory_conditioning == "text"
                    else None
                ),
                normalize_delta=config.content_memory_normalize_delta,
                normalize_slots=config.content_memory_normalize_slots,
                max_write_rate=config.content_memory_max_write_rate,
            )

    @property
    def uses_token_state_interaction(self) -> bool:
        return self.mode == "hybrid" and self.config.interaction in {
            "state_query", "state_kv_modulation"
        }

    @property
    def uses_state_summary(self) -> bool:
        return self.mode == "hybrid" and (
            self.uses_token_state_interaction or self.content_memory is not None
        )

    @property
    def uses_output_interaction(self) -> bool:
        return self.mode == "hybrid" and self.output_interaction is not None

    @property
    def uses_delta_rule(self) -> bool:
        return self.mode == "hybrid" and self.delta_rule is not None

    @staticmethod
    def _raw_summary_state(mamba_state: Any) -> torch.Tensor:
        if isinstance(mamba_state, torch.Tensor):
            return mamba_state
        ssm_state = getattr(mamba_state, "ssm_state", None)
        if isinstance(ssm_state, torch.Tensor):
            return ssm_state
        raise TypeError("state projector requires a tensor or a state with .ssm_state")

    def _project_state(self, mamba_state: Any) -> torch.Tensor:
        return self.state_projector(self._raw_summary_state(mamba_state))

    def _mamba_parameter(self) -> torch.Tensor:
        parameter = getattr(getattr(self.mamba_path, "in_proj", None), "weight", None)
        if isinstance(parameter, torch.Tensor):
            return parameter
        return next(self.mamba_path.parameters())

    @property
    def mode(self) -> str:
        return self.config.mode

    def init_state(
        self,
        *,
        batch_size: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> SequenceKernelState:
        attention_cache = self.attention.init_cache(
            batch_size=batch_size, device=device, dtype=dtype
        )
        mamba_state = None
        if self.mode == "hybrid":
            init_state = getattr(self._mamba_wrapper, "init_state", None)
            if init_state is not None:
                mamba_parameter = self._mamba_parameter()
                mamba_state = init_state(
                    batch_size,
                    device=mamba_parameter.device,
                    dtype=mamba_parameter.dtype,
                )
        state_summary = None
        state_delta = None
        persistent_memory = None
        if self.uses_state_summary:
            if mamba_state is None:
                raise RuntimeError("state-aware memory requires an initialized Mamba state")
            state_summary = self._project_state(mamba_state)
            state_delta = torch.zeros_like(state_summary)
        if self.content_memory is not None:
            parameter = self.content_memory.content_decoder.weight
            persistent_memory = self.content_memory.initial(
                batch_size, device=parameter.device, dtype=parameter.dtype
            )
        return SequenceKernelState(
            attention_cache=attention_cache,
            mamba_state=mamba_state,
            source_kernel_id=id(self),
            state_summary=state_summary,
            state_delta=state_delta,
            persistent_memory=persistent_memory,
        )

    def _validate_state(self, state: SequenceKernelState, batch_size: int) -> None:
        if state.source_kernel_id != id(self):
            raise ValueError("state was created by a different sequence kernel")
        if state.attention_cache.batch_size != batch_size:
            raise ValueError(
                f"state batch size is {state.attention_cache.batch_size}, got input batch {batch_size}"
            )
        if self.mode == "minimal" and state.mamba_state is not None:
            raise ValueError("minimal state must not contain recurrent Mamba state")
        if self.uses_state_summary and (
            state.state_summary is None or state.state_delta is None
        ):
            raise ValueError("state-aware memory requires committed state_summary and state_delta")
        if self.content_memory is not None and state.persistent_memory is None:
            raise ValueError("content-memory state requires persistent_memory slots")

    def forward_chunk(
        self,
        x: torch.Tensor,
        *,
        state: SequenceKernelState,
        do_read: bool = True,
        commit: bool = False,
        chunk_id: Optional[int] = None,
        current_mask: Optional[torch.Tensor] = None,
        detach_on_commit: bool = True,
        enable_attention_path: bool = True,
        enable_mamba_path: Optional[bool] = None,
        content_memory_read_enabled: Optional[bool] = None,
        content_memory_write_enabled: bool = True,
        content_memory_delta: Optional[torch.Tensor] = None,
        content_memory_delta_mode: str = "normal",
        content_memory_condition: Optional[torch.Tensor] = None,
        content_memory_read_view: Optional[torch.Tensor] = None,
        content_memory_read_view_level: str = "slots",
    ) -> tuple[torch.Tensor, PendingSequenceKernelState, dict]:
        """Evaluate selected paths and return their uncommitted next memory."""
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,L,D], got {tuple(x.shape)}")
        self._validate_state(state, x.shape[0])

        if enable_mamba_path is None:
            enable_mamba_path = self.mode == "hybrid"
        if self.mode == "minimal" and enable_mamba_path:
            raise ValueError("Mamba path is unavailable in minimal mode")
        if not enable_attention_path and not enable_mamba_path:
            raise ValueError("at least one sequence path must be enabled")
        if not isinstance(content_memory_write_enabled, bool):
            raise TypeError("content_memory_write_enabled must be bool")
        if content_memory_delta_mode not in {"normal", "zero", "permute"}:
            raise ValueError("content_memory_delta_mode must be normal, zero, or permute")
        if content_memory_read_enabled is not None and not isinstance(
            content_memory_read_enabled, bool
        ):
            raise TypeError("content_memory_read_enabled must be bool or None")
        if current_mask is not None:
            expected_mask_shape = (x.shape[0], x.shape[1])
            if current_mask.shape != expected_mask_shape or current_mask.dtype != torch.bool:
                raise ValueError(
                    f"current_mask must be bool with shape {expected_mask_shape}, "
                    f"got shape={tuple(current_mask.shape)} dtype={current_mask.dtype}"
                )
            if enable_mamba_path and not bool(current_mask.all()):
                raise NotImplementedError(
                    "the existing dense Mamba chunk path cannot skip padded per-row "
                    "timesteps; use unpadded hybrid chunks"
                )

        attention_output = None
        attention_pending = None
        attention_aux = None
        query_bias = None
        history_chunk_scales = None
        interaction_aux = None
        memory_content = None
        read_residual_scale = None
        augmented_minus_base_norm = None
        memory_read_enabled = (
            self.config.content_memory_read_enabled
            if content_memory_read_enabled is None
            else content_memory_read_enabled
        )
        if content_memory_read_view_level not in {"slots", "read_ready"}:
            raise ValueError("content_memory_read_view_level must be slots or read_ready")
        durable_slots_debug = None
        if self.record_memory_read_debug and state.persistent_memory is not None:
            durable_slots_debug = state.persistent_memory.detach().clone()
        if self.content_memory is not None and do_read and memory_read_enabled:
            read_slots = state.persistent_memory
            if content_memory_read_view is not None:
                if content_memory_read_view.shape != state.persistent_memory.shape:
                    raise ValueError(
                        "content_memory_read_view must match persistent memory shape"
                    )
                # Read-view interventions are deliberately non-mutating.  The
                # durable slots remain the source for the write proposal below.
                read_slots = content_memory_read_view
            if content_memory_read_view is not None and content_memory_read_view_level == "read_ready":
                if read_slots.device != x.device or read_slots.dtype != x.dtype:
                    raise ValueError("read-ready external memory must match mixer input device/dtype")
                if not read_slots.is_contiguous():
                    raise ValueError("read-ready external memory must be contiguous")
                memory_content = self.content_memory.validate_read_ready(read_slots)
            else:
                memory_content = self.content_memory.read(read_slots, enabled=True).to(x.dtype)
            read_residual_scale = self.content_memory.read_residual_scale
            if self.record_memory_read_debug:
                self.memory_read_debug_trace.append({
                    "epoch": int(self.memory_read_debug_epoch),
                    "phase": str(self.memory_read_debug_phase),
                    "level": "read_ready",
                    "durable_slots": durable_slots_debug,
                    "read_ready": memory_content.detach().clone(),
                    "dtype": str(memory_content.dtype),
                    "device": str(memory_content.device),
                    "stride": tuple(memory_content.stride()),
                    "contiguous": bool(memory_content.is_contiguous()),
                })
        if enable_attention_path and self.uses_token_state_interaction:
            summary = state.state_summary
            delta = state.state_delta
            if not do_read:
                summary = torch.zeros_like(summary)
                delta = torch.zeros_like(delta)
            if self.config.interaction == "state_query":
                query_bias = self.token_state_interaction.query_bias(summary, delta).to(
                    dtype=x.dtype
                )
                interaction_aux = {
                    "query_bias_norm": query_bias.detach().float().norm(dim=-1).mean(),
                }
            else:
                history_chunk_scales, raw_scores = (
                    self.token_state_interaction.chunk_key_scales(
                        summary,
                        delta,
                        state.attention_cache,
                    )
                )
                interaction_aux = {
                    "chunk_key_scales": history_chunk_scales.detach(),
                    "chunk_key_scores": raw_scores.detach(),
                }
        if enable_attention_path:
            attention_output, attention_pending, attention_aux = self.attention.forward_chunk(
                x,
                cache=state.attention_cache,
                do_read=do_read,
                commit=False,
                current_mask=current_mask,
                query_bias=query_bias,
                history_chunk_scales=history_chunk_scales,
                extra_content=None,
            )
            if memory_content is not None:
                augmented_output, _, augmented_aux = self.attention.forward_chunk(
                    x,
                    cache=state.attention_cache,
                    do_read=do_read,
                    commit=False,
                    current_mask=current_mask,
                    query_bias=query_bias,
                    history_chunk_scales=history_chunk_scales,
                    extra_content=memory_content,
                )
                attention_difference = augmented_output - attention_output
                attention_output = attention_output + (
                    read_residual_scale.to(attention_difference.dtype)
                    * attention_difference
                )
                augmented_minus_base_norm = (
                    attention_difference.detach().float().norm(dim=-1).mean().item()
                )
                attention_aux = {
                    **attention_aux,
                    "augmented_backend_used": augmented_aux["backend_used"],
                    "persistent_content_tokens_read": memory_content.shape[1],
                    "augmented_minus_base_attention_norm": augmented_minus_base_norm,
                }

        mamba_output = None
        next_mamba_state = None
        if enable_mamba_path:
            mamba_parameter = self._mamba_parameter()
            mamba_input = x.to(
                device=mamba_parameter.device, dtype=mamba_parameter.dtype
            )
            mamba_output, next_mamba_state = self._mamba_wrapper.forward_chunk(
                mamba_input,
                state.mamba_state if do_read else None,
            )
            if self.uses_delta_rule:
                next_mamba_state, interaction_aux = self.delta_rule.correct(
                    next_mamba_state,
                    attention_output if attention_output is not None else x,
                )

        next_state_summary = None
        next_state_delta = None
        if self.uses_state_summary and next_mamba_state is not None:
            next_state_summary = self._project_state(next_mamba_state)
            next_state_delta = next_state_summary - state.state_summary

        next_memory = None
        content_pending = None
        if self.content_memory is not None:
            next_memory = state.persistent_memory
            if (
                next_state_summary is not None
                and next_state_delta is not None
                and content_memory_write_enabled
            ):
                write_delta = next_state_delta
                if content_memory_delta is not None:
                    if content_memory_delta.shape != next_state_delta.shape or content_memory_delta.device != next_state_delta.device:
                        raise ValueError("content_memory_delta must match next state_delta")
                    write_delta = content_memory_delta.to(next_state_delta.dtype)
                elif content_memory_delta_mode == "zero":
                    write_delta = torch.zeros_like(next_state_delta)
                elif content_memory_delta_mode == "permute":
                    if next_state_delta.shape[0] <= 1:
                        raise ValueError("permute content delta requires batch size > 1")
                    write_delta = torch.roll(next_state_delta, shifts=1, dims=0)
                if not self.config.content_memory_use_state_delta:
                    write_delta = torch.zeros_like(write_delta)
                content_pending = self.content_memory.propose(
                    next_state_summary,
                    write_delta,
                    state.persistent_memory,
                    condition=content_memory_condition,
                )
                next_memory = content_pending.slots

        if attention_output is not None and mamba_output is not None:
            if self.uses_output_interaction:
                output, interaction_aux = self.output_interaction(
                    attention_output, mamba_output
                )
            elif self.uses_token_state_interaction:
                if self.config.state_aware_output_mode == "native_gated_add":
                    gate = (
                        torch.sigmoid(self.fusion_gate_logit)
                        * float(self.runtime_mamba_output_scale)
                    ).to(mamba_output.dtype)
                    output = attention_output + (gate * mamba_output).to(
                        attention_output.dtype
                    )
                else:
                    # Historical C behavior: Mamba only builds durable address
                    # state and does not directly contribute output values.
                    output = attention_output
            elif self.uses_delta_rule:
                output = attention_output + (
                    self.delta_rule.output_scale * mamba_output.float()
                ).to(attention_output.dtype)
            else:
                gate = (
                    torch.sigmoid(self.fusion_gate_logit)
                    * float(self.runtime_mamba_output_scale)
                ).to(mamba_output.dtype)
                output = attention_output + (gate * mamba_output).to(
                    attention_output.dtype
                )
        elif attention_output is not None:
            output = attention_output
        else:
            output = mamba_output

        pending = PendingSequenceKernelState(
            attention_kv=attention_pending,
            mamba_state=next_mamba_state,
            source_kernel_id=id(self),
            state_summary=next_state_summary,
            state_delta=next_state_delta,
            persistent_memory=next_memory,
            base_commit_version=state.commit_version,
        )
        if commit:
            self.commit(
                state,
                pending,
                chunk_id=chunk_id,
                detach=detach_on_commit,
            )

        aux = {
            "mode": self.mode,
            "attention_path_enabled": enable_attention_path,
            "mamba_path_enabled": bool(enable_mamba_path),
            "fusion": (
                (
                    "gated_add"
                    if self.config.interaction == "naive_gated_add"
                    else self.config.interaction
                )
                if attention_output is not None and mamba_output is not None
                else None
            ),
            "fusion_gate": (
                float(
                    (
                        torch.sigmoid(self.fusion_gate_logit.detach())
                        * float(self.runtime_mamba_output_scale)
                    ).cpu()
                )
                if self.mode == "hybrid"
                and self.fusion_gate_logit is not None
                else None
            ),
            "state_conditioning_scale": (
                float(
                    (
                        self.output_interaction.scale
                        if self.uses_output_interaction
                        else (
                            self.delta_rule.beta
                            if self.uses_delta_rule
                            else self.token_state_interaction.effective_scale
                        )
                    ).detach().cpu()
                )
                if self.uses_token_state_interaction or self.uses_output_interaction or self.uses_delta_rule
                else None
            ),
            "committed_state_norm": (
                None
                if state.state_summary is None
                else state.state_summary.detach().float().norm(dim=-1).mean().item()
            ),
            "committed_delta_norm": (
                None
                if state.state_delta is None
                else state.state_delta.detach().float().norm(dim=-1).mean().item()
            ),
            "interaction": interaction_aux,
            "attention": attention_aux,
            "state_committed": commit,
            "persistent_content_memory": {
                "enabled": self.content_memory is not None,
                "read_enabled": bool(memory_content is not None),
                "write_enabled": content_memory_write_enabled,
                "uses_state_delta": self.config.content_memory_use_state_delta,
                "slots": 0 if self.content_memory is None else self.config.content_memory_slots,
                "read_norm": None if memory_content is None else memory_content.float().norm(dim=-1).mean().item(),
                "read_residual_scale": (
                    None
                    if self.content_memory is None
                    else float(self.content_memory.read_residual_scale.detach().cpu())
                ),
                "write_gate_context": (
                    None
                    if self.content_memory is None
                    else self.content_memory.write_gate_context
                ),
                "conditioning": self.config.content_memory_conditioning,
                "normalize_delta": self.config.content_memory_normalize_delta,
                "normalize_slots": self.config.content_memory_normalize_slots,
                "max_write_rate": self.config.content_memory_max_write_rate,
                "memory_slot_norm": (
                    None
                    if state.persistent_memory is None
                    else state.persistent_memory.detach().float().norm(dim=-1).mean().item()
                ),
                "memory_update_norm": (
                    None
                    if content_pending is None
                    else content_pending.update.detach().float().norm(dim=-1).mean().item()
                ),
                "write_gate_mean": (
                    None
                    if content_pending is None
                    else content_pending.write_gate.detach().float().mean().item()
                ),
                "write_gate_min": (
                    None
                    if content_pending is None
                    else content_pending.write_gate.detach().float().min().item()
                ),
                "write_gate_max": (
                    None
                    if content_pending is None
                    else content_pending.write_gate.detach().float().max().item()
                ),
                "state_summary_norm": (
                    None
                    if next_state_summary is None
                    else next_state_summary.detach().float().norm(dim=-1).mean().item()
                ),
                "state_delta_norm": (
                    None
                    if next_state_delta is None
                    else next_state_delta.detach().float().norm(dim=-1).mean().item()
                ),
                "content_proposal_norm": (
                    None
                    if content_pending is None
                    else content_pending.proposal.detach().float().norm(dim=-1).mean().item()
                ),
                "augmented_minus_base_attention_norm": augmented_minus_base_norm,
            },
        }
        self.last_content_memory_diagnostics = aux["persistent_content_memory"]
        return output, pending, aux

    def commit(
        self,
        state: SequenceKernelState,
        pending: PendingSequenceKernelState,
        *,
        chunk_id: Optional[int] = None,
        detach: bool = True,
    ) -> SequenceKernelState:
        """Accept exactly one candidate attention chunk and recurrent state."""
        self._validate_state(state, state.attention_cache.batch_size)
        if pending.source_kernel_id != id(self):
            raise ValueError("pending state was produced by a different sequence kernel")
        if pending.base_commit_version != state.commit_version:
            raise RuntimeError(
                "pending state is stale or was already committed; each accepted window "
                "must be committed exactly once"
            )
        if pending.attention_kv is not None:
            if chunk_id is None:
                raise ValueError("committing attention K/V requires chunk_id")
            self.attention.commit(
                state.attention_cache,
                pending.attention_kv,
                chunk_id=chunk_id,
                detach=detach,
            )
        if pending.mamba_state is not None:
            state.mamba_state = (
                pending.mamba_state.detach() if detach else pending.mamba_state
            )
        if pending.state_summary is not None:
            state.state_summary = (
                pending.state_summary.detach() if detach else pending.state_summary
            )
        if pending.state_delta is not None:
            state.state_delta = (
                pending.state_delta.detach() if detach else pending.state_delta
            )
        if pending.persistent_memory is not None:
            state.persistent_memory = (
                pending.persistent_memory.detach() if detach else pending.persistent_memory
            )
        state.commit_version += 1
        return state

    def reset_state(
        self,
        state: SequenceKernelState,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> SequenceKernelState:
        """Reset attention history and recurrent state without conflating them."""
        self._validate_state(state, state.attention_cache.batch_size)
        self.attention.reset_cache(state.attention_cache, reset_mask)
        if self.mode == "hybrid" and state.mamba_state is not None:
            recurrent_reset_mask = reset_mask
            if recurrent_reset_mask is None:
                recurrent_reset_mask = torch.ones(
                    state.attention_cache.batch_size,
                    dtype=torch.bool,
                    device=state.attention_cache.device,
                )
            state.mamba_state = self._mamba_reset_fn(
                state.mamba_state, recurrent_reset_mask
            )
            if self.uses_token_state_interaction:
                projected = self._project_state(state.mamba_state)
                state.state_summary = projected
                state.state_delta = torch.where(
                    recurrent_reset_mask[:, None],
                    torch.zeros_like(projected),
                    state.state_delta,
                )
        if self.content_memory is not None and state.persistent_memory is not None:
            state.persistent_memory = self.content_memory.reset(state.persistent_memory, reset_mask)
        state.commit_version += 1
        return state
