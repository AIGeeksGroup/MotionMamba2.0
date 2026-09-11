"""ARDY diffusion lifecycle adapter: noisy reads, one accepted commit."""

from __future__ import annotations

import inspect
import copy
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .ardy_dit_block import ArdyDiTSequenceBlock
from .history_commit import canonical_history_prime


@dataclass(frozen=True)
class DiffusionMemoryTrace:
    completed_windows: int
    accepted_commits: int
    last_sampling_timestep: Optional[int]


class ArdyDiffusionMemoryController:
    """Attach commit semantics to ARDY without replacing its sampler loop.

    The original bound ``denoising_step`` remains responsible for diffusion.
    This controller observes its result. Every noisy call is naturally
    read-only because the DiT blocks never auto-commit. Only a call whose
    sampling index is zero triggers one extra clean block evaluation with an
    accepted-commit context.
    """

    def __init__(self, ardy_model: Any) -> None:
        self.ardy_model = ardy_model
        self.raw_denoiser = ardy_model.denoiser.model
        self.root_block = self.raw_denoiser.root_model
        self.body_block = self.raw_denoiser.body_model
        if not isinstance(self.root_block, ArdyDiTSequenceBlock) or not isinstance(
            self.body_block, ArdyDiTSequenceBlock
        ):
            raise TypeError("ARDY root/body models must be ArdyDiTSequenceBlock")
        if self.root_block.sequence_mode != self.body_block.sequence_mode:
            raise ValueError("root and body sequence modes must match")
        self._original_denoising_step = ardy_model.denoising_step
        self._signature = inspect.signature(self._original_denoising_step)
        self._installed = False
        self._next_chunk_id = 0
        self._completed_windows = 0
        self._accepted_commits = 0
        self._last_sampling_timestep: Optional[int] = None
        # A clean t=0 prediction is only eligible for memory after ARDY has
        # recentered the completed window. Committing inside denoising_step
        # stores the pre-recenter coordinate frame and causes boundary jumps.
        self._original_post_window_hook = getattr(
            ardy_model, "_sequence_kernel_post_window_hook", None
        )
        self._supports_post_window_hook = hasattr(ardy_model, "_recenter_history")
        self._last_text_feat: Optional[torch.Tensor] = None
        self._last_text_pad_mask: Optional[torch.Tensor] = None
        self.periodic_reset_windows = 0
        self.record_state_diagnostics = False
        self.state_diagnostics: list[dict[str, Any]] = []
        self.record_content_memory_tensors = False
        self.content_memory_tensor_trace: list[dict[str, Any]] = []
        self._previous_mamba_states: dict[str, list[Any]] = {}
        self._capture_training_snapshots = False
        self._training_snapshot_depths: set[int] = set()
        self.training_state_snapshots: dict[int, tuple[Any, Any]] = {}
        self.training_history_chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.mamba_output_decay_start_windows = 0
        self.mamba_output_decay_half_life_windows = 0.0
        self.mamba_output_decay_min_scale = 0.0
        self.state_intervention = "carry"
        self._frozen_mamba_states: dict[str, list[Any]] = {}
        self._last_intervention_checksum: Optional[str] = None
        self._intervention_applied_window: Optional[int] = None
        self.intervention_audit: list[dict[str, Any]] = []
        self.content_memory_intervention = "normal"
        self._content_memory_permutation_applied = False
        self._content_memory_permutation: Optional[torch.Tensor] = None
        self._content_memory_read_trace_window: Optional[int] = None
        self.content_memory_audit: list[dict[str, Any]] = []
        # Optional read-only cross-run memory source. The bank is keyed by
        # (donor_id, accepted_window, block, layer) and is never committed.
        self._external_memory_bank: Optional[dict[tuple[str, int, str, int], torch.Tensor]] = None
        self._external_memory_recipient_ids: Optional[list[str]] = None
        self._external_memory_donor_ids: Optional[list[str]] = None
        self._external_memory_mode: Optional[str] = None
        self._external_memory_level = "slots"

    def configure_content_memory_intervention(
        self,
        strategy: str = "normal",
        *,
        permutation: Optional[torch.Tensor] = None,
    ) -> None:
        allowed = {
            "normal", "zero_delta", "permute_delta", "write_off", "readout_off",
            "permute_memory", "persistent_read_permutation",
            "freeze_write_after_w4", "one_shot_permute_plus_freeze_write",
            "semantic_mismatch_persistent_read_permutation",
        }
        if strategy not in allowed:
            raise ValueError(f"unknown content-memory intervention {strategy!r}")
        if permutation is not None:
            if permutation.ndim != 1 or permutation.dtype != torch.long:
                raise ValueError("content-memory permutation must be a 1-D long tensor")
        self.content_memory_intervention = strategy
        self._content_memory_permutation = permutation.detach().clone() if permutation is not None else None
        self._content_memory_permutation_applied = False
        self._content_memory_read_trace_window = None
        self.content_memory_audit.clear()
        self._set_content_memory_runtime()

    def configure_external_memory_read(
        self,
        bank: Optional[dict[tuple[str, int, str, int], torch.Tensor]],
        *,
        recipient_ids: Optional[list[str]] = None,
        donor_ids: Optional[list[str]] = None,
        mode: str = "donor",
        level: str = "read_ready",
    ) -> None:
        """Install a fixed, read-only cross-run memory source for one batch.

        ``bank`` contains committed slots indexed by donor identity, accepted
        window index, block name and DiT layer. ``mode='own'`` is an explicit
        offline replay control; ``mode='donor'`` uses ``donor_ids``. Durable
        recipient state and all write proposals remain untouched.
        """
        if bank is None:
            self._external_memory_bank = None
            self._external_memory_recipient_ids = None
            self._external_memory_donor_ids = None
            self._external_memory_mode = None
            return
        if mode not in {"own", "donor"}:
            raise ValueError("external memory mode must be 'own' or 'donor'")
        if level not in {"slots", "read_ready"}:
            raise ValueError("external memory level must be slots or read_ready")
        if not recipient_ids or len(recipient_ids) == 0:
            raise ValueError("recipient_ids are required for external memory")
        if mode == "donor" and (not donor_ids or len(donor_ids) != len(recipient_ids)):
            raise ValueError("donor_ids must match recipient_ids for donor mode")
        if mode == "own":
            donor_ids = list(recipient_ids)
        if any(str(a) == str(b) for a, b in zip(recipient_ids, donor_ids)) and mode == "donor":
            raise ValueError("donor mapping must not contain self mappings")
        self._external_memory_bank = bank
        self._external_memory_recipient_ids = [str(x) for x in recipient_ids]
        self._external_memory_donor_ids = [str(x) for x in donor_ids]
        self._external_memory_mode = mode
        self._external_memory_level = level
        self._content_memory_read_trace_window = None
        self._set_content_memory_runtime()

    def configure_memory_read_debug(self, *, enabled: bool, epoch: int = 0, phase: str = "noisy_read") -> None:
        if phase not in {"noisy_read", "accepted_commit"}:
            raise ValueError("memory debug phase must be noisy_read or accepted_commit")
        for block in (self.root_block, self.body_block):
            for layer in block.sequence_stack.layers:
                layer.mixer.record_memory_read_debug = bool(enabled)
                layer.mixer.memory_read_debug_epoch = int(epoch)
                layer.mixer.memory_read_debug_phase = phase
                if not enabled:
                    layer.mixer.memory_read_debug_trace.clear()

    def _external_memory_views(self, completed_windows: int, block_name: str, block) -> list[Optional[torch.Tensor]]:
        if self._external_memory_bank is None or self._external_memory_donor_ids is None:
            return []
        if block._sequence_state is None:
            return []
        if len(self._external_memory_donor_ids) != block._sequence_state.layers[0].attention_cache.batch_size:
            raise ValueError("external memory mapping batch size does not match sequence state")
        views: list[Optional[torch.Tensor]] = []
        for layer_index, layer in enumerate(block._sequence_state.layers):
            if layer.persistent_memory is None:
                views.append(None)
                continue
            rows = []
            for donor_id in self._external_memory_donor_ids:
                key = (str(donor_id), int(completed_windows), block_name, int(layer_index))
                if key not in self._external_memory_bank:
                    raise KeyError(f"external memory bank missing {key}")
                rows.append(self._external_memory_bank[key])
            view = torch.stack(rows, dim=0).to(
                device=layer.persistent_memory.device,
                dtype=layer.persistent_memory.dtype,
            )
            if view.shape != layer.persistent_memory.shape:
                raise ValueError(f"external memory shape {tuple(view.shape)} != {tuple(layer.persistent_memory.shape)}")
            views.append(view)
        return views

    def _set_content_memory_runtime(self) -> None:
        strategy = getattr(self, "content_memory_intervention", "normal")
        read_enabled = not (strategy == "readout_off" and self._completed_windows >= 4)
        if strategy == "write_off":
            write_enabled = False
        elif strategy in {"freeze_write_after_w4", "one_shot_permute_plus_freeze_write"}:
            write_enabled = self._completed_windows < 4
        else:
            write_enabled = True
        mode = {
            "zero_delta": "zero",
            "permute_delta": "permute",
        }.get(strategy, "normal")
        for block in (self.root_block, self.body_block):
            block.content_memory_read_enabled_runtime = read_enabled
            block.content_memory_write_enabled_runtime = write_enabled
            block.content_memory_delta_mode_runtime = mode
            block.content_memory_read_view_runtime = None
            block.content_memory_read_view_level_runtime = "slots"

        if getattr(self, "_external_memory_bank", None) is not None and self._completed_windows >= 4:
            for block_name, block in (("root", self.root_block), ("body", self.body_block)):
                block.content_memory_read_view_runtime = self._external_memory_views(
                    self._completed_windows, block_name, block
                )
                block.content_memory_read_view_level_runtime = getattr(self, "_external_memory_level", "slots")
            if getattr(self, "_content_memory_read_trace_window", None) != self._completed_windows:
                self.content_memory_audit.append({
                    "event": "external_memory_read_window",
                    "completed_windows": self._completed_windows,
                    "mode": self._external_memory_mode,
                    "recipient_ids": list(self._external_memory_recipient_ids or []),
                    "donor_ids": list(self._external_memory_donor_ids or []),
                    "durable_memory_checksum_before": self._content_memory_checksum(),
                    "durable_memory_checksum_after": self._content_memory_checksum(),
                    "attention_kv_checksum": self._attention_checksum(),
                })
                self._content_memory_read_trace_window = self._completed_windows

        persistent = {
            "persistent_read_permutation",
            "semantic_mismatch_persistent_read_permutation",
        }
        if strategy in persistent and self._completed_windows >= 4:
            permutation = self._ensure_content_memory_permutation()
            for block in (self.root_block, self.body_block):
                state = block._sequence_state
                if state is None or not state.layers:
                    continue
                # Every layer reads the same sample identity mapping while
                # retaining its own independent K=2 durable slots.
                read_views = []
                for layer in state.layers:
                    if layer.persistent_memory is not None:
                        read_views.append(layer.persistent_memory.index_select(
                            0, permutation.to(layer.persistent_memory.device)
                        ))
                    else:
                        read_views.append(None)
                block.content_memory_read_view_runtime = read_views
                block.content_memory_read_view_level_runtime = "slots"
            if getattr(self, "_content_memory_read_trace_window", None) != self._completed_windows:
                durable = self._content_memory_checksum()
                read_digest = __import__("hashlib").sha256()
                for block in (self.root_block, self.body_block):
                    views = block.content_memory_read_view_runtime or []
                    for view in views:
                        if view is not None:
                            read_digest.update(
                                view.detach().float().cpu().contiguous().numpy().tobytes()
                            )
                mamba_digest = []
                for block in (self.root_block, self.body_block):
                    if block._sequence_state is not None:
                        mamba_digest.extend(
                            self._state_checksum(layer.mamba_state)
                            for layer in block._sequence_state.layers
                        )
                self.content_memory_audit.append({
                    "event": "persistent_read_window",
                    "completed_windows": self._completed_windows,
                    "permutation": permutation.detach().cpu().tolist(),
                    "durable_memory_checksum_before": durable,
                    "durable_memory_checksum_after": durable,
                    "read_memory_checksum": read_digest.hexdigest(),
                    "attention_kv_checksum": self._attention_checksum(),
                    "mamba_state_checksum": ":".join(mamba_digest),
                })
                self._content_memory_read_trace_window = self._completed_windows

    def _ensure_content_memory_permutation(self) -> torch.Tensor:
        states = [
            block._sequence_state
            for block in (self.root_block, self.body_block)
            if block._sequence_state is not None and block._sequence_state.layers
        ]
        if not states:
            raise RuntimeError("content-memory permutation requires initialized state")
        batch = states[0].layers[0].attention_cache.batch_size
        device = states[0].layers[0].attention_cache.device
        if batch <= 1:
            raise ValueError("persistent read permutation requires batch size > 1")
        if getattr(self, "_content_memory_permutation", None) is None:
            self._content_memory_permutation = torch.roll(
                torch.arange(batch, device=device), 1
            ).long().cpu()
        permutation = self._content_memory_permutation
        if permutation.shape != (batch,):
            raise ValueError(
                f"content-memory permutation shape {tuple(permutation.shape)} != {(batch,)}"
            )
        if sorted(permutation.tolist()) != list(range(batch)):
            raise ValueError("content-memory permutation must be a bijection")
        if any(int(value) == index for index, value in enumerate(permutation.tolist())):
            raise ValueError("content-memory permutation must be deranged")
        return permutation.to(device)

    @property
    def trace(self) -> DiffusionMemoryTrace:
        return DiffusionMemoryTrace(
            completed_windows=self._completed_windows,
            accepted_commits=self._accepted_commits,
            last_sampling_timestep=self._last_sampling_timestep,
        )

    @property
    def commit_enabled(self) -> bool:
        return self.root_block.has_cross_window_memory

    def reset(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        self.root_block.reset_memory(reset_mask)
        self.body_block.reset_memory(reset_mask)
        if reset_mask is None:
            self._next_chunk_id = 0
            self._completed_windows = 0
            self._accepted_commits = 0
            self._last_sampling_timestep = None
            self._previous_mamba_states.clear()
            self._capture_training_snapshots = False
            self._training_snapshot_depths.clear()
            self.training_state_snapshots.clear()
            self.training_history_chunks.clear()
            self._frozen_mamba_states.clear()
            self._last_intervention_checksum = None
            self._intervention_applied_window = None
            self.intervention_audit.clear()
            self._content_memory_permutation_applied = False
            self._content_memory_permutation = None
            self._content_memory_read_trace_window = None
            self.content_memory_audit.clear()
            self.content_memory_tensor_trace.clear()
            self._set_runtime_mamba_output_scale(1.0)
            self._set_content_memory_runtime()

    @staticmethod
    def _transform_state(state: Any, transform) -> Any:
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return transform(state)
        if isinstance(state, dict):
            return {key: ArdyDiffusionMemoryController._transform_state(value, transform) for key, value in state.items()}
        if isinstance(state, list):
            return [ArdyDiffusionMemoryController._transform_state(value, transform) for value in state]
        if isinstance(state, tuple):
            values = [
                ArdyDiffusionMemoryController._transform_state(value, transform)
                for value in state
            ]
            return tuple(values) if type(state) is tuple else type(state)(*values)
        result = copy.copy(state)
        changed = False
        for name in ("angle_dt_state", "ssm_state", "k_state", "v_state"):
            value = getattr(state, name, None)
            if value is not None:
                setattr(
                    result,
                    name,
                    ArdyDiffusionMemoryController._transform_state(value, transform),
                )
                changed = True
        if not changed:
            raise TypeError(f"unsupported recurrent state type: {type(state).__name__}")
        return result

    @staticmethod
    def _state_checksum(state: Any) -> str:
        import hashlib
        digest = hashlib.sha256()
        for tensor in ArdyDiffusionMemoryController._state_tensors(state):
            digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _attention_checksum(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        for block in (self.root_block, self.body_block):
            state = block._sequence_state
            if state is None:
                digest.update(b"none")
                continue
            for layer in state.layers:
                cache = layer.attention_cache
                digest.update(str(cache.chunk_ids).encode())
                for chunk in cache.chunks:
                    for tensor in (chunk.key, chunk.value, chunk.token_valid):
                        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def configure_state_intervention(self, strategy: str = "carry") -> None:
        allowed = {
            "carry", "fresh_each_window", "zero_after_w4", "freeze_after_w4",
            "permute_after_w4", "readout_off_after_w4", "no_decay",
        }
        if strategy not in allowed:
            raise ValueError(f"unknown state intervention {strategy!r}; choose from {sorted(allowed)}")
        self.state_intervention = strategy
        self._frozen_mamba_states.clear()
        self._last_intervention_checksum = None
        self._intervention_applied_window = None
        if strategy == "no_decay":
            self.configure_mamba_output_decay(start_windows=0, half_life_windows=0.0, min_scale=0.0)

    def reset_mamba_only(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        """Clear recurrent state while preserving every Attention K/V tensor."""
        attention_before = self._attention_checksum()
        for block in (self.root_block, self.body_block):
            sequence_state = block._sequence_state
            if sequence_state is None:
                continue
            if not sequence_state.layers:
                continue
            batch = sequence_state.layers[0].attention_cache.batch_size
            if reset_mask is not None and (
                reset_mask.shape != (batch,) or reset_mask.dtype != torch.bool
            ):
                raise ValueError("Mamba reset mask must be bool [batch]")
            for layer in sequence_state.layers:
                if layer.mamba_state is not None:
                    if reset_mask is None:
                        layer.mamba_state = self._transform_state(
                            layer.mamba_state, torch.zeros_like
                        )
                    else:
                        layer.mamba_state = self._transform_state(
                            layer.mamba_state,
                            lambda value: torch.where(
                                reset_mask.to(value.device).reshape(
                                    (batch,) + (1,) * (value.ndim - 1)
                                ),
                                torch.zeros_like(value),
                                value,
                            ),
                        )
                for name in ("state_summary", "state_delta"):
                    value = getattr(layer, name)
                    if value is not None:
                        if reset_mask is None:
                            setattr(layer, name, torch.zeros_like(value))
                        else:
                            setattr(
                                layer,
                                name,
                                torch.where(
                                    reset_mask.to(value.device)[:, None],
                                    torch.zeros_like(value),
                                    value,
                                ),
                            )
        if self._attention_checksum() != attention_before:
            raise RuntimeError("Mamba-only reset modified Attention K/V")

    def _apply_mamba_state_intervention(self) -> None:
        strategy = self.state_intervention
        self._set_content_memory_runtime()
        self._maybe_permute_content_memory()
        if strategy in {"carry", "no_decay"}:
            return
        if self._intervention_applied_window == self._completed_windows:
            return
        after_w4 = self._completed_windows >= 4
        if strategy in {"zero_after_w4", "freeze_after_w4", "permute_after_w4", "readout_off_after_w4"} and not after_w4:
            return
        attention_before = self._attention_checksum()
        mamba_before = []
        for block in (self.root_block, self.body_block):
            if block._sequence_state is not None:
                mamba_before.extend(
                    self._state_checksum(layer.mamba_state)
                    for layer in block._sequence_state.layers
                )
        for block_name, block in (("root", self.root_block), ("body", self.body_block)):
            sequence_state = block._sequence_state
            if sequence_state is None:
                continue
            if strategy == "freeze_after_w4":
                if block_name not in self._frozen_mamba_states:
                    self._frozen_mamba_states[block_name] = [
                        self._clone_mamba_state(layer.mamba_state)
                        for layer in sequence_state.layers
                    ]
                for layer, saved in zip(sequence_state.layers, self._frozen_mamba_states[block_name]):
                    layer.mamba_state = self._clone_mamba_state(saved)
            elif strategy in {"fresh_each_window", "zero_after_w4"}:
                for layer in sequence_state.layers:
                    layer.mamba_state = self._transform_state(layer.mamba_state, torch.zeros_like)
                    if layer.state_summary is not None:
                        layer.state_summary.zero_()
                    if layer.state_delta is not None:
                        layer.state_delta.zero_()
            elif strategy == "permute_after_w4":
                if not sequence_state.layers:
                    continue
                batch = sequence_state.layers[0].attention_cache.batch_size
                if batch <= 1:
                    raise ValueError("permute_after_w4 requires batch size > 1")
                permutation = torch.roll(
                    torch.arange(
                        batch,
                        device=sequence_state.layers[0].attention_cache.device,
                    ),
                    1,
                )
                for layer in sequence_state.layers:
                    layer.mamba_state = self._transform_state(
                        layer.mamba_state, lambda value: value.index_select(0, permutation.to(value.device))
                    )
                    if layer.state_summary is not None:
                        layer.state_summary = layer.state_summary.index_select(0, permutation.to(layer.state_summary.device))
                    if layer.state_delta is not None:
                        layer.state_delta = layer.state_delta.index_select(0, permutation.to(layer.state_delta.device))
            elif strategy == "readout_off_after_w4":
                self._set_runtime_mamba_output_scale(0.0)
        checksums = []
        for block in (self.root_block, self.body_block):
            if block._sequence_state is not None:
                checksums.extend(self._state_checksum(layer.mamba_state) for layer in block._sequence_state.layers)
        self._last_intervention_checksum = ":".join(checksums)
        self._intervention_applied_window = self._completed_windows
        attention_after = self._attention_checksum()
        if attention_before != attention_after:
            raise RuntimeError("Mamba-only intervention modified Attention K/V")
        self.intervention_audit.append({
            "strategy": strategy,
            "completed_windows": self._completed_windows,
            "attention_checksum_before": attention_before,
            "attention_checksum_after": attention_after,
            "mamba_checksum_before": ":".join(mamba_before),
            "mamba_checksum_after": self._last_intervention_checksum,
        })

    def _permute_content_memory(self) -> None:
        for block in (self.root_block, self.body_block):
            sequence_state = block._sequence_state
            if sequence_state is None or not sequence_state.layers:
                continue
            batch = sequence_state.layers[0].attention_cache.batch_size
            if batch <= 1:
                raise ValueError("permute_memory requires batch size > 1")
            permutation = self._ensure_content_memory_permutation()
            for layer in sequence_state.layers:
                if layer.persistent_memory is not None:
                    layer.persistent_memory = layer.persistent_memory.index_select(0, permutation.to(layer.persistent_memory.device))

    def _maybe_permute_content_memory(self) -> None:
        if (
            getattr(self, "content_memory_intervention", "normal")
            not in {"permute_memory", "one_shot_permute_plus_freeze_write"}
            or self._completed_windows < 4
            or self._content_memory_permutation_applied
        ):
            return
        attention_before = self._attention_checksum()
        memory_before = self._content_memory_checksum()
        self._permute_content_memory()
        attention_after = self._attention_checksum()
        if attention_after != attention_before:
            raise RuntimeError("content-memory permutation modified Attention K/V")
        self._content_memory_permutation_applied = True
        self.content_memory_audit.append({
            "strategy": "permute_memory",
            "completed_windows": self._completed_windows,
            "attention_checksum_before": attention_before,
            "attention_checksum_after": attention_after,
            "memory_checksum_before": memory_before,
            "memory_checksum_after": self._content_memory_checksum(),
        })

    def _content_memory_checksum(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        found = False
        for block in (self.root_block, self.body_block):
            state = block._sequence_state
            if state is None:
                continue
            for layer in state.layers:
                if layer.persistent_memory is not None:
                    found = True
                    digest.update(
                        layer.persistent_memory.detach().float().cpu().contiguous().numpy().tobytes()
                    )
        return digest.hexdigest() if found else "none"

    def configure_mamba_output_decay(
        self,
        *,
        start_windows: int = 0,
        half_life_windows: float = 0.0,
        min_scale: float = 0.0,
    ) -> None:
        """Decay only Hybrid's Mamba readout after a chosen rollout depth."""
        if start_windows < 0 or half_life_windows < 0:
            raise ValueError("Mamba output decay settings must be non-negative")
        if not 0.0 <= min_scale <= 1.0:
            raise ValueError("Mamba output decay minimum scale must be in [0, 1]")
        self.mamba_output_decay_start_windows = start_windows
        self.mamba_output_decay_half_life_windows = half_life_windows
        self.mamba_output_decay_min_scale = min_scale
        self._set_runtime_mamba_output_scale(1.0)

    def _set_runtime_mamba_output_scale(self, scale: float) -> None:
        for block in (self.root_block, self.body_block):
            for layer in block.sequence_stack.layers:
                layer.mixer.runtime_mamba_output_scale = float(scale)

    def _update_runtime_mamba_output_scale(self) -> None:
        self.set_runtime_mamba_output_scale_for_depth(self._completed_windows)

    def set_runtime_mamba_output_scale_for_depth(
        self, completed_windows: int
    ) -> None:
        if completed_windows < 0:
            raise ValueError("completed_windows must be non-negative")
        half_life = self.mamba_output_decay_half_life_windows
        if not half_life:
            self._set_runtime_mamba_output_scale(1.0)
            return
        elapsed = max(
            0, completed_windows - self.mamba_output_decay_start_windows
        )
        scale = max(
            self.mamba_output_decay_min_scale,
            2.0 ** (-elapsed / half_life),
        )
        self._set_runtime_mamba_output_scale(scale)

    def configure_rollout_diagnostics(
        self,
        *,
        enabled: bool,
        periodic_reset_windows: int = 0,
    ) -> None:
        """Configure per-commit state tracing and a generated-window reset ablation."""
        if periodic_reset_windows < 0:
            raise ValueError("periodic_reset_windows must be non-negative")
        self.record_state_diagnostics = enabled
        self.periodic_reset_windows = periodic_reset_windows
        self.state_diagnostics.clear()
        self._previous_mamba_states.clear()

    def configure_content_memory_tensor_trace(self, *, enabled: bool) -> None:
        self.record_content_memory_tensors = bool(enabled)
        self.content_memory_tensor_trace.clear()

    def begin_training_snapshot_capture(self, depths: list[int]) -> None:
        """Capture selected generated-prefix states for multi-depth supervision."""
        if not depths or any(depth <= 0 for depth in depths):
            raise ValueError("training snapshot depths must be positive")
        if len(set(depths)) != len(depths):
            raise ValueError("training snapshot depths must be unique")
        self.training_state_snapshots.clear()
        self.training_history_chunks.clear()
        self._training_snapshot_depths = set(depths)
        self._capture_training_snapshots = True

    def end_training_snapshot_capture(
        self,
    ) -> tuple[dict[int, tuple[Any, Any]], list[tuple[torch.Tensor, torch.Tensor]]]:
        self._capture_training_snapshots = False
        snapshots = self.training_state_snapshots
        history_chunks = self.training_history_chunks
        self.training_state_snapshots = {}
        self.training_history_chunks = []
        self._training_snapshot_depths.clear()
        return snapshots, history_chunks

    def restore_training_snapshot(self, snapshot: tuple[Any, Any]) -> None:
        """Restore one accepted prefix without changing trace counters."""
        root_state, body_state = snapshot
        self.root_block._sequence_state = (
            None if root_state is None else root_state.clone()
        )
        self.body_block._sequence_state = (
            None if body_state is None else body_state.clone()
        )

    @staticmethod
    def _state_tensors(state: Any) -> list[torch.Tensor]:
        if state is None:
            return []
        if isinstance(state, torch.Tensor):
            return [state]
        if isinstance(state, dict):
            return [
                tensor
                for key in sorted(state)
                for tensor in ArdyDiffusionMemoryController._state_tensors(state[key])
            ]
        if isinstance(state, (list, tuple)):
            return [
                tensor
                for value in state
                for tensor in ArdyDiffusionMemoryController._state_tensors(value)
            ]
        tensors = []
        for name in ("angle_dt_state", "ssm_state", "k_state", "v_state"):
            value = getattr(state, name, None)
            if value is not None:
                tensors.extend(ArdyDiffusionMemoryController._state_tensors(value))
        return tensors

    @classmethod
    def _state_norm(cls, state: Any) -> float | None:
        tensors = cls._state_tensors(state)
        if not tensors:
            return None
        total = sum(tensor.detach().float().square().sum() for tensor in tensors)
        return float(total.sqrt())

    @classmethod
    def _state_drift_norm(cls, current: Any, previous: Any) -> float | None:
        current_tensors = cls._state_tensors(current)
        previous_tensors = cls._state_tensors(previous)
        if not current_tensors or len(current_tensors) != len(previous_tensors):
            return None
        total = sum(
            (now.detach().float() - before.detach().float()).square().sum()
            for now, before in zip(current_tensors, previous_tensors)
        )
        return float(total.sqrt())

    @staticmethod
    def _clone_mamba_state(state: Any) -> Any:
        if state is None:
            return None
        return state.clone()

    def _record_state_snapshot(self, *, event: str, reset_after_commit: bool) -> None:
        if not self.record_state_diagnostics:
            return
        record: dict[str, Any] = {
            "event": event,
            "completed_windows": self._completed_windows,
            "accepted_commits": self._accepted_commits,
            "periodic_reset_after_commit": reset_after_commit,
            "blocks": {},
        }
        all_finite = True
        for block_name, block in (("root", self.root_block), ("body", self.body_block)):
            layer_records = []
            current_states = []
            previous_states = self._previous_mamba_states.get(block_name, [])
            sequence_state = block._sequence_state
            if sequence_state is not None:
                for layer_index, layer_state in enumerate(sequence_state.layers):
                    mamba_state = layer_state.mamba_state
                    state_norm = self._state_norm(mamba_state)
                    drift_norm = self._state_drift_norm(
                        mamba_state,
                        previous_states[layer_index]
                        if layer_index < len(previous_states) else None,
                    )
                    mixer = block.sequence_stack.layers[layer_index].mixer
                    gate = getattr(mixer, "fusion_gate_logit", None)
                    fusion_gate = (
                        None
                        if gate is None
                        else float(torch.sigmoid(gate.detach().float()))
                        * float(getattr(mixer, "runtime_mamba_output_scale", 1.0))
                    )
                    cache = layer_state.attention_cache
                    attention_bytes = sum(
                        tensor.numel() * tensor.element_size()
                        for chunk in cache.chunks
                        for tensor in (chunk.key, chunk.value, chunk.token_valid)
                    )
                    mamba_bytes = sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in self._state_tensors(mamba_state)
                    )
                    content_memory = layer_state.persistent_memory
                    content_memory_bytes = (
                        0
                        if content_memory is None
                        else content_memory.numel() * content_memory.element_size()
                    )
                    content_memory_norm = (
                        None
                        if content_memory is None
                        else float(content_memory.detach().float().norm(dim=-1).mean())
                    )
                    content_module = getattr(mixer, "content_memory", None)
                    content_read_scale = (
                        None
                        if content_module is None
                        else float(content_module.read_residual_scale.detach().float())
                    )
                    finite = all(
                        bool(torch.isfinite(tensor).all())
                        for tensor in self._state_tensors(mamba_state)
                    )
                    content_diag = getattr(mixer, "last_content_memory_diagnostics", {})
                    all_finite = all_finite and finite
                    layer_records.append(
                        {
                            "layer": layer_index,
                            "mamba_state_norm": state_norm,
                            "mamba_state_drift_norm": drift_norm,
                            "attention_cache_chunks": cache.num_chunks,
                            "attention_history_tokens": cache.num_history_tokens,
                            "attention_state_bytes": attention_bytes,
                            "mamba_state_bytes": mamba_bytes,
                            "content_memory_bytes": content_memory_bytes,
                            "content_memory_norm": content_memory_norm,
                            "content_memory_read_residual_scale": content_read_scale,
                            "content_memory_update_norm": content_diag.get("memory_update_norm"),
                            "content_memory_write_gate_mean": content_diag.get("write_gate_mean"),
                            "content_memory_write_gate_min": content_diag.get("write_gate_min"),
                            "content_memory_write_gate_max": content_diag.get("write_gate_max"),
                            "content_memory_proposal_norm": content_diag.get("content_proposal_norm"),
                            "state_summary_norm": content_diag.get("state_summary_norm"),
                            "state_delta_norm": content_diag.get("state_delta_norm"),
                            "augmented_minus_base_attention_norm": content_diag.get("augmented_minus_base_attention_norm"),
                            "persistent_state_bytes": (
                                attention_bytes + mamba_bytes + content_memory_bytes
                            ),
                            "fusion_gate": fusion_gate,
                            "finite": finite,
                        }
                    )
                    current_states.append(self._clone_mamba_state(mamba_state))
                    if self.record_content_memory_tensors and content_memory is not None:
                        self.content_memory_tensor_trace.append({
                            "event": event,
                            "completed_windows": self._completed_windows,
                            "block": block_name,
                            "layer": layer_index,
                            "slots": content_memory.detach().float().cpu().clone(),
                        })
            record["blocks"][block_name] = layer_records
            self._previous_mamba_states[block_name] = current_states
        record["all_finite"] = all_finite
        self.state_diagnostics.append(record)

    def _finish_generated_commit(self) -> None:
        reset_after_commit = bool(
            self.periodic_reset_windows
            and self._completed_windows % self.periodic_reset_windows == 0
        )
        self._record_state_snapshot(
            event="generated_window", reset_after_commit=reset_after_commit
        )
        if reset_after_commit:
            # Preserve ARDY's outer translation/heading lifecycle and only
            # clear adapter-owned KV/SSM memory for the reset ablation.
            self.root_block.reset_memory()
            self.body_block.reset_memory()
            self._previous_mamba_states.clear()
        if (
            self._capture_training_snapshots
            and self._completed_windows in self._training_snapshot_depths
        ):
            self.training_state_snapshots[self._completed_windows] = (
                None
                if self.root_block._sequence_state is None
                else self.root_block._sequence_state.clone(),
                None
                if self.body_block._sequence_state is None
                else self.body_block._sequence_state.clone(),
            )

    def prime_accepted_history(
        self,
        history_hybrid: torch.Tensor,
        *,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        first_heading_angle: torch.Tensor,
    ) -> None:
        """Commit a ground-truth prefix before cache-only autoregressive rollout."""
        if not self.commit_enabled:
            raise RuntimeError("this sequence kernel has no persistent history state")
        self._set_content_memory_runtime()
        canonical_history_prime(
            self.raw_denoiser,
            history_hybrid,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            first_heading_angle=first_heading_angle,
            next_generation_frames=int(self.ardy_model.gen_horizon_len),
            chunk_id=self._next_chunk_id,
        )
        self._next_chunk_id += 1
        self._accepted_commits += 1
        self._record_state_snapshot(event="initial_history", reset_after_commit=False)

    def install(self) -> "ArdyDiffusionMemoryController":
        if self._installed:
            return self

        def controlled_denoising_step(*args, **kwargs):
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if self.commit_enabled:
                self._update_runtime_mamba_output_scale()
                self._apply_mamba_state_intervention()
                # ARDY's CFG wrapper expands B to 2B/3B internally. Keep the
                # persistent cache at the accepted user batch B so guidance
                # branches can read it independently and the final clean B
                # result can commit atomically.
                base_batch = bound.arguments["x"].shape[0]
                self.root_block._ensure_state(base_batch)
                self.body_block._ensure_state(base_batch)
            result = self._original_denoising_step(*args, **kwargs)
            timestep = bound.arguments["t"]
            self._last_sampling_timestep = int(timestep.max().item())
            self._last_text_feat = bound.arguments.get("text_feat")
            self._last_text_pad_mask = bound.arguments.get("text_pad_mask")
            if (
                self.commit_enabled
                and not self._supports_post_window_hook
                and bool((timestep == 0).all())
            ):
                # Lightweight test doubles have no ARDY outer lifecycle;
                # retain the old immediate path only for those tests.
                self._commit_accepted_result(result, bound.arguments)
            return result

        self.ardy_model.denoising_step = controlled_denoising_step
        self.ardy_model._sequence_kernel_post_window_hook = self._post_window_hook
        self._installed = True
        return self

    def uninstall(self) -> None:
        if self._installed:
            self.ardy_model.denoising_step = self._original_denoising_step
            self.ardy_model._sequence_kernel_post_window_hook = self._original_post_window_hook
            self._installed = False

    def _post_window_hook(
        self,
        recentered_sequence: torch.Tensor,
        center_pos: torch.Tensor,
        new_root_motion: torch.Tensor,
        next_heading_angle: Optional[torch.Tensor],
        auto_step: int,
    ) -> None:
        """Commit the recentered accepted tail as next-window history."""
        if not self.commit_enabled:
            return
        nfpt = int(self.raw_denoiser.num_frames_per_token)
        generation_frames = int(self.ardy_model.gen_horizon_len)
        generation_tokens = generation_frames // nfpt
        if recentered_sequence.shape[1] < generation_tokens:
            raise RuntimeError("recentered history is shorter than current generation")
        if next_heading_angle is None:
            raise RuntimeError("post-recenter hook must provide next-window heading")
        if self._last_text_feat is None or self._last_text_pad_mask is None:
            raise RuntimeError("missing post-recenter text conditioning")
        history_hybrid = recentered_sequence[:, -generation_tokens:]
        if self._capture_training_snapshots:
            self.training_history_chunks.append(
                (
                    history_hybrid.detach().clone(),
                    next_heading_angle.detach().clone(),
                )
                )
        # Content interventions have explicit all-write semantics. Diffusion
        # forwards remain read-only, while every accepted history write uses
        # the same registered intervention.
        self._set_content_memory_runtime()
        # External donor memory is a read-only view for noisy future-window
        # reads. The clean accepted commit must read the recipient's durable
        # slots so donor content cannot leak into its write proposal.
        if getattr(self, "_external_memory_bank", None) is not None:
            for block in (self.root_block, self.body_block):
                block.content_memory_read_view_runtime = None
        canonical_history_prime(
            self.raw_denoiser,
            history_hybrid,
            text_feat=self._last_text_feat,
            text_pad_mask=self._last_text_pad_mask,
            first_heading_angle=next_heading_angle,
            next_generation_frames=generation_frames,
            chunk_id=self._next_chunk_id,
        )
        self._next_chunk_id += 1
        self._completed_windows += 1
        self._accepted_commits += 1
        self._finish_generated_commit()

    def _commit_accepted_result(self, accepted_x: torch.Tensor, call: dict) -> None:
        generation_mask = call["generation_token_mask"]
        chunk_id = self._next_chunk_id
        root_state_before = (
            None
            if self.root_block._sequence_state is None
            else self.root_block._sequence_state.clone()
        )
        body_state_before = (
            None
            if self.body_block._sequence_state is None
            else self.body_block._sequence_state.clone()
        )
        root_commits_before = self.root_block.accepted_commits
        body_commits_before = self.body_block.accepted_commits
        self.root_block.prepare_accepted_commit(generation_mask, chunk_id=chunk_id)
        self.body_block.prepare_accepted_commit(generation_mask, chunk_id=chunk_id)
        try:
            # Reuse ARDY's exact projections, root/body coupling, constraints,
            # masks, and motion conversion. The only special behavior lives in
            # the two blocks, which commit only selected generation tokens.
            with torch.inference_mode():
                self.raw_denoiser(
                    accepted_x,
                    call["history_len"],
                    call["generation_len"],
                    call["future_len"],
                    call["history_mask"],
                    call["generation_mask"],
                    call["future_mask"],
                    call["history_token_mask"],
                    generation_mask,
                    call["future_token_mask"],
                    call["text_feat"],
                    call["text_pad_mask"],
                    torch.zeros_like(call["t"]),
                    call["first_heading_angle"],
                    call["motion_mask"],
                    call["observed_motion"],
                )
            if self.root_block.accepted_commits != root_commits_before + 1:
                raise RuntimeError(
                    "accepted ARDY window must commit root memory exactly once"
                )
            if self.body_block.accepted_commits != body_commits_before + 1:
                raise RuntimeError(
                    "accepted ARDY window must commit body memory exactly once"
                )
        except Exception:
            # Root and body are one accepted ARDY window. Roll both back if
            # either block/projection fails so no half-committed window leaks.
            self.root_block._sequence_state = root_state_before
            self.body_block._sequence_state = body_state_before
            self.root_block.accepted_commits = root_commits_before
            self.body_block.accepted_commits = body_commits_before
            raise
        finally:
            # A failed raw pass must never leave a future noisy call armed to
            # commit accidentally.
            self.root_block.cancel_accepted_commit()
            self.body_block.cancel_accepted_commit()

        self._next_chunk_id += 1
        self._completed_windows += 1
        self._accepted_commits += 1
        self._finish_generated_commit()


class ArdySequenceKernelClient:
    """Generation client that resets ephemeral kernel state per request."""

    def __init__(
        self,
        ardy_model: Any,
        controller: Optional[ArdyDiffusionMemoryController],
    ) -> None:
        self.model = ardy_model
        self.controller = controller

    def reset_memory(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        if self.controller is not None:
            self.controller.reset(reset_mask)

    def __call__(self, *args, **kwargs):
        self.reset_memory()
        return self.model(*args, **kwargs)

    def autoregressive_step(self, *args, **kwargs):
        self.reset_memory()
        return self.model.autoregressive_step(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.model, name)
