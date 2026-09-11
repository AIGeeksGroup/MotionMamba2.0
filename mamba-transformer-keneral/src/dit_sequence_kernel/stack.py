"""Reusable multi-layer DiT denoiser core over generic sequence mixers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from sequence_kernel import SequenceKernel

from .layer import DiTSequenceLayer
from .state import DiTSequenceState, PendingDiTSequenceState


class DiTSequenceStack(nn.Module):
    """A DiT-style stack that knows no attention or Mamba implementation detail."""

    def __init__(
        self,
        *,
        d_model: int,
        ff_size: int,
        num_layers: int,
        mixer_factory: Callable[[int], SequenceKernel],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [
                DiTSequenceLayer(
                    d_model=d_model,
                    ff_size=ff_size,
                    mixer=mixer_factory(index),
                    dropout=dropout,
                )
                for index in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def init_state(
        self,
        *,
        batch_size: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> DiTSequenceState:
        return DiTSequenceState(
            [
                layer.mixer.init_state(
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                )
                for layer in self.layers
            ]
        )

    def _validate_state(self, state: DiTSequenceState) -> None:
        if len(state.layers) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer states, got {len(state.layers)}"
            )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        *,
        state: DiTSequenceState,
        do_read: bool = True,
        commit: bool = False,
        chunk_id: int | None = None,
        current_mask: torch.Tensor | None = None,
        detach_on_commit: bool = True,
        content_memory_read_enabled: bool | None = None,
        content_memory_write_enabled: bool = True,
        content_memory_delta_mode: str = "normal",
        content_memory_condition: torch.Tensor | None = None,
        content_memory_read_view: torch.Tensor | list[torch.Tensor] | None = None,
        content_memory_read_view_level: str = "slots",
    ):
        self._validate_state(state)
        pending_layers = []
        layer_aux = []
        hidden = x
        for layer_index, (layer, layer_state) in enumerate(zip(self.layers, state.layers)):
            layer_read_view = (
                content_memory_read_view[layer_index]
                if isinstance(content_memory_read_view, list)
                else content_memory_read_view
            )
            hidden, pending, aux = layer(
                hidden,
                condition,
                state=layer_state,
                do_read=do_read,
                commit=commit,
                chunk_id=chunk_id,
                current_mask=current_mask,
                detach_on_commit=detach_on_commit,
                content_memory_read_enabled=content_memory_read_enabled,
                content_memory_write_enabled=content_memory_write_enabled,
                content_memory_delta_mode=content_memory_delta_mode,
                content_memory_condition=content_memory_condition,
                content_memory_read_view=layer_read_view,
                content_memory_read_view_level=content_memory_read_view_level,
            )
            pending_layers.append(pending)
            layer_aux.append(aux)
        output = self.final_norm(hidden)
        if current_mask is not None:
            output = output * current_mask[:, :, None].to(output.dtype)
        return output, PendingDiTSequenceState(tuple(pending_layers)), {"layers": layer_aux}

    def commit(
        self,
        state: DiTSequenceState,
        pending: PendingDiTSequenceState,
        *,
        chunk_id: int,
        detach: bool = True,
    ) -> DiTSequenceState:
        self._validate_state(state)
        if len(pending.layers) != len(self.layers):
            raise ValueError("pending layer count does not match this stack")
        for layer, layer_state, layer_pending in zip(
            self.layers, state.layers, pending.layers
        ):
            layer.mixer.commit(
                layer_state,
                layer_pending,
                chunk_id=chunk_id,
                detach=detach,
            )
        return state

    def reset_state(
        self,
        state: DiTSequenceState,
        reset_mask: torch.Tensor | None = None,
    ) -> DiTSequenceState:
        self._validate_state(state)
        for layer, layer_state in zip(self.layers, state.layers):
            layer.mixer.reset_state(layer_state, reset_mask)
        return state
