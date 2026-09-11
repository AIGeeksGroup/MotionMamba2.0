"""Explicit per-layer state for the reusable DiT sequence stack."""

from __future__ import annotations

from dataclasses import dataclass

from sequence_kernel import PendingSequenceKernelState, SequenceKernelState


@dataclass
class DiTSequenceState:
    layers: list[SequenceKernelState]

    def clone(self) -> "DiTSequenceState":
        return DiTSequenceState([state.clone() for state in self.layers])

    def detach(self) -> "DiTSequenceState":
        for state in self.layers:
            state.detach()
        return self


@dataclass(frozen=True)
class PendingDiTSequenceState:
    layers: tuple[PendingSequenceKernelState, ...]
