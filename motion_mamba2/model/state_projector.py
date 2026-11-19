'State projector.'
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
if TYPE_CHECKING:
    from motion_mamba2.model.mamba_state import Mamba3State

class StateProjector(nn.Module):

    def __init__(self, nheads: int, headdim: int, d_summary: int=128, pooling: str='mean') -> None:
        super().__init__()
        if pooling not in ('mean', 'rms'):
            raise ValueError(f"pooling must be 'mean' or 'rms', got {pooling!r}")
        self.nheads = nheads
        self.headdim = headdim
        self.pooling = pooling
        self.linear = nn.Linear(nheads * headdim, d_summary)
        self.norm = nn.LayerNorm(d_summary)

    def _pool(self, ssm_state: torch.Tensor) -> torch.Tensor:
        ssm_state = ssm_state.float()
        if self.pooling == 'mean':
            return ssm_state.mean(dim=-1)
        return torch.sqrt(torch.mean(ssm_state ** 2, dim=-1) + 1e-06)

    def forward(self, ssm_state: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(ssm_state)
        flat = pooled.reshape(pooled.shape[0], -1).to(self.linear.weight.dtype)
        return self.norm(self.linear(flat))

@dataclass
class StateTransition:
    s_before: torch.Tensor
    s_after: torch.Tensor
    delta: torch.Tensor

def project_transition(projector: StateProjector, state_before: Mamba3State, state_after: Mamba3State) -> StateTransition:
    s_before = projector(state_before.ssm_state)
    s_after = projector(state_after.ssm_state)
    return StateTransition(s_before=s_before, s_after=s_after, delta=s_after - s_before)
