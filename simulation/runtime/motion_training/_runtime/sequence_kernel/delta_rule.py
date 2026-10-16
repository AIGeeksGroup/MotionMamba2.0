"""Isolated Stage-07 H2 adapter for accepted-state delta corrections."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn


class DeltaRuleStateCorrection(nn.Module):
    """Read and out-of-place correct only the matrix-like Mamba SSM state."""

    def __init__(self, d_model: int, nheads: int, headdim: int, d_state: int,
                 *, scale_init: float = 0.0) -> None:
        super().__init__()
        self.nheads, self.headdim, self.d_state = nheads, headdim, d_state
        self.token_norm = nn.LayerNorm(d_model)
        self.key = nn.Linear(d_model, nheads * d_state, bias=False)
        self.value = nn.Linear(d_model, nheads * headdim, bias=False)
        self.beta_logit = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))
        self.output_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def beta(self) -> torch.Tensor:
        return torch.tanh(self.beta_logit)

    def correct(self, state, token: torch.Tensor):
        ssm = getattr(state, "ssm_state", None)
        if not isinstance(ssm, torch.Tensor) or ssm.ndim != 4:
            raise TypeError("H2 requires a Mamba state with 4-D .ssm_state")
        if tuple(ssm.shape[1:]) != (self.nheads, self.headdim, self.d_state):
            raise ValueError("H2 state shape does not match configured Mamba geometry")
        z = self.token_norm(token.float())
        key = self.key(z).view(*z.shape[:2], self.nheads, self.d_state)
        key = torch.nn.functional.normalize(key, dim=-1)
        value = self.value(z).view(*z.shape[:2], self.nheads, self.headdim)
        predicted = torch.einsum("bhpn,blhn->blhp", ssm.float(), key)
        innovation = value - predicted
        correction = torch.einsum("blhp,blhn->bhpn", innovation, key) / token.shape[1]
        corrected = replace(state, ssm_state=ssm + self.beta * correction.to(ssm.dtype))
        return corrected, {
            "beta": self.beta.detach(),
            "state_norm": ssm.detach().float().norm(dim=(-2, -1)).mean(),
            "delta_norm": innovation.detach().float().norm(dim=(-2, -1)).mean(),
            "correction_norm": correction.detach().float().norm(dim=(-2, -1)).mean(),
        }
