"""Bounded Stage-07 output interactions for H0 and H1."""

from __future__ import annotations

import torch
from torch import nn


class OutputInteraction(nn.Module):
    """Fuse token-aligned Attention and Mamba outputs with exact-zero init."""

    def __init__(self, variant: str, d_model: int, scale_init: float = 0.0) -> None:
        super().__init__()
        if variant not in {"h0_parallel", "h1_innovation"}:
            raise ValueError(f"invalid output interaction {variant!r}")
        self.variant = variant
        self.attention_norm = nn.LayerNorm(d_model)
        self.mamba_norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))
        if variant == "h0_parallel":
            self.attention_weight = nn.Parameter(torch.ones((), dtype=torch.float32))
            self.mamba_weight = nn.Parameter(torch.ones((), dtype=torch.float32))
            self.output_projection = nn.Linear(d_model, d_model)
            nn.init.eye_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
            self.gate = None
        else:
            self.attention_weight = None
            self.mamba_weight = None
            self.output_projection = nn.Linear(d_model, d_model)
            nn.init.eye_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
            self.gate = nn.Linear(3 * d_model, d_model)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(
        self, attention: torch.Tensor, mamba: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a = self.attention_norm(attention.float())
        m = self.mamba_norm(mamba.float())
        if self.variant == "h0_parallel":
            source = self.attention_weight * a + self.mamba_weight * m
            gate = None
            delta = a - m
        else:
            delta = a - m
            gate = torch.sigmoid(self.gate(torch.cat((a, m, delta), dim=-1)))
            source = delta
        correction = self.output_projection(source)
        if gate is not None:
            # H1 equation: g(a,m,delta) elementwise gates the projected
            # innovation, not the input to a dense projection.
            correction = gate * correction
        output = attention + self.scale * correction.to(attention.dtype)
        diagnostics = {
            "scale": self.scale.detach(),
            "attention_norm": a.detach().norm(dim=-1).mean(),
            "mamba_norm": m.detach().norm(dim=-1).mean(),
            "delta_norm": delta.detach().norm(dim=-1).mean(),
            "correction_norm": correction.detach().norm(dim=-1).mean(),
        }
        if gate is not None:
            diagnostics["gate_mean"] = gate.detach().mean()
        return output, diagnostics
