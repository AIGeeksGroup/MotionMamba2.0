"""DiT-style residual layer whose mixer is a generic SequenceKernel."""

from __future__ import annotations

import torch
from torch import nn

from sequence_kernel import SequenceKernel, SequenceKernelState


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None]) + shift[:, None]


class DiTSequenceLayer(nn.Module):
    """AdaLN residual mixer + MLP layer with an injected sequence kernel."""

    def __init__(
        self,
        *,
        d_model: int,
        ff_size: int,
        mixer: SequenceKernel,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mixer.config.d_model != d_model:
            raise ValueError(
                f"mixer d_model {mixer.config.d_model} does not match layer {d_model}"
            )
        self.d_model = d_model
        self.mixer = mixer
        self.norm_mixer = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False)
        self.condition_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ff_size, d_model),
            nn.Dropout(dropout),
        )

        # DiT-style identity-friendly initialization. Gate biases of one keep
        # both paths trainable immediately while shifts/scales start neutral.
        modulation = self.condition_modulation[-1]
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)
        with torch.no_grad():
            modulation.bias[2 * d_model : 3 * d_model].fill_(1.0)
            modulation.bias[5 * d_model : 6 * d_model].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        *,
        state: SequenceKernelState,
        do_read: bool = True,
        commit: bool = False,
        chunk_id: int | None = None,
        current_mask: torch.Tensor | None = None,
        detach_on_commit: bool = True,
        content_memory_read_enabled: bool | None = None,
        content_memory_write_enabled: bool = True,
        content_memory_delta_mode: str = "normal",
        content_memory_condition: torch.Tensor | None = None,
        content_memory_read_view: torch.Tensor | None = None,
        content_memory_read_view_level: str = "slots",
    ):
        if condition.shape != (x.shape[0], self.d_model):
            raise ValueError(
                f"condition must have shape {(x.shape[0], self.d_model)}, "
                f"got {tuple(condition.shape)}"
            )
        modulation = self.condition_modulation(condition).chunk(6, dim=-1)
        shift_mixer, scale_mixer, gate_mixer, shift_mlp, scale_mlp, gate_mlp = modulation

        mixer_input = _modulate(self.norm_mixer(x), shift_mixer, scale_mixer)
        # CUDA autocast keeps some normalization outputs in fp32 even when a
        # Hybrid/C1 mixer is intentionally bf16. Sequence kernels enforce an
        # exact dtype boundary because persistent K/V and recurrent state use
        # the mixer parameter dtype.
        mixer_parameter = self.mixer.attention.q_proj.weight
        mixer_input = mixer_input.to(
            device=mixer_parameter.device, dtype=mixer_parameter.dtype
        )
        mixed, pending, aux = self.mixer.forward_chunk(
            mixer_input,
            state=state,
            do_read=do_read,
            commit=commit,
            chunk_id=chunk_id,
            current_mask=current_mask,
            detach_on_commit=detach_on_commit,
            content_memory_read_enabled=content_memory_read_enabled,
            content_memory_write_enabled=content_memory_write_enabled,
            content_memory_delta_mode=content_memory_delta_mode,
            content_memory_condition=content_memory_condition,
            content_memory_read_view=content_memory_read_view,
            content_memory_read_view_level=content_memory_read_view_level,
        )
        x = x + gate_mixer[:, None] * mixed
        mlp_input = _modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None] * self.mlp(mlp_input)
        if current_mask is not None:
            x = x * current_mask[:, :, None].to(x.dtype)
        return x, pending, aux
