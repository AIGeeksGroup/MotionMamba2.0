"""Chunk wrapper around the official Mamba-3 SISO kernel.

Ported from `mamba_algorithm/src/state_memory/mamba3_wrapper.py` (research
repo) -- validated there across the full Stage-1 synthetic-task program.
General-purpose: operates on raw `(batch, chunk_len, d_model)` tensors with
no assumption about what the content represents (motion frames, tokens,
anything else).

`Mamba3.forward()` cannot resume a multi-token chunk from a non-zero
incoming state (it hardcodes `Input_States=None` on the prefill path), and
its `seqlen_offset > 0` decode dispatch to `.step()` is unverified/likely
broken for 3-D input. This module instead calls the raw
`mamba3_siso_combined` Triton kernel directly, replicating the exact
preprocessing `Mamba3.forward()` performs (in_proj split, A/DT computation,
RoPE angle computation, B/C RMSNorm), so a `Mamba3ChunkWrapper` built around
an existing `Mamba3` instance reproduces `Mamba3.forward()` bit-for-bit for
a single chunk spanning the whole sequence, while also supporting
`Input_States` for genuine multi-chunk continuation.

Requires the official Mamba-3 source (`mamba_ssm`,
https://github.com/state-spaces/mamba) importable on `sys.path`.
SISO only: MIMO chunk continuation with non-zero incoming state has no
upstream kernel support, so this wrapper rejects `is_mimo=True` models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba3 import Mamba3, heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


@dataclass
class Mamba3State:
    """Mirrors `Mamba3.allocate_inference_cache`'s 4-tuple, SISO shapes.

    angle_dt_state: (batch, nheads, num_rope_angles), fp32
    ssm_state:      (batch, nheads, headdim, d_state), fp32
    k_state:        (batch, 1, nheads, d_state), compute dtype
    v_state:        (batch, nheads, headdim), compute dtype

    Treat as opaque except for `.detach()`, `.clone()`, and
    `apply_reset_mask()`.
    """

    angle_dt_state: torch.Tensor
    ssm_state: torch.Tensor
    k_state: torch.Tensor
    v_state: torch.Tensor

    def detach(self) -> "Mamba3State":
        return Mamba3State(
            angle_dt_state=self.angle_dt_state.detach(),
            ssm_state=self.ssm_state.detach(),
            k_state=self.k_state.detach(),
            v_state=self.v_state.detach(),
        )

    def clone(self) -> "Mamba3State":
        return Mamba3State(
            angle_dt_state=self.angle_dt_state.clone(),
            ssm_state=self.ssm_state.clone(),
            k_state=self.k_state.clone(),
            v_state=self.v_state.clone(),
        )


def apply_reset_mask(state: Mamba3State, reset_mask: Optional[torch.Tensor]) -> Mamba3State:
    """Zero the state for every batch row where `reset_mask` is True,
    leaving other rows unchanged -- for episode boundaries in a batch where
    different rows reset at different times (the common case for motion
    training, where sequences in a batch don't share reset points).

    Out-of-place (`torch.where`, never an in-place index-assign) so this is
    safe to call on a state tensor that's still part of an active autograd
    graph from the previous chunk.
    """
    if reset_mask is None or not bool(reset_mask.any()):
        return state

    def _mask_like(t: torch.Tensor) -> torch.Tensor:
        shape = [reset_mask.shape[0]] + [1] * (t.dim() - 1)
        return reset_mask.view(*shape)

    zero = lambda t: torch.zeros_like(t)
    return Mamba3State(
        angle_dt_state=torch.where(_mask_like(state.angle_dt_state), zero(state.angle_dt_state), state.angle_dt_state),
        ssm_state=torch.where(_mask_like(state.ssm_state), zero(state.ssm_state), state.ssm_state),
        k_state=torch.where(_mask_like(state.k_state), zero(state.k_state), state.k_state),
        v_state=torch.where(_mask_like(state.v_state), zero(state.v_state), state.v_state),
    )


class Mamba3ChunkWrapper:
    """Exposes `forward_chunk(x, state) -> (y, next_state)` for a SISO Mamba3.

    Not an `nn.Module` itself -- it wraps an existing `Mamba3` module and
    reuses its parameters, so full-sequence execution via `mamba3.forward(x)`
    and single-chunk execution via `forward_chunk(x, None)` are guaranteed to
    use identical weights (this is what the Phase-0 chunk-consistency test
    relies on -- see `tests/test_smoke.py`).
    """

    def __init__(self, mamba3: Mamba3) -> None:
        if mamba3.is_mimo:
            raise ValueError(
                "Mamba3ChunkWrapper only supports SISO (is_mimo=False); MIMO "
                "chunk continuation from a non-zero state has no upstream "
                "kernel support."
            )
        self.mamba3 = mamba3

    def init_state(
        self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> Mamba3State:
        angle_dt_state, ssm_state, k_state, v_state = self.mamba3.allocate_inference_cache(
            batch_size, max_seqlen=1, device=device, dtype=dtype
        )
        return Mamba3State(angle_dt_state, ssm_state, k_state, v_state)

    def forward_chunk(
        self,
        x: torch.Tensor,
        state: Optional[Mamba3State] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Mamba3State]:
        """x: (batch, chunk_len, d_model) -> out: same shape, next_state.

        `reset_mask`: (batch,) bool, True = reset this row's incoming state
        to zero before running the chunk (episode boundary). Applied to
        `state` only; has no effect when `state is None` (already zero).
        """
        if state is not None and reset_mask is not None:
            state = apply_reset_mask(state, reset_mask)

        m = self.mamba3

        zxBCdtAtrap = m.in_proj(x)
        z, xh, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                m.d_inner,
                m.d_inner,
                m.d_state * m.num_bc_heads * m.mimo_rank,
                m.d_state * m.num_bc_heads * m.mimo_rank,
                m.nheads,
                m.nheads,
                m.nheads,
                m.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=m.headdim)
        xh = rearrange(xh, "b l (h p) -> b l h p", p=m.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        _A = -heavy_tail_activation(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-m.A_floor)
        DT = F.softplus(dd_dt + m.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        angles = angles.unsqueeze(-2).expand(-1, -1, m.nheads, -1).to(torch.float32)

        B = m.B_norm(B)
        C = m.C_norm(C)

        if state is not None:
            input_states = (
                state.angle_dt_state,
                state.ssm_state,
                state.k_state.squeeze(1),
                state.v_state,
            )
        else:
            input_states = None

        y, final_angle, final_ssm, final_k, final_v = mamba3_siso_combined(
            Q=C.squeeze(2),
            K=B.squeeze(2),
            V=xh,
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=m.C_bias.squeeze(1),
            K_bias=m.B_bias.squeeze(1),
            Angles=angles,
            D=m.D,
            Z=z if not m.is_outproj_norm else None,
            chunk_size=m.chunk_size,
            Input_States=input_states,
            return_final_states=True,
            cu_seqlens=None,
        )

        y = rearrange(y, "b l h p -> b l (h p)")
        if m.is_outproj_norm:
            z_flat = rearrange(z, "b l h p -> b l (h p)")
            y = m.norm(y, z_flat)
        out = m.out_proj(y.to(xh.dtype))

        next_state = Mamba3State(
            angle_dt_state=final_angle,
            ssm_state=final_ssm,
            k_state=final_k.unsqueeze(1),
            v_state=final_v,
        )
        return out, next_state
