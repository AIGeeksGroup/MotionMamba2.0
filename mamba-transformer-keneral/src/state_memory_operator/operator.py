"""StateTransitionMemoryOperator: the composed, ready-to-use module.

Wires together (in the validated order -- see `docs/mamba_integration_contract.md`):

    raw Mamba recurrent state
      -> StateProjector            (state_projector.py)
      -> s_before, s_after, delta
      -> state-based Q / K         (state_qkv.py: StateQKVEncoder)
      -> content-based V           (state_qkv.py: ContentValueEncoder)
      -> fixed-capacity Event Memory (event_memory.py, FIFO or Participative)
      -> memory-conditioned next chunk

Two usage levels, matching the contract's Level 1 / Level 2 split:

    forward_chunk(...)              -- pure Mamba, no memory (Level 1)
    forward_chunk_with_memory(...)  -- full operator (Level 2)

See `examples/quickstart.py` for a runnable end-to-end example and
`tests/test_smoke.py` for the validation this module is expected to pass
before anything downstream depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import StateMemoryOperatorConfig
from .event_memory import EventMemory
from .mamba3_chunk_wrapper import Mamba3ChunkWrapper, Mamba3State, apply_reset_mask
from .state_projector import StateProjector
from .state_qkv import ContentQueryEncoder, ContentValueEncoder, StateQKVEncoder


@dataclass
class OperatorState:
    """Everything `forward_chunk_with_memory` needs to carry across chunks,
    bundled so callers only manage one opaque object. `s_prev`/`delta_prev`
    are the (state BEFORE this chunk, transition INTO this chunk) pair the
    NEXT call's query is built from -- retrieval for chunk c always uses the
    state as of the END of chunk c-1, since chunk c hasn't run yet when its
    query needs to be built (this ordering is required, not incidental: it's
    what "retrieval before running the query chunk" means in the validated
    design)."""

    mamba_state: Optional[Mamba3State]
    s_prev: torch.Tensor
    delta_prev: torch.Tensor

    def detach(self) -> "OperatorState":
        return OperatorState(
            mamba_state=self.mamba_state.detach() if self.mamba_state is not None else None,
            s_prev=self.s_prev.detach(),
            delta_prev=self.delta_prev.detach(),
        )


class StateTransitionMemoryOperator(nn.Module):
    def __init__(self, mamba3, config: StateMemoryOperatorConfig) -> None:
        super().__init__()
        self.config = config
        self.wrapper = Mamba3ChunkWrapper(mamba3)
        self.projector = StateProjector(
            nheads=mamba3.nheads, headdim=mamba3.headdim, d_summary=config.d_summary, pooling=config.pooling
        )
        self.qkv = StateQKVEncoder(d_summary=config.d_summary, d_qkv=config.d_qkv)
        # Content-facing modules (read/write the chunk's own representation,
        # e.g. bf16 Mamba activations) must match that compute dtype -- state
        # -facing modules (StateQKVEncoder, StateProjector) stay float32
        # since StateProjector always upcasts ssm_state internally.
        compute_dtype = mamba3.in_proj.weight.dtype
        self.value_encoder = ContentValueEncoder(config.d_model, config.d_value, dtype=compute_dtype)
        self.memory_proj = nn.Linear(config.d_value, config.d_model, dtype=compute_dtype)
        if config.use_content_query:
            self.query_content_encoder = ContentQueryEncoder(config.d_model, config.d_qkv, dtype=compute_dtype)

    # -- state / memory construction -------------------------------------

    def init_operator_state(
        self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> OperatorState:
        mamba_state = self.wrapper.init_state(batch_size, device=device, dtype=dtype)
        zeros = torch.zeros(batch_size, self.config.d_summary, device=device)
        return OperatorState(mamba_state=mamba_state, s_prev=zeros, delta_prev=zeros.clone())

    def init_memory(self, batch_size: int, device: Optional[torch.device] = None) -> EventMemory:
        return EventMemory(
            batch_size,
            self.config.memory_capacity,
            self.config.d_qkv,
            self.config.d_value,
            device=device,
            backend=self.config.memory_backend,
            participation_decay=self.config.participation_decay,
        )

    # -- Level 1: pure Mamba, no memory ------------------------------------

    def forward_chunk(self, x: torch.Tensor, state: Optional[Mamba3State] = None, reset_mask=None):
        out, next_state = self.wrapper.forward_chunk(x, state, reset_mask=reset_mask)
        aux = {
            "state_shape": tuple(next_state.ssm_state.shape),
            "chunk_len": x.shape[1],
            "state_was_reset": bool(reset_mask.any()) if reset_mask is not None else False,
            "peak_gpu_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        }
        return out, next_state, aux

    # -- Level 2: full state-transition memory operator --------------------

    def forward_chunk_with_memory(
        self,
        x: torch.Tensor,
        op_state: OperatorState,
        memory: Optional[EventMemory] = None,
        *,
        reset_mask: Optional[torch.Tensor] = None,
        do_read: bool = True,
        update_memory: bool = True,
        read_content: Optional[torch.Tensor] = None,
        write_content: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
        keep_mask: Optional[torch.Tensor] = None,
    ):
        """
        x: (batch, chunk_len, d_model) -- one chunk.
        op_state: from `init_operator_state` (first call) or the previous
            call's returned `next_op_state`.
        memory: from `init_memory`, or None to run pure-Mamba this call
            (equivalent to `forward_chunk` but keeping op_state bookkeeping
            consistent -- useful for warmup chunks before memory is engaged).
        read_content / write_content: (batch, d_model) content
            representations for this chunk -- e.g. a pooled/first-frame
            embedding of `x`. Required whenever `use_content_query`
            (read_content) or `update_memory` (write_content) apply; kept
            explicit rather than auto-derived from `x` since what "this
            chunk's content" means is a downstream (motion-specific)
            decision, not something this module should guess.
        write_mask: (batch,) bool, which rows actually commit a write this
            call (None = all). For ARDY-style usage: commit only after a
            clean/generated window is accepted, not on noisy intermediate
            denoising steps.
        keep_mask: (batch,) bool, memory-dropout mask -- rows where False
            get the injection zeroed this call. VALIDATED FINDING
            (2026-07-25): this repo's default recommendation is to NOT use
            dropout (leave `keep_mask=None`) unless you have already
            confirmed your task's query density can tolerate it -- a
            0.7-drop-rate default in the research repo caused a stuck-at-0%
            training failure on a sparse-query synthetic task that was
            initially (wrongly) blamed on the architecture.

        Returns (out, next_op_state, memory, aux).
        """
        if reset_mask is not None:
            mamba_state = (
                apply_reset_mask(op_state.mamba_state, reset_mask) if op_state.mamba_state is not None else None
            )
            keep = (~reset_mask).view(-1, 1).to(op_state.s_prev.dtype)
            op_state = OperatorState(
                mamba_state=mamba_state, s_prev=op_state.s_prev * keep, delta_prev=op_state.delta_prev * keep
            )
            if memory is not None:
                memory.reset_rows(reset_mask)

        retrieval_aux = {}
        r = None
        if do_read and memory is not None and memory.num_entries > 0:
            q = self.qkv.encode_query(op_state.s_prev, op_state.delta_prev)
            if self.config.use_content_query:
                if read_content is None:
                    raise ValueError("use_content_query=True requires read_content this call")
                q = self.query_content_encoder(q, read_content)
            r, topk_idx, weights, sims = memory.retrieve(
                q, topk=self.config.topk, temperature=self.config.retrieval_temperature
            )
            retrieval_aux = {
                "retrieval_topk_idx": topk_idx,
                "retrieval_weights": weights,
                "retrieval_sims": sims,
                "retrieval_entropy": -(weights.clamp_min(1e-9).log() * weights).sum(-1),
            }

        injection_delta_norm = None
        if r is not None:
            m = self.memory_proj(r.to(x.dtype))
            if keep_mask is not None:
                m = m * keep_mask.to(m.dtype).view(-1, 1)
            injection_delta_norm = m.float().norm(dim=-1)
            if self.config.injection_mode == "input":
                hidden, next_mamba_state = self.wrapper.forward_chunk(x + m.unsqueeze(1), op_state.mamba_state)
                out = hidden
            else:
                hidden, next_mamba_state = self.wrapper.forward_chunk(x, op_state.mamba_state)
                out = hidden + m.unsqueeze(1)
        else:
            hidden, next_mamba_state = self.wrapper.forward_chunk(x, op_state.mamba_state)
            out = hidden

        s_after = self.projector(next_mamba_state.ssm_state)
        delta = s_after - op_state.s_prev

        if update_memory and memory is not None:
            if write_content is None:
                raise ValueError("update_memory=True requires write_content this call")
            k = self.qkv.encode_key(op_state.s_prev, delta)
            v = self.value_encoder(write_content)
            memory.write(k, v, write_mask=write_mask)

        next_op_state = OperatorState(mamba_state=next_mamba_state, s_prev=s_after, delta_prev=delta)

        aux = {
            "state_before_summary": op_state.s_prev,
            "state_after_summary": s_after,
            "state_delta": delta,
            "injection_delta_norm": injection_delta_norm,
            "memory_usage_norm": (
                torch.stack(memory._values, dim=1).float().norm(dim=-1).mean().item()
                if memory is not None and memory.num_entries > 0
                else None
            ),
            "peak_gpu_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            **retrieval_aux,
        }
        return out, next_op_state, memory, aux
