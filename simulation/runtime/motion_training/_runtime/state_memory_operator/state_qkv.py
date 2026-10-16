"""State-based Query/Key encoder for the fixed-capacity Event Memory.

Ported from `mamba_algorithm/src/state_memory/state_qkv.py` (research
repo). Validated finding (2026-07-21/22, see
`docs/mamba_integration_contract.md`): state is a reliable INDEX (retrieval
Hit@1 near 100%) but not a reliable PAYLOAD -- a linear probe over the full
raw state tops out under 1% decodability for fine-grained content at
realistic vocabulary/class-count scale. Consequently this module only
produces Q/K (addressing); V (the payload written into memory) must come
from a content-based projection instead -- see `event_memory.py`'s
`ContentValueEncoder`, not from this module.

    q_c = normalize(W_q [s_c^+ ; delta_c])   -- current state and direction
    k_c = normalize(W_k [s_c^- ; delta_c])   -- historical state-transition identity
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateQKVEncoder(nn.Module):
    def __init__(self, d_summary: int, d_qkv: int) -> None:
        super().__init__()
        self.Wq = nn.Linear(2 * d_summary, d_qkv)
        self.Wk = nn.Linear(2 * d_summary, d_qkv)

    def encode_query(self, s_after: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.Wq(torch.cat([s_after, delta], dim=-1)), dim=-1)

    def encode_key(self, s_before: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.Wk(torch.cat([s_before, delta], dim=-1)), dim=-1)


class ContentValueEncoder(nn.Module):
    """Projects a chunk's own content representation (NOT its state) into
    the memory's value space. This is the fix for the state-decodability gap
    above: an embedding/projection of content is lossless-ish by
    construction, where a compressed recurrent-state summary is not."""

    def __init__(self, d_model: int, d_value: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_value, dtype=dtype)

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        return self.proj(content.to(self.proj.weight.dtype))


class ContentQueryEncoder(nn.Module):
    """Optional addition to the state-derived query: mixes in a projection
    of the CURRENT chunk's own content. Needed whenever the pre-chunk state
    alone cannot disambiguate "which query is this" -- e.g. the query must
    see its own key/identity to know what's being asked, and that
    information only becomes available once this chunk's input is seen,
    which is legal exactly when query construction happens before/alongside
    running this chunk through Mamba (see `operator.py`)."""

    def __init__(self, d_model: int, d_qkv: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_qkv, dtype=dtype)

    def forward(self, state_query: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        content_q = F.normalize(self.proj(content.to(self.proj.weight.dtype)).float(), dim=-1)
        return F.normalize(state_query + content_q, dim=-1)
