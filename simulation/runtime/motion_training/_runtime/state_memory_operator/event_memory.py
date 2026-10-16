"""Fixed-capacity, per-row-independent Event Memory with pluggable eviction.

Unifies the research repo's two separate classes (`StreamingEventMemory` /
FIFO and `ParticipativeEventMemory`) into one, since they differ only in the
score used to pick an eviction victim -- everything else (append-only
storage, per-row boolean validity mask, autograd safety) is identical:

    backend="fifo"          priority = write time (oldest evicted first)
    backend="participative" priority = EMA-decayed access credit
                             (least-recently-useful evicted first)

Validated (2026-07-25, `mamba_algorithm` research repo, repeated_reference
task): FIFO shows a hard accuracy cliff the instant a write falls outside
`capacity`, regardless of how often it's later queried (94% within capacity,
0% beyond, both at 800 steps). Participative Compression (Deep-Forcing
-inspired) keeps flat accuracy (~90%) across every distance tested, up to
3.6x raw capacity, and was also the more seed-stable of the two policies in
a second run.

Two things this version adds over the research-repo originals, both
required for real (motion) use and not exercised by the synthetic tasks
that validated the eviction behavior above:

1. **Per-row asynchronous reset** (`reset_rows`). The originals had a single
   shared write pointer / write-lockstep assumption -- fine when every batch
   row writes every step in lockstep (true for the synthetic tasks), false
   in general (real sequences in a batch reset at different times). Here,
   writing and eviction are already fully per-row (via the validity mask),
   so supporting async reset is just "invalidate this row's entries";
   nothing else needs to change.
2. **Compaction** (`compact`, called automatically). The append-only design
   trades unbounded list growth for autograd safety and per-row-independent
   eviction (see the research repo's own docstring for why in-place buffer
   writes break retrieval's autograd graph). Left unchecked over a long
   motion sequence, the list would grow with total elapsed time, not
   `capacity`. `compact()` physically drops history columns that are
   invalid for EVERY row, which is safe to do between chunks (all indices
   from a `retrieve()`/`write()` call are consumed immediately within that
   same chunk's processing, never cached across chunks) and keeps the list
   length close to `capacity` (plus write-mask asynchrony slack) instead of
   growing with total sequence length. See `tests/test_smoke.py` for a test
   that exercises this across many write/retrieve/compact cycles.
"""

from __future__ import annotations

from typing import Optional

import torch


class EventMemory:
    def __init__(
        self,
        batch_size: int,
        capacity: int,
        d_key: int,
        d_value: int,
        device,
        dtype: torch.dtype = torch.float32,
        backend: str = "fifo",
        participation_decay: float = 0.98,
        compact_every: int = 0,
    ) -> None:
        assert backend in ("fifo", "participative"), backend
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.backend = backend
        self.participation_decay = participation_decay
        # Compact once the list grows this far past capacity; 0 = auto (3x
        # capacity, at least 32) -- small enough to bound memory, large
        # enough that compaction (an O(history) op) doesn't run every step.
        self.compact_every = compact_every or max(32, capacity * 3)

        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._valid = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        self._write_time = torch.zeros(batch_size, 0, device=device)
        self._participation = torch.zeros(batch_size, 0, device=device)
        self._clock = 0

    @property
    def num_entries(self) -> int:
        return len(self._keys)

    def reset_rows(self, reset_mask: Optional[torch.Tensor]) -> None:
        """Invalidate every entry for rows where `reset_mask` is True (an
        episode boundary). Out-of-place; safe to call between chunks."""
        if reset_mask is None or not bool(reset_mask.any()):
            return
        self._valid = self._valid & (~reset_mask).unsqueeze(1)

    def _priority(self) -> torch.Tensor:
        return self._write_time if self.backend == "fifo" else self._participation

    def write(self, keys: torch.Tensor, values: torch.Tensor, write_mask: Optional[torch.Tensor] = None) -> None:
        """keys/values: (batch, d_key/d_value), one candidate entry per row.
        `write_mask`: (batch,) bool, which rows actually write this call
        (None = all rows) -- rows that don't write still get a column
        appended (kept invalid) so every row's list index stays aligned."""
        self._keys.append(keys)
        self._values.append(values)
        valid_col = (
            torch.ones(self.batch_size, 1, dtype=torch.bool, device=self.device)
            if write_mask is None
            else write_mask.view(-1, 1).to(self.device)
        )
        self._valid = torch.cat([self._valid, valid_col], dim=1)
        self._write_time = torch.cat(
            [self._write_time, torch.full((self.batch_size, 1), float(self._clock), device=self.device)], dim=1
        )
        self._participation = torch.cat([self._participation, torch.zeros(self.batch_size, 1, device=self.device)], dim=1)
        self._clock += 1

        valid_count = self._valid.sum(dim=1)
        over = valid_count > self.capacity
        if over.any():
            priority = self._priority().masked_fill(~self._valid, float("inf"))
            evict_idx = priority.argmin(dim=1)
            rows = over.nonzero(as_tuple=True)[0]
            valid = self._valid.clone()
            valid[rows, evict_idx[rows]] = False
            self._valid = valid

        if len(self._keys) >= self.compact_every:
            self.compact()

    def compact(self) -> None:
        """Physically drop history columns invalid for EVERY row. Safe
        between chunks (see module docstring); a no-op if nothing is dead."""
        if not self._valid.numel():
            return
        alive = self._valid.any(dim=0)
        if bool(alive.all()):
            return
        keep = alive.nonzero(as_tuple=True)[0].tolist()
        self._keys = [self._keys[i] for i in keep]
        self._values = [self._values[i] for i in keep]
        self._valid = self._valid[:, keep]
        self._write_time = self._write_time[:, keep]
        self._participation = self._participation[:, keep]

    def retrieve(self, query: torch.Tensor, topk: int = 4, temperature: float = 1.0):
        """query: (batch, d_key), L2-normalized. Returns (r, topk_idx,
        weights, sims). `topk_idx` indexes the CURRENT internal list order
        -- consume it immediately (same chunk), don't cache across chunks,
        since `compact()` can renumber it afterwards."""
        K = torch.stack(self._keys, dim=1)
        V = torch.stack(self._values, dim=1)
        sims = torch.einsum("bd,bnd->bn", query, K)
        sims = sims.masked_fill(~self._valid, float("-inf"))
        k = min(topk, K.shape[1])
        topk_val, topk_idx = sims.topk(k, dim=-1)
        weights = torch.softmax(topk_val / temperature, dim=-1)
        gathered = torch.gather(V, 1, topk_idx.unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        r = (weights.unsqueeze(-1) * gathered).sum(dim=1)
        if self.backend == "participative":
            with torch.no_grad():
                self._participation *= self.participation_decay
                self._participation.scatter_add_(1, topk_idx, weights.detach())
        return r, topk_idx, weights, sims

    def read_at(self, indices: torch.Tensor) -> torch.Tensor:
        """Oracle-style hard gather at a GIVEN per-row slot index (e.g. for
        controlled ablation), bypassing learned retrieval entirely."""
        V = torch.stack(self._values, dim=1)
        return torch.gather(V, 1, indices.view(-1, 1, 1).expand(-1, 1, V.shape[-1])).squeeze(1)

    def still_valid(self, rows: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self._valid[rows, indices]
