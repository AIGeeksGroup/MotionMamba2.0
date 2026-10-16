"""Explicit chunk-level sliding-window K/V cache.

The cache contains only normal episode-derived chunks. Attention sinks are
learned static parameters owned by the kernel, so they cannot leak episode
data and never need eviction or reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class PendingChunkKV:
    """Projected current-chunk K/V returned by a read-only forward pass."""

    key: torch.Tensor
    value: torch.Tensor
    token_valid: torch.Tensor
    source_kernel_id: int
    projection_version: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return self.key.shape[2]

    def detach(self) -> "PendingChunkKV":
        return PendingChunkKV(
            key=self.key.detach(),
            value=self.value.detach(),
            token_valid=self.token_valid.detach(),
            source_kernel_id=self.source_kernel_id,
            projection_version=self.projection_version,
        )


@dataclass(frozen=True)
class CachedChunk:
    """One whole committed chunk; boundaries are never flattened on write."""

    chunk_id: int
    key: torch.Tensor
    value: torch.Tensor
    token_valid: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return self.key.shape[2]

    def detach(self) -> "CachedChunk":
        return CachedChunk(
            chunk_id=self.chunk_id,
            key=self.key.detach(),
            value=self.value.detach(),
            token_valid=self.token_valid.detach(),
        )

    def clone(self) -> "CachedChunk":
        return CachedChunk(
            chunk_id=self.chunk_id,
            key=self.key.clone(),
            value=self.value.clone(),
            token_valid=self.token_valid.clone(),
        )


class ChunkSlidingWindowKVCache:
    """FIFO cache whose capacity is measured in chunks, not tokens.

    Tensor convention is ``[batch, heads, tokens, head_dim]``. Variable chunk
    lengths are supported because each :class:`CachedChunk` is stored
    separately and concatenated only for a read.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        window_size_chunks: int,
        num_sink_tokens: int,
        device: torch.device | str,
        dtype: torch.dtype,
        source_kernel_id: Optional[int] = None,
        projection_version: tuple[int, ...] = (),
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        if window_size_chunks < 0:
            raise ValueError("window_size_chunks must be non-negative")
        if num_sink_tokens < 0:
            raise ValueError("num_sink_tokens must be non-negative")
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size_chunks = window_size_chunks
        self.num_sink_tokens = num_sink_tokens
        self.device = torch.device(device)
        self.dtype = dtype
        self.source_kernel_id = source_kernel_id
        self.projection_version = projection_version
        self._chunks: list[CachedChunk] = []

    @property
    def chunks(self) -> tuple[CachedChunk, ...]:
        return tuple(self._chunks)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def num_history_tokens(self) -> int:
        return sum(chunk.num_tokens for chunk in self._chunks)

    @property
    def num_tokens(self) -> int:
        """Total readable K/V slots including static sink slots."""
        return self.num_sink_tokens + self.num_history_tokens

    @property
    def chunk_ids(self) -> tuple[int, ...]:
        return tuple(chunk.chunk_id for chunk in self._chunks)

    @property
    def oldest_chunk_id(self) -> Optional[int]:
        return self._chunks[0].chunk_id if self._chunks else None

    @property
    def newest_chunk_id(self) -> Optional[int]:
        return self._chunks[-1].chunk_id if self._chunks else None

    @property
    def num_chunks_per_row(self) -> torch.Tensor:
        if not self._chunks:
            return torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        return torch.stack(
            [chunk.token_valid.any(dim=1) for chunk in self._chunks], dim=1
        ).sum(dim=1)

    @property
    def num_history_tokens_per_row(self) -> torch.Tensor:
        if not self._chunks:
            return torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        return torch.cat([chunk.token_valid for chunk in self._chunks], dim=1).sum(dim=1)

    def _validate_kv(
        self, key: torch.Tensor, value: torch.Tensor, token_valid: torch.Tensor
    ) -> None:
        if key.ndim != 4:
            raise ValueError(f"key must have shape [B,H,T,D], got {tuple(key.shape)}")
        if key.shape != value.shape:
            raise ValueError(
                f"key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}"
            )
        expected_prefix = (self.batch_size, self.num_heads)
        if key.shape[:2] != expected_prefix or key.shape[3] != self.head_dim:
            raise ValueError(
                "K/V shape incompatible with cache: expected "
                f"[{self.batch_size},{self.num_heads},T,{self.head_dim}], got {tuple(key.shape)}"
            )
        if token_valid.shape != (self.batch_size, key.shape[2]):
            raise ValueError(
                f"token_valid must have shape {(self.batch_size, key.shape[2])}, "
                f"got {tuple(token_valid.shape)}"
            )
        if token_valid.dtype != torch.bool:
            raise TypeError(f"token_valid must be bool, got {token_valid.dtype}")
        if key.device != self.device or value.device != self.device or token_valid.device != self.device:
            raise ValueError("K/V and token_valid must be on the cache device")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError(
                f"K/V dtype must match cache dtype {self.dtype}, got {key.dtype}/{value.dtype}"
            )

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        chunk_id: int,
        token_valid: Optional[torch.Tensor] = None,
        detach: bool = True,
    ) -> "ChunkSlidingWindowKVCache":
        """Commit one whole normal chunk and apply chunk-level FIFO eviction."""
        if not isinstance(chunk_id, int):
            raise TypeError(f"chunk_id must be int, got {type(chunk_id).__name__}")
        if self._chunks and chunk_id <= self._chunks[-1].chunk_id:
            raise ValueError(
                f"chunk_id must increase monotonically; newest is {self._chunks[-1].chunk_id}, got {chunk_id}"
            )
        if token_valid is None:
            token_valid = torch.ones(
                self.batch_size, key.shape[2], dtype=torch.bool, device=self.device
            )
        self._validate_kv(key, value, token_valid)

        if self.window_size_chunks == 0:
            return self

        chunk = CachedChunk(
            chunk_id=chunk_id,
            key=key.detach() if detach else key,
            value=value.detach() if detach else value,
            token_valid=token_valid.detach(),
        )
        self._chunks.append(chunk)
        if len(self._chunks) > self.window_size_chunks:
            self._chunks = self._chunks[-self.window_size_chunks :]
        return self

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return concatenated normal history K/V and a valid-key mask."""
        if not self._chunks:
            shape = (self.batch_size, self.num_heads, 0, self.head_dim)
            empty = torch.empty(shape, device=self.device, dtype=self.dtype)
            valid = torch.empty(
                self.batch_size, 0, device=self.device, dtype=torch.bool
            )
            return empty, empty.clone(), valid
        return (
            torch.cat([chunk.key for chunk in self._chunks], dim=2),
            torch.cat([chunk.value for chunk in self._chunks], dim=2),
            torch.cat([chunk.token_valid for chunk in self._chunks], dim=1),
        )

    def reset(
        self, reset_mask: Optional[torch.Tensor] = None
    ) -> "ChunkSlidingWindowKVCache":
        """Clear all history, or only selected batch rows.

        Static learned sinks are not stored here and therefore remain available.
        """
        if reset_mask is None:
            self._chunks.clear()
            return self
        if reset_mask.shape != (self.batch_size,) or reset_mask.dtype != torch.bool:
            raise ValueError(
                f"reset_mask must be bool with shape {(self.batch_size,)}, "
                f"got shape={tuple(reset_mask.shape)} dtype={reset_mask.dtype}"
            )
        reset_mask = reset_mask.to(self.device)
        if not bool(reset_mask.any()):
            return self
        keep_rows = (~reset_mask)[:, None]
        next_chunks = []
        for chunk in self._chunks:
            token_valid = chunk.token_valid & keep_rows
            if bool(token_valid.any()):
                kv_keep = keep_rows[:, None, :, None].to(chunk.key.dtype)
                next_chunks.append(
                    CachedChunk(
                        chunk_id=chunk.chunk_id,
                        key=chunk.key * kv_keep,
                        value=chunk.value * kv_keep,
                        token_valid=token_valid,
                    )
                )
        self._chunks = next_chunks
        return self

    def detach(self) -> "ChunkSlidingWindowKVCache":
        self._chunks = [chunk.detach() for chunk in self._chunks]
        return self

    def bind_projection_version(self, projection_version: tuple[int, ...]) -> None:
        """Rebind an empty cache after projection weights have changed."""
        if self._chunks:
            raise RuntimeError("cannot rebind projection version while cached history exists")
        self.projection_version = projection_version

    def clone(self) -> "ChunkSlidingWindowKVCache":
        result = ChunkSlidingWindowKVCache(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            window_size_chunks=self.window_size_chunks,
            num_sink_tokens=self.num_sink_tokens,
            device=self.device,
            dtype=self.dtype,
            source_kernel_id=self.source_kernel_id,
            projection_version=self.projection_version,
        )
        result._chunks = [chunk.clone() for chunk in self._chunks]
        return result
