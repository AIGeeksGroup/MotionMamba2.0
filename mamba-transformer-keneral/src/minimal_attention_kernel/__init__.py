"""Mamba-free attention kernel with explicit chunk memory.

Importing this package intentionally imports neither ``state_memory_operator``
nor ``mamba_ssm``.
"""

from .attention import flash2_attention_available
from .config import MinimalAttentionKernelConfig
from .kv_cache import CachedChunk, ChunkSlidingWindowKVCache, PendingChunkKV
from .operator import MinimalAttentionKernel

__all__ = [
    "MinimalAttentionKernelConfig",
    "MinimalAttentionKernel",
    "ChunkSlidingWindowKVCache",
    "CachedChunk",
    "PendingChunkKV",
    "flash2_attention_available",
]
