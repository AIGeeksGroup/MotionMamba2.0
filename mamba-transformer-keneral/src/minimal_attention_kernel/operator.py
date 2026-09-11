"""Standalone attention + chunk sliding K/V + attention-sink operator."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .attention import run_attention
from .config import MinimalAttentionKernelConfig
from .kv_cache import ChunkSlidingWindowKVCache, PendingChunkKV


class MinimalAttentionKernel(nn.Module):
    """One reusable self-attention layer with explicit cross-chunk memory.

    This class deliberately does not include residual connections, norms, or
    an FFN: downstream Transformer/DiT stacks own those model-level choices and
    instantiate one explicit cache per layer.
    """

    def __init__(self, config: MinimalAttentionKernelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=config.bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=config.bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=config.bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=config.bias)

        if config.use_attention_sinks:
            # Standard learned static sinks: they are generic model parameters,
            # not tokens copied from an episode or designated from chunk 0.
            self.sink_tokens = nn.Parameter(torch.empty(config.num_sink_tokens, d_model))
            nn.init.normal_(self.sink_tokens, std=1.0 / math.sqrt(d_model))
        else:
            self.register_parameter("sink_tokens", None)

        self._reset_projection_parameters()

    def _reset_projection_parameters(self) -> None:
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    @property
    def head_dim(self) -> int:
        return self.config.resolved_head_dim

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.config.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, tokens, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, tokens, self.config.d_model)

    def _projection_version(self) -> tuple[int, ...]:
        """PyTorch mutation counters for parameters that produce persistent K/V."""
        parameters = (
            self.k_proj.weight,
            self.k_proj.bias,
            self.v_proj.weight,
            self.v_proj.bias,
        )
        return tuple(parameter._version for parameter in parameters if parameter is not None)

    def init_cache(
        self,
        *,
        batch_size: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> ChunkSlidingWindowKVCache:
        parameter = self.q_proj.weight
        resolved_device = parameter.device if device is None else torch.device(device)
        # ``torch.device("cuda")`` denotes the current accelerator but does
        # not compare equal to the concrete ``cuda:N`` carried by tensors.
        # Bind an unindexed device of the same type to this kernel's actual
        # parameter device so strict cache validation remains meaningful.
        if (
            resolved_device.index is None
            and resolved_device.type == parameter.device.type
        ):
            resolved_device = parameter.device
        return ChunkSlidingWindowKVCache(
            batch_size=batch_size,
            num_heads=self.config.num_heads,
            head_dim=self.head_dim,
            window_size_chunks=self.config.window_size_chunks,
            num_sink_tokens=self.config.num_sink_tokens if self.config.use_attention_sinks else 0,
            device=resolved_device,
            dtype=parameter.dtype if dtype is None else dtype,
            source_kernel_id=id(self),
            projection_version=self._projection_version(),
        )

    def _validate_cache(
        self,
        cache: ChunkSlidingWindowKVCache,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        expected = (
            batch_size,
            self.config.num_heads,
            self.head_dim,
            self.config.window_size_chunks,
            self.config.num_sink_tokens if self.config.use_attention_sinks else 0,
        )
        actual = (
            cache.batch_size,
            cache.num_heads,
            cache.head_dim,
            cache.window_size_chunks,
            cache.num_sink_tokens,
        )
        if actual != expected:
            raise ValueError(f"cache configuration mismatch: expected {expected}, got {actual}")
        if cache.source_kernel_id != id(self):
            raise ValueError("cache was created by a different kernel instance")
        if cache.projection_version != self._projection_version():
            raise RuntimeError(
                "cached projected K/V are stale because K/V projection weights changed; "
                "call kernel.reset_cache(cache) after optimizer.step()"
            )
        if cache.device != device or cache.dtype != dtype:
            raise ValueError(
                f"cache device/dtype ({cache.device}, {cache.dtype}) must match "
                f"input ({device}, {dtype})"
            )

    def _sink_kv(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sink_tokens is None:
            shape = (batch_size, self.config.num_heads, 0, self.head_dim)
            empty = torch.empty(shape, device=self.q_proj.weight.device, dtype=self.q_proj.weight.dtype)
            return empty, empty.clone()
        sinks = self.sink_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return self._split_heads(self.k_proj(sinks)), self._split_heads(self.v_proj(sinks))

    @staticmethod
    def _build_allowed_mask(
        *,
        key_valid: torch.Tensor,
        query_valid: torch.Tensor,
        prefix_tokens: int,
        causal_within_chunk: bool,
    ) -> Optional[torch.Tensor]:
        batch_size, q_len = query_valid.shape
        kv_len = key_valid.shape[1]
        all_valid = bool(key_valid.all()) and bool(query_valid.all())
        if all_valid:
            return None

        allowed = key_valid[:, None, :].expand(batch_size, q_len, kv_len)
        if causal_within_chunk:
            q_positions = torch.arange(q_len, device=key_valid.device)[:, None]
            key_positions = torch.arange(kv_len, device=key_valid.device)[None, :]
            causal = key_positions <= (prefix_tokens + q_positions)
            allowed = allowed & causal[None, :, :]
        # Invalid queries are zeroed again after attention, but including them
        # here makes the reference semantics explicit. Avoid all-False rows
        # producing backend-specific NaNs by temporarily letting them see key 0.
        allowed = allowed & query_valid[:, :, None]
        invalid_query = ~query_valid
        if bool(invalid_query.any()) and kv_len > 0:
            allowed = allowed.clone()
            row, query = invalid_query.nonzero(as_tuple=True)
            allowed[row, query, 0] = True
        return allowed

    def forward_chunk(
        self,
        x: torch.Tensor,
        *,
        cache: Optional[ChunkSlidingWindowKVCache] = None,
        do_read: bool = True,
        commit: bool = False,
        chunk_id: Optional[int] = None,
        current_mask: Optional[torch.Tensor] = None,
        detach_on_commit: bool = True,
        query_bias: Optional[torch.Tensor] = None,
        history_chunk_scales: Optional[torch.Tensor] = None,
        extra_content: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, PendingChunkKV, dict]:
        """Read sinks/history, attend with current K/V, and optionally commit.

        ``commit=False`` never mutates ``cache``. The returned pending K/V can
        be accepted later with :meth:`commit`.
        """
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError(
                f"x must have shape [B,L,{self.config.d_model}], got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got {x.dtype}")
        if x.device != self.q_proj.weight.device or x.dtype != self.q_proj.weight.dtype:
            raise ValueError(
                "x device/dtype must match kernel parameters; got "
                f"x=({x.device}, {x.dtype}) and q_proj="
                f"({self.q_proj.weight.device}, {self.q_proj.weight.dtype}); "
                "move/cast the kernel or input explicitly"
            )
        batch_size, chunk_len, _ = x.shape
        if chunk_len == 0:
            raise ValueError("current chunk must contain at least one token")
        if current_mask is None:
            current_mask = torch.ones(batch_size, chunk_len, dtype=torch.bool, device=x.device)
        elif current_mask.shape != (batch_size, chunk_len) or current_mask.dtype != torch.bool:
            raise ValueError(
                f"current_mask must be bool with shape {(batch_size, chunk_len)}, "
                f"got shape={tuple(current_mask.shape)} dtype={current_mask.dtype}"
            )
        else:
            current_mask = current_mask.to(x.device)

        if cache is not None:
            self._validate_cache(
                cache, batch_size=batch_size, device=x.device, dtype=x.dtype
            )
        elif commit:
            raise ValueError("commit=True requires an explicit cache")

        projected_query = self.q_proj(x)
        if query_bias is not None:
            expected_bias_shape = (batch_size, self.config.d_model)
            if query_bias.shape != expected_bias_shape:
                raise ValueError(
                    f"query_bias must have shape {expected_bias_shape}, "
                    f"got {tuple(query_bias.shape)}"
                )
            if query_bias.device != x.device or query_bias.dtype != x.dtype:
                raise ValueError("query_bias device/dtype must match x")
            projected_query = projected_query + query_bias[:, None, :]
        query = self._split_heads(projected_query)
        current_key = self._split_heads(self.k_proj(x))
        current_value = self._split_heads(self.v_proj(x))
        pending = PendingChunkKV(
            key=current_key,
            value=current_value,
            token_valid=current_mask,
            source_kernel_id=id(self),
            projection_version=self._projection_version(),
        )

        sink_key, sink_value = self._sink_kv(batch_size)
        sink_valid = torch.ones(
            batch_size, sink_key.shape[2], dtype=torch.bool, device=x.device
        )
        if do_read and cache is not None:
            history_key, history_value, history_valid = cache.get()
            if history_chunk_scales is not None:
                expected_scale_shape = (batch_size, cache.num_chunks)
                if history_chunk_scales.shape != expected_scale_shape:
                    raise ValueError(
                        "history_chunk_scales must have shape "
                        f"{expected_scale_shape}, got {tuple(history_chunk_scales.shape)}"
                    )
                if history_chunk_scales.device != x.device:
                    raise ValueError("history_chunk_scales must be on the input device")
                token_scales = torch.cat(
                    [
                        history_chunk_scales[:, index : index + 1].expand(
                            -1, chunk.num_tokens
                        )
                        for index, chunk in enumerate(cache.chunks)
                    ],
                    dim=1,
                ) if cache.num_chunks else history_chunk_scales.new_empty(batch_size, 0)
                history_key = history_key * token_scales[:, None, :, None].to(
                    history_key.dtype
                )
        else:
            if history_chunk_scales is not None:
                raise ValueError(
                    "history_chunk_scales requires do_read=True and an explicit cache"
                )
            shape = (batch_size, self.config.num_heads, 0, self.head_dim)
            history_key = torch.empty(shape, dtype=x.dtype, device=x.device)
            history_value = torch.empty_like(history_key)
            history_valid = torch.empty(batch_size, 0, dtype=torch.bool, device=x.device)

        if extra_content is not None:
            if extra_content.ndim != 3 or extra_content.shape[0] != batch_size or extra_content.shape[2] != self.config.d_model:
                raise ValueError("extra_content must have shape [B,K,d_model]")
            if extra_content.device != x.device or extra_content.dtype != x.dtype:
                raise ValueError("extra_content device/dtype must match x")
            memory_key = self._split_heads(self.k_proj(extra_content))
            memory_value = self._split_heads(self.v_proj(extra_content))
            memory_valid = torch.ones(batch_size, extra_content.shape[1], dtype=torch.bool, device=x.device)
        else:
            memory_key = torch.empty(batch_size, self.config.num_heads, 0, self.head_dim, dtype=x.dtype, device=x.device)
            memory_value = torch.empty_like(memory_key)
            memory_valid = torch.empty(batch_size, 0, dtype=torch.bool, device=x.device)

        key = torch.cat((sink_key, memory_key, history_key, current_key), dim=2)
        value = torch.cat((sink_value, memory_value, history_value, current_value), dim=2)
        key_valid = torch.cat((sink_valid, memory_valid, history_valid, current_mask), dim=1)
        prefix_tokens = sink_key.shape[2] + memory_key.shape[2] + history_key.shape[2]
        allowed_mask = self._build_allowed_mask(
            key_valid=key_valid,
            query_valid=current_mask,
            prefix_tokens=prefix_tokens,
            causal_within_chunk=self.config.causal_within_chunk,
        )

        dropout_p = self.config.attention_dropout if self.training else 0.0
        attention = run_attention(
            query,
            key,
            value,
            backend=self.config.attention_backend,
            allowed_mask=allowed_mask,
            causal_with_prefix=self.config.causal_within_chunk,
            dropout_p=dropout_p,
            allow_fallback=self.config.allow_backend_fallback,
        )
        output = self.out_proj(self._merge_heads(attention.output))
        output = output * current_mask.unsqueeze(-1).to(output.dtype)

        if commit:
            if chunk_id is None:
                raise ValueError("commit=True requires chunk_id")
            self.commit(cache, pending, chunk_id=chunk_id, detach=detach_on_commit)

        aux = {
            "backend_requested": self.config.attention_backend,
            "backend_used": attention.backend_used,
            "backend_fallback_reason": attention.fallback_reason,
            "query_tokens": chunk_len,
            "kv_tokens": key.shape[2],
            "history_tokens_read": history_key.shape[2],
            "persistent_content_tokens_read": memory_key.shape[2],
            "sink_tokens": sink_key.shape[2],
            "cache_committed": commit,
            "cached_chunks": cache.num_chunks if cache is not None else 0,
            "cached_history_tokens": cache.num_history_tokens if cache is not None else 0,
            "query_bias_norm": (
                None
                if query_bias is None
                else query_bias.detach().float().norm(dim=-1).mean().item()
            ),
            "history_chunk_scales": (
                None
                if history_chunk_scales is None
                else history_chunk_scales.detach()
            ),
        }
        return output, pending, aux

    def commit(
        self,
        cache: ChunkSlidingWindowKVCache,
        pending: PendingChunkKV,
        *,
        chunk_id: int,
        detach: bool = True,
    ) -> ChunkSlidingWindowKVCache:
        """Persist a previously computed current chunk for future calls."""
        if pending.source_kernel_id != id(self):
            raise ValueError("pending K/V was produced by a different kernel instance")
        self._validate_cache(
            cache,
            batch_size=pending.key.shape[0],
            device=pending.key.device,
            dtype=pending.key.dtype,
        )
        if pending.projection_version != self._projection_version():
            raise RuntimeError(
                "pending projected K/V are stale because projection weights changed; "
                "rerun forward_chunk after kernel.reset_cache(cache)"
            )
        cache.append(
            pending.key,
            pending.value,
            chunk_id=chunk_id,
            token_valid=pending.token_valid,
            detach=detach,
        )
        return cache

    def reset_cache(
        self,
        cache: ChunkSlidingWindowKVCache,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> ChunkSlidingWindowKVCache:
        """Reset episode history and safely refresh projection-weight binding.

        A full reset is required after ``optimizer.step()``. A partial reset
        cannot make still-retained rows compatible with new projection weights.
        """
        if cache.source_kernel_id != id(self):
            raise ValueError("cache was created by a different kernel instance")
        if reset_mask is not None and cache.projection_version != self._projection_version():
            raise RuntimeError(
                "projection weights changed; partial reset is unsafe because unreset rows "
                "still contain stale K/V. Perform a full kernel.reset_cache(cache)."
            )
        cache.reset(reset_mask)
        if reset_mask is None or cache.num_chunks == 0:
            cache.bind_projection_version(self._projection_version())
        return cache
