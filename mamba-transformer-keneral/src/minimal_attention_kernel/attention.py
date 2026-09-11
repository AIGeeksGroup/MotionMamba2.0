"""Attention backend abstraction shared by the minimal kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttentionResult:
    output: torch.Tensor
    backend_used: str
    fallback_reason: Optional[str] = None


def flash2_attention_available() -> tuple[bool, str]:
    """Probe FlashAttention-2 lazily so importing the package stays light."""
    try:
        from flash_attn import flash_attn_func  # noqa: F401
    except (ImportError, OSError) as error:
        return False, f"{type(error).__name__}: {error}"
    return True, "FlashAttention-2 flash_attn_func available"


def _sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    allowed_mask: Optional[torch.Tensor],
    causal_with_prefix: bool,
    dropout_p: float,
) -> torch.Tensor:
    if allowed_mask is None and causal_with_prefix:
        q_len = query.shape[2]
        kv_len = key.shape[2]
        prefix_tokens = kv_len - q_len
        q_positions = torch.arange(q_len, device=query.device)[:, None]
        key_positions = torch.arange(kv_len, device=query.device)[None, :]
        allowed_mask = (key_positions <= prefix_tokens + q_positions)[None, :, :].expand(
            query.shape[0], -1, -1
        )
    mask = allowed_mask[:, None, :, :] if allowed_mask is not None else None
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=False,
    )


def run_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    backend: str,
    allowed_mask: Optional[torch.Tensor],
    causal_with_prefix: bool,
    dropout_p: float,
    allow_fallback: bool,
) -> AttentionResult:
    """Run attention over `[B,H,Q,D]` and `[B,H,KV,D]` tensors.

    ``causal_with_prefix`` is only passed directly to FlashAttention when no
    arbitrary validity mask is needed. FlashAttention uses bottom-right causal
    alignment for Q/KV length mismatch, which is exactly "all prefix history +
    current positions up to this query". SDPA receives an explicit combined
    mask assembled by the operator.
    """
    if backend == "sdpa":
        return AttentionResult(
            _sdpa(
                query,
                key,
                value,
                allowed_mask=allowed_mask,
                causal_with_prefix=causal_with_prefix,
                dropout_p=dropout_p,
            ),
            backend_used="sdpa",
        )
    if backend == "flash4":
        raise NotImplementedError(
            "attention_backend='flash4' is reserved for future hardware/kernel work; "
            "use 'sdpa' or 'flash2'"
        )
    if backend != "flash2":
        raise ValueError(f"Unsupported attention backend {backend!r}")

    # The dense FlashAttention API has no arbitrary padding-mask argument.
    if allowed_mask is not None:
        reason = "arbitrary key/query validity mask requires SDPA"
        if not allow_fallback:
            raise RuntimeError(reason)
        return AttentionResult(
            _sdpa(
                query,
                key,
                value,
                allowed_mask=allowed_mask,
                causal_with_prefix=causal_with_prefix,
                dropout_p=dropout_p,
            ),
            backend_used="sdpa",
            fallback_reason=reason,
        )

    available, reason = flash2_attention_available()
    if not available:
        if not allow_fallback:
            raise RuntimeError(f"FlashAttention-2 unavailable: {reason}")
        return AttentionResult(
            _sdpa(
                query,
                key,
                value,
                allowed_mask=None,
                causal_with_prefix=causal_with_prefix,
                dropout_p=dropout_p,
            ),
            backend_used="sdpa",
            fallback_reason=f"FlashAttention-2 unavailable: {reason}",
        )

    if query.device.type != "cuda" or query.dtype not in {torch.float16, torch.bfloat16}:
        reason = (
            "FlashAttention-2 requires CUDA fp16/bf16 tensors; "
            f"got device={query.device.type}, dtype={query.dtype}"
        )
        if not allow_fallback:
            raise RuntimeError(reason)
        return AttentionResult(
            _sdpa(
                query,
                key,
                value,
                allowed_mask=None,
                causal_with_prefix=causal_with_prefix,
                dropout_p=dropout_p,
            ),
            backend_used="sdpa",
            fallback_reason=reason,
        )

    try:
        from flash_attn import flash_attn_func

        output = flash_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=dropout_p,
            softmax_scale=None,
            causal=causal_with_prefix,
        ).transpose(1, 2)
    except (RuntimeError, TypeError, ValueError) as error:
        reason = f"FlashAttention-2 execution failed: {type(error).__name__}: {error}"
        if not allow_fallback:
            raise RuntimeError(reason) from error
        return AttentionResult(
            _sdpa(
                query,
                key,
                value,
                allowed_mask=None,
                causal_with_prefix=causal_with_prefix,
                dropout_p=dropout_p,
            ),
            backend_used="sdpa",
            fallback_reason=reason,
        )
    return AttentionResult(output, backend_used="flash2")
