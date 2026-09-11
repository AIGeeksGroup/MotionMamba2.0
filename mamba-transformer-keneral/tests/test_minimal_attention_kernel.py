"""Focused validation for the Mamba-free minimal attention package."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_attention_kernel import (  # noqa: E402
    MinimalAttentionKernel,
    MinimalAttentionKernelConfig,
    flash2_attention_available,
)


def make_config(**overrides) -> MinimalAttentionKernelConfig:
    values = dict(
        d_model=16,
        num_heads=4,
        window_size_chunks=3,
        use_attention_sinks=True,
        num_sink_tokens=2,
        attention_backend="sdpa",
        attention_dropout=0.0,
    )
    values.update(overrides)
    return MinimalAttentionKernelConfig(**values)


def make_kernel(**overrides) -> MinimalAttentionKernel:
    torch.manual_seed(7)
    return MinimalAttentionKernel(make_config(**overrides)).eval()


def test_no_cache_forward_shape_and_finite():
    kernel = make_kernel()
    x = torch.randn(2, 5, 16)
    output, pending, aux = kernel.forward_chunk(x)
    assert output.shape == x.shape
    assert pending.key.shape == (2, 4, 5, 4)
    assert torch.isfinite(output).all()
    assert aux["query_tokens"] == 5
    assert aux["kv_tokens"] == 2 + 5


def test_backward_has_finite_gradients():
    kernel = make_kernel()
    x = torch.randn(2, 5, 16, requires_grad=True)
    output, _, _ = kernel.forward_chunk(x)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in kernel.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_commit_one_chunk_and_commit_false_is_read_only():
    kernel = make_kernel()
    cache = kernel.init_cache(batch_size=2)
    x = torch.randn(2, 5, 16)
    _, pending, aux = kernel.forward_chunk(x, cache=cache, commit=False)
    assert cache.num_chunks == 0
    assert aux["cache_committed"] is False
    kernel.commit(cache, pending, chunk_id=10)
    assert cache.num_chunks == 1
    assert cache.chunk_ids == (10,)
    assert cache.num_history_tokens == 5


def test_previous_chunk_is_actually_consumed():
    kernel = make_kernel(use_attention_sinks=False, num_sink_tokens=0)
    current = torch.randn(1, 4, 16)

    cache_a = kernel.init_cache(batch_size=1)
    _, pending_a, _ = kernel.forward_chunk(torch.full((1, 4, 16), -3.0), cache=cache_a)
    kernel.commit(cache_a, pending_a, chunk_id=0)

    cache_b = kernel.init_cache(batch_size=1)
    _, pending_b, _ = kernel.forward_chunk(torch.full((1, 4, 16), 3.0), cache=cache_b)
    kernel.commit(cache_b, pending_b, chunk_id=0)

    out_a, _, aux_a = kernel.forward_chunk(current, cache=cache_a)
    out_b, _, aux_b = kernel.forward_chunk(current, cache=cache_b)
    assert aux_a["history_tokens_read"] == aux_b["history_tokens_read"] == 4
    assert not torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_current_chunk_is_bidirectional_self_attention():
    kernel = make_kernel(use_attention_sinks=False, num_sink_tokens=0)
    current_a = torch.randn(1, 4, 16)
    current_b = current_a.clone()
    current_b[:, -1] += 20.0
    out_a, _, _ = kernel.forward_chunk(current_a)
    out_b, _, _ = kernel.forward_chunk(current_b)
    # Changing one current token changes another current token's output, so
    # current K/V are present and default within-chunk attention is bidirectional.
    assert not torch.allclose(out_a[:, 0], out_b[:, 0])


def test_do_read_false_ignores_cache_without_mutating_it():
    kernel = make_kernel(use_attention_sinks=False, num_sink_tokens=0)
    cache = kernel.init_cache(batch_size=1)
    _, pending, _ = kernel.forward_chunk(torch.randn(1, 3, 16), cache=cache)
    kernel.commit(cache, pending, chunk_id=0)
    current = torch.randn(1, 2, 16)
    read_output, _, _ = kernel.forward_chunk(current, cache=cache, do_read=True)
    no_read_output, _, aux = kernel.forward_chunk(current, cache=cache, do_read=False)
    fresh_output, _, _ = kernel.forward_chunk(
        current, cache=kernel.init_cache(batch_size=1)
    )
    torch.testing.assert_close(no_read_output, fresh_output)
    assert not torch.allclose(read_output, no_read_output)
    assert aux["history_tokens_read"] == 0
    assert cache.chunk_ids == (0,)


def test_many_read_only_passes_then_exactly_one_commit():
    kernel = make_kernel()
    cache = kernel.init_cache(batch_size=1)
    _, history_pending, _ = kernel.forward_chunk(torch.randn(1, 3, 16), cache=cache)
    kernel.commit(cache, history_pending, chunk_id=0)
    before_ids = cache.chunk_ids
    before_key = cache.chunks[0].key.clone()

    final_pending = None
    for _ in range(7):
        _, final_pending, aux = kernel.forward_chunk(
            torch.randn(1, 4, 16), cache=cache, commit=False
        )
        assert aux["cache_committed"] is False
        assert cache.chunk_ids == before_ids
        torch.testing.assert_close(cache.chunks[0].key, before_key)
    kernel.commit(cache, final_pending, chunk_id=1)
    assert cache.chunk_ids == (0, 1)


def test_chunk_level_sliding_eviction_and_sink_persistence():
    kernel = make_kernel(window_size_chunks=3, num_sink_tokens=2)
    cache = kernel.init_cache(batch_size=1)
    committed = {}
    for chunk_id in range(5):
        x = torch.full((1, chunk_id + 1, 16), float(chunk_id))
        _, pending, _ = kernel.forward_chunk(x, cache=cache)
        committed[chunk_id] = (pending.key.detach().clone(), pending.value.detach().clone())
        kernel.commit(cache, pending, chunk_id=chunk_id)

    assert cache.chunk_ids == (2, 3, 4)
    assert cache.num_chunks == 3
    assert [chunk.num_tokens for chunk in cache.chunks] == [3, 4, 5]
    assert cache.num_history_tokens == 12
    assert cache.num_sink_tokens == 2
    history_key, history_value, history_valid = cache.get()
    torch.testing.assert_close(history_key, torch.cat([committed[i][0] for i in (2, 3, 4)], dim=2))
    torch.testing.assert_close(history_value, torch.cat([committed[i][1] for i in (2, 3, 4)], dim=2))
    assert history_valid.all()
    _, _, aux = kernel.forward_chunk(torch.randn(1, 2, 16), cache=cache)
    assert aux["sink_tokens"] == 2
    assert aux["kv_tokens"] == 2 + 12 + 2


def test_variable_chunk_lengths_have_exact_boundaries_offsets_and_eviction():
    kernel = make_kernel(window_size_chunks=2, use_attention_sinks=False, num_sink_tokens=0)
    cache = kernel.init_cache(batch_size=1)
    committed = {}
    for chunk_id, length in enumerate((8, 13, 6)):
        _, pending, _ = kernel.forward_chunk(torch.randn(1, length, 16), cache=cache)
        committed[chunk_id] = (pending.key.detach().clone(), pending.value.detach().clone())
        kernel.commit(cache, pending, chunk_id=chunk_id)
    assert cache.chunk_ids == (1, 2)
    assert [chunk.num_tokens for chunk in cache.chunks] == [13, 6]
    key, value, valid = cache.get()
    assert key.shape == value.shape == (1, 4, 19, 4)
    torch.testing.assert_close(key[:, :, :13], committed[1][0])
    torch.testing.assert_close(key[:, :, 13:], committed[2][0])
    torch.testing.assert_close(value[:, :, :13], committed[1][1])
    torch.testing.assert_close(value[:, :, 13:], committed[2][1])
    assert valid.shape == (1, 19) and valid.all()


def test_dedicated_learnable_sinks_survive_eviction_and_reset():
    kernel = make_kernel(window_size_chunks=2, num_sink_tokens=2)
    cache = kernel.init_cache(batch_size=1)
    sink_parameter = kernel.sink_tokens.detach().clone()
    sink_key_before, sink_value_before = kernel._sink_kv(batch_size=1)
    for chunk_id in range(20):
        _, pending, _ = kernel.forward_chunk(torch.randn(1, 3, 16), cache=cache)
        kernel.commit(cache, pending, chunk_id=chunk_id)
    assert cache.num_chunks == 2
    kernel.reset_cache(cache)
    sink_key_after, sink_value_after = kernel._sink_kv(batch_size=1)
    assert cache.num_chunks == 0 and cache.num_sink_tokens == 2
    torch.testing.assert_close(kernel.sink_tokens, sink_parameter)
    torch.testing.assert_close(sink_key_after, sink_key_before)
    torch.testing.assert_close(sink_value_after, sink_value_before)


def test_window_zero_never_persists_normal_chunks():
    kernel = make_kernel(window_size_chunks=0)
    cache = kernel.init_cache(batch_size=1)
    _, pending, _ = kernel.forward_chunk(torch.randn(1, 3, 16), cache=cache)
    kernel.commit(cache, pending, chunk_id=0)
    assert cache.num_chunks == 0
    assert cache.num_history_tokens == 0


def test_full_reset_removes_old_episode_history():
    kernel = make_kernel(use_attention_sinks=False, num_sink_tokens=0)
    cache = kernel.init_cache(batch_size=1)
    history = torch.randn(1, 3, 16)
    current = torch.randn(1, 3, 16)
    _, pending, _ = kernel.forward_chunk(history, cache=cache)
    kernel.commit(cache, pending, chunk_id=0)
    with_history, _, _ = kernel.forward_chunk(current, cache=cache)
    cache.reset()
    after_reset, _, _ = kernel.forward_chunk(current, cache=cache)
    fresh, _, _ = kernel.forward_chunk(current, cache=kernel.init_cache(batch_size=1))
    assert cache.num_chunks == 0
    torch.testing.assert_close(after_reset, fresh)
    assert not torch.allclose(with_history, after_reset)


def test_per_row_reset_prevents_cross_episode_leakage():
    kernel = make_kernel(use_attention_sinks=False, num_sink_tokens=0)
    cache = kernel.init_cache(batch_size=2)
    _, pending, _ = kernel.forward_chunk(torch.randn(2, 3, 16), cache=cache)
    kernel.commit(cache, pending, chunk_id=0)
    cache_before_reset = cache.clone()
    current = torch.randn(2, 2, 16)
    before_reset_output, _, _ = kernel.forward_chunk(current, cache=cache_before_reset)
    kernel.reset_cache(cache, torch.tensor([True, False]))
    assert cache.num_chunks_per_row.tolist() == [0, 1]
    assert cache.num_history_tokens_per_row.tolist() == [0, 3]

    reset_output, _, aux = kernel.forward_chunk(current, cache=cache)
    fresh_row0, _, _ = kernel.forward_chunk(
        current[:1], cache=kernel.init_cache(batch_size=1)
    )
    torch.testing.assert_close(reset_output[:1], fresh_row0, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(reset_output[1:], before_reset_output[1:], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(reset_output[:1], before_reset_output[:1])
    assert aux["backend_used"] == "sdpa"


def test_cache_clone_and_detach_are_explicit_and_independent():
    kernel = make_kernel()
    cache = kernel.init_cache(batch_size=1)
    _, pending, _ = kernel.forward_chunk(torch.randn(1, 3, 16), cache=cache)
    kernel.commit(cache, pending, chunk_id=0, detach=False)
    clone = cache.clone().detach()
    clone.reset()
    assert clone.num_chunks == 0
    assert cache.num_chunks == 1
    assert cache.chunks[0].key.grad_fn is not None


def test_optimizer_step_changes_weights_and_stale_cache_is_rejected_until_reset():
    kernel = make_kernel()
    kernel.train()
    optimizer = torch.optim.SGD(kernel.parameters(), lr=0.05)
    cache = kernel.init_cache(batch_size=2)
    first = torch.randn(2, 4, 16)
    second = torch.randn(2, 4, 16)
    _, pending, _ = kernel.forward_chunk(first, cache=cache)
    kernel.commit(cache, pending, chunk_id=0, detach=False)
    output, _, _ = kernel.forward_chunk(second, cache=cache)
    loss = output.square().mean()
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in kernel.parameters()
    )
    weight_before = kernel.k_proj.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(weight_before, kernel.k_proj.weight)
    with pytest.raises(RuntimeError, match="stale"):
        kernel.forward_chunk(second, cache=cache)
    kernel.reset_cache(cache)
    assert cache.num_chunks == 0
    fresh_output, _, _ = kernel.forward_chunk(second, cache=cache)
    assert torch.isfinite(fresh_output).all()


def test_q_length_differs_from_kv_length():
    kernel = make_kernel(window_size_chunks=2, num_sink_tokens=3)
    cache = kernel.init_cache(batch_size=1)
    for chunk_id, length in enumerate((2, 5)):
        _, pending, _ = kernel.forward_chunk(torch.randn(1, length, 16), cache=cache)
        kernel.commit(cache, pending, chunk_id=chunk_id)
    output, _, aux = kernel.forward_chunk(torch.randn(1, 4, 16), cache=cache)
    assert output.shape == (1, 4, 16)
    assert aux["query_tokens"] == 4
    assert aux["kv_tokens"] == 3 + 2 + 5 + 4


def test_sdpa_matches_explicit_concatenation_reference():
    # Required deterministic oracle configuration: B=1, H=2, Dh=16,
    # current L=8, W=2, two dedicated learnable sinks.
    torch.manual_seed(23)
    kernel = MinimalAttentionKernel(
        MinimalAttentionKernelConfig(
            d_model=32,
            num_heads=2,
            head_dim=16,
            window_size_chunks=2,
            use_attention_sinks=True,
            num_sink_tokens=2,
            attention_backend="sdpa",
            bias=False,
        )
    ).eval()
    cache = kernel.init_cache(batch_size=1)
    for chunk_id in range(2):
        history = torch.randn(1, 8, 32)
        _, pending, _ = kernel.forward_chunk(history, cache=cache)
        kernel.commit(cache, pending, chunk_id=chunk_id, detach=False)
    current = torch.randn(1, 8, 32)

    actual, _, _ = kernel.forward_chunk(current, cache=cache)
    q = kernel._split_heads(kernel.q_proj(current))
    current_k = kernel._split_heads(kernel.k_proj(current))
    current_v = kernel._split_heads(kernel.v_proj(current))
    sink_k, sink_v = kernel._sink_kv(batch_size=1)
    history_k, history_v, _ = cache.get()
    full_k = torch.cat((sink_k, history_k, current_k), dim=2)
    full_v = torch.cat((sink_v, history_v, current_v), dim=2)
    weights = torch.softmax(q @ full_k.transpose(-2, -1) / math.sqrt(kernel.head_dim), dim=-1)
    reference = kernel.out_proj(kernel._merge_heads(weights @ full_v))
    assert actual.shape == (1, 8, 32)
    assert q.shape == (1, 2, 8, 16)
    assert full_k.shape == (1, 2, 2 + 2 * 8 + 8, 16)
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    abs_error = (actual - reference).abs()
    print(
        f"SDPA_REFERENCE max_abs_error={abs_error.max().item():.10f} "
        f"mean_abs_error={abs_error.mean().item():.10f}"
    )
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


def test_optional_causal_mode_keeps_history_and_masks_current_future():
    kernel = make_kernel(
        causal_within_chunk=True, use_attention_sinks=False, num_sink_tokens=0
    )
    cache = kernel.init_cache(batch_size=1)
    _, pending, _ = kernel.forward_chunk(torch.randn(1, 2, 16), cache=cache)
    kernel.commit(cache, pending, chunk_id=0)
    current = torch.randn(1, 4, 16)
    baseline, _, _ = kernel.forward_chunk(current, cache=cache)
    changed = current.clone()
    changed[:, -1] += 100.0
    changed_out, _, _ = kernel.forward_chunk(changed, cache=cache)
    # Future token changes cannot alter earlier current-token outputs.
    torch.testing.assert_close(baseline[:, :-1], changed_out[:, :-1], atol=1e-5, rtol=1e-5)


def test_flash2_backend_falls_back_explicitly_on_cpu():
    kernel = make_kernel(attention_backend="flash2", allow_backend_fallback=True)
    output, _, aux = kernel.forward_chunk(torch.randn(1, 3, 16))
    assert torch.isfinite(output).all()
    assert aux["backend_used"] == "sdpa"
    assert "requires CUDA" in aux["backend_fallback_reason"] or "unavailable" in aux[
        "backend_fallback_reason"
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention-2 parity requires CUDA")
def test_flash2_vs_sdpa_cuda_forward_backward_and_error_metrics():
    available, reason = flash2_attention_available()
    if not available:
        pytest.skip(f"FlashAttention unavailable: {reason}")
    config_flash = MinimalAttentionKernelConfig(
        d_model=128,
        num_heads=4,
        window_size_chunks=2,
        use_attention_sinks=True,
        num_sink_tokens=2,
        attention_backend="flash2",
        allow_backend_fallback=False,
    )
    config_sdpa = MinimalAttentionKernelConfig(
        d_model=128,
        num_heads=4,
        window_size_chunks=2,
        use_attention_sinks=True,
        num_sink_tokens=2,
        attention_backend="sdpa",
    )
    sdpa = MinimalAttentionKernel(config_sdpa).cuda().half().eval()
    flash = MinimalAttentionKernel(config_flash).cuda().half().eval()
    flash.load_state_dict(sdpa.state_dict())
    cache_sdpa = sdpa.init_cache(batch_size=1)
    cache_flash = flash.init_cache(batch_size=1)
    histories = [
        torch.randn(1, 16, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    current_sdpa = torch.randn(
        1, 16, 128, device="cuda", dtype=torch.float16, requires_grad=True
    )
    current_flash = current_sdpa.detach().clone().requires_grad_(True)
    for kernel, cache in ((sdpa, cache_sdpa), (flash, cache_flash)):
        for chunk_id, history in enumerate(histories):
            _, pending, _ = kernel.forward_chunk(history, cache=cache, commit=False)
            kernel.commit(cache, pending, chunk_id=chunk_id)
    assert cache_sdpa.chunk_ids == cache_flash.chunk_ids == (1, 2)
    out_sdpa, _, _ = sdpa.forward_chunk(current_sdpa, cache=cache_sdpa, commit=False)
    out_flash, pending, aux = flash.forward_chunk(
        current_flash, cache=cache_flash, commit=False
    )
    assert aux["backend_used"] == "flash2"
    assert aux["query_tokens"] == 16
    assert aux["kv_tokens"] == 2 + 2 * 16 + 16
    assert cache_flash.num_chunks == 2

    abs_error = (out_flash.float() - out_sdpa.float()).abs()
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    relative_error = (
        abs_error / out_sdpa.float().abs().clamp_min(1e-5)
    ).mean().item()
    print(
        f"FA2_PARITY max_abs_error={max_abs_error:.8f} "
        f"mean_abs_error={mean_abs_error:.8f} relative_error={relative_error:.8f}"
    )
    torch.testing.assert_close(out_flash, out_sdpa, atol=4e-3, rtol=4e-3)

    loss = out_flash.float().square().mean()
    loss.backward()
    assert current_flash.grad is not None and torch.isfinite(current_flash.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in flash.parameters()
    )
    flash.commit(cache_flash, pending, chunk_id=3)
    assert cache_flash.chunk_ids == (2, 3)


def test_minimal_import_and_execution_do_not_import_mamba_ssm():
    script = """
import sys, torch
from minimal_attention_kernel import MinimalAttentionKernel, MinimalAttentionKernelConfig
assert 'mamba_ssm' not in sys.modules
cfg = MinimalAttentionKernelConfig(d_model=8, num_heads=2, window_size_chunks=1,
    use_attention_sinks=False, num_sink_tokens=0, attention_backend='sdpa')
kernel = MinimalAttentionKernel(cfg)
x = torch.randn(1, 2, 8, requires_grad=True)
cache = kernel.init_cache(batch_size=1)
out, pending, _ = kernel.forward_chunk(x, cache=cache)
kernel.commit(cache, pending, chunk_id=0)
out.sum().backward()
assert 'mamba_ssm' not in sys.modules
assert cache.num_chunks == 1 and x.grad is not None
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"d_model": 15}, "divisible"),
        ({"window_size_chunks": -1}, "non-negative"),
        ({"use_attention_sinks": False}, "num_sink_tokens=0"),
        ({"attention_backend": "fa3"}, "'sdpa', 'flash2'"),
    ],
)
def test_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_config(**kwargs)


def test_flash4_is_reserved_but_not_falsely_implemented():
    kernel = make_kernel(attention_backend="flash4")
    with pytest.raises(NotImplementedError, match="reserved"):
        kernel.forward_chunk(torch.randn(1, 3, 16))
