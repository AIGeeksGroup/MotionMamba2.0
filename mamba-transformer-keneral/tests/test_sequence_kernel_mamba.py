"""Real Mamba-3 CUDA acceptance test for the Stage-2 hybrid kernel."""

from __future__ import annotations

import torch

from conftest import requires_cuda


def _make_real_hybrid(
    interaction: str = "naive_gated_add", *, use_state_delta: bool = True
):
    from mamba_ssm.modules.mamba3 import Mamba3
    from minimal_attention_kernel import MinimalAttentionKernelConfig
    from sequence_kernel import SequenceKernel, SequenceKernelConfig

    mamba = Mamba3(
        d_model=64,
        d_state=32,
        expand=2,
        headdim=32,
        ngroups=1,
        is_mimo=False,
        chunk_size=16,
        layer_idx=0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    attention = MinimalAttentionKernelConfig(
        d_model=64,
        num_heads=4,
        window_size_chunks=2,
        use_attention_sinks=True,
        num_sink_tokens=2,
        attention_backend="sdpa",
    )
    return SequenceKernel(
        SequenceKernelConfig(
            mode="hybrid",
            attention=attention,
            interaction=interaction,
            state_summary_dim=16,
            use_state_delta=use_state_delta,
        ),
        mamba3=mamba,
    ).to(device="cuda", dtype=torch.bfloat16)


@requires_cuda
def test_real_mamba_hybrid_forward_backward_state_cache_reset_and_ablations():
    torch.manual_seed(23)
    kernel = _make_real_hybrid().train()
    state = kernel.init_state(
        batch_size=2, device=torch.device("cuda"), dtype=torch.bfloat16
    )

    history = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    _, history_pending, _ = kernel.forward_chunk(history, state=state)
    kernel.commit(state, history_pending, chunk_id=0)
    assert state.attention_cache.chunk_ids == (0,)
    assert state.mamba_state.ssm_state.shape == (2, 4, 32, 32)

    current = torch.randn(
        2, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    output, pending, aux = kernel.forward_chunk(current, state=state)
    assert output.shape == current.shape
    assert torch.isfinite(output.float()).all()
    assert aux["fusion"] == "gated_add"
    assert pending.attention_kv is not None and pending.mamba_state is not None

    output.float().square().mean().backward()
    assert current.grad is not None and torch.isfinite(current.grad.float()).all()
    assert kernel.attention.q_proj.weight.grad is not None
    assert kernel.mamba_path.in_proj.weight.grad is not None
    assert kernel.fusion_gate_logit.grad is not None

    kernel.eval()
    with torch.no_grad():
        fresh = kernel.init_state(
            batch_size=2, device=torch.device("cuda"), dtype=torch.bfloat16
        )
        attention_fresh, _, _ = kernel.forward_chunk(
            current.detach(),
            state=fresh,
            enable_attention_path=True,
            enable_mamba_path=False,
        )
        attention_history, _, _ = kernel.forward_chunk(
            current.detach(),
            state=state,
            enable_attention_path=True,
            enable_mamba_path=False,
        )
        assert not torch.allclose(attention_fresh, attention_history)

        mamba_fresh, _, _ = kernel.forward_chunk(
            current.detach(),
            state=fresh,
            enable_attention_path=False,
            enable_mamba_path=True,
        )
        mamba_history, _, _ = kernel.forward_chunk(
            current.detach(),
            state=state,
            enable_attention_path=False,
            enable_mamba_path=True,
        )
        assert not torch.allclose(mamba_fresh, mamba_history)

        row1_before = state.mamba_state.ssm_state[1].clone()
        kernel.reset_state(state, torch.tensor([True, False], device="cuda"))
        assert not bool(state.attention_cache.chunks[0].token_valid[0].any())
        assert bool(state.attention_cache.chunks[0].token_valid[1].any())
        assert torch.count_nonzero(state.mamba_state.ssm_state[0]) == 0
        assert torch.equal(state.mamba_state.ssm_state[1], row1_before)


@requires_cuda
def test_real_mamba_c1_state_query_differentiable_carry_and_no_current_leakage():
    torch.manual_seed(29)
    kernel = _make_real_hybrid("state_query").train()
    state = kernel.init_state(
        batch_size=1, device=torch.device("cuda"), dtype=torch.bfloat16
    )
    assert state.state_summary.shape == (1, 16)
    assert torch.count_nonzero(state.state_delta) == 0

    history = torch.randn(
        1, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    _, history_pending, _ = kernel.forward_chunk(history, state=state)
    kernel.commit(state, history_pending, chunk_id=0, detach=False)
    assert state.attention_cache.chunk_ids == (0,)
    assert torch.count_nonzero(state.state_delta) > 0

    kernel.token_state_interaction.scale.data.fill_(0.5)
    current = torch.randn(
        1, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    output, pending, aux = kernel.forward_chunk(current, state=state)
    assert output.shape == current.shape and torch.isfinite(output.float()).all()
    assert aux["fusion"] == "state_query"
    assert aux["interaction"]["query_bias_norm"] > 0
    assert pending.state_summary.shape == state.state_summary.shape
    output.float().square().mean().backward()

    assert kernel.mamba_path.in_proj.weight.grad is not None
    assert kernel.state_projector.linear.weight.grad is not None
    assert kernel.token_state_interaction.state_to_address.weight.grad is not None
    assert kernel.token_state_interaction.scale.grad is not None
    assert kernel.attention.q_proj.weight.grad is not None
    assert history.grad is not None and torch.isfinite(history.grad.float()).all()
    assert torch.count_nonzero(history.grad) > 0
    assert current.grad is not None and torch.isfinite(current.grad.float()).all()


@requires_cuda
def test_real_mamba_c1_kv_modulation_forward_backward():
    torch.manual_seed(31)
    kernel = _make_real_hybrid("state_kv_modulation").train()
    state = kernel.init_state(
        batch_size=1, device=torch.device("cuda"), dtype=torch.bfloat16
    )
    history = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16)
    _, pending, _ = kernel.forward_chunk(history, state=state)
    kernel.commit(state, pending, chunk_id=0, detach=False)
    kernel.token_state_interaction.scale.data.fill_(0.5)
    current = torch.randn(
        1, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    output, _, aux = kernel.forward_chunk(current, state=state)
    scales = aux["interaction"]["chunk_key_scales"]
    assert scales.shape == (1, 1)
    assert not torch.equal(scales, torch.ones_like(scales))
    output.float().square().mean().backward()
    assert kernel.mamba_path.in_proj.weight.grad is not None
    assert kernel.token_state_interaction.state_to_address.weight.grad is not None
