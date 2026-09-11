"""Generic DiT stack tests independent of ARDY and real Mamba CUDA."""

from __future__ import annotations

import torch
from torch import nn

from dit_sequence_kernel import DiTSequenceStack
from minimal_attention_kernel import MinimalAttentionKernelConfig
from sequence_kernel import SequenceKernel, SequenceKernelConfig


def _minimal_mixer(_layer_index: int) -> SequenceKernel:
    attention = MinimalAttentionKernelConfig(
        d_model=32,
        num_heads=2,
        window_size_chunks=2,
        num_sink_tokens=2,
        attention_backend="sdpa",
    )
    return SequenceKernel(SequenceKernelConfig(mode="minimal", attention=attention))


def test_dit_stack_forward_backward_and_explicit_commit_lifecycle():
    torch.manual_seed(31)
    stack = DiTSequenceStack(
        d_model=32,
        ff_size=64,
        num_layers=3,
        mixer_factory=_minimal_mixer,
    )
    state = stack.init_state(batch_size=2)
    x = torch.randn(2, 5, 32, requires_grad=True)
    condition = torch.randn(2, 32)

    output, pending, _ = stack(x, condition, state=state, commit=False)
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert all(layer_state.attention_cache.num_chunks == 0 for layer_state in state.layers)

    stack.commit(state, pending, chunk_id=0)
    assert all(layer_state.attention_cache.chunk_ids == (0,) for layer_state in state.layers)

    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert stack.layers[0].mixer.attention.q_proj.weight.grad is not None


def test_dit_noisy_reads_are_repeatable_and_never_commit():
    stack = DiTSequenceStack(
        d_model=32,
        ff_size=64,
        num_layers=2,
        mixer_factory=_minimal_mixer,
    ).eval()
    state = stack.init_state(batch_size=1)
    condition = torch.randn(1, 32)
    accepted = torch.randn(1, 4, 32)
    _, accepted_pending, _ = stack(accepted, condition, state=state)
    stack.commit(state, accepted_pending, chunk_id=0)
    snapshot = state.clone()

    noisy = torch.randn(1, 4, 32)
    outputs = []
    for _ in range(6):
        output, _, _ = stack(noisy, condition, state=state, commit=False)
        outputs.append(output)
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
    for actual, expected in zip(state.layers, snapshot.layers):
        assert actual.attention_cache.chunk_ids == expected.attention_cache.chunk_ids
        assert torch.equal(
            actual.attention_cache.chunks[0].key,
            expected.attention_cache.chunks[0].key,
        )


def test_dit_per_row_reset_propagates_to_every_layer():
    stack = DiTSequenceStack(
        d_model=32,
        ff_size=64,
        num_layers=2,
        mixer_factory=_minimal_mixer,
    )
    state = stack.init_state(batch_size=2)
    x = torch.randn(2, 4, 32)
    condition = torch.randn(2, 32)
    _, pending, _ = stack(x, condition, state=state)
    stack.commit(state, pending, chunk_id=4)

    stack.reset_state(state, torch.tensor([True, False]))
    for layer_state in state.layers:
        chunk = layer_state.attention_cache.chunks[0]
        assert not bool(chunk.token_valid[0].any())
        assert bool(chunk.token_valid[1].any())


def test_dit_stack_knows_only_generic_sequence_kernel_type():
    stack = DiTSequenceStack(
        d_model=32,
        ff_size=64,
        num_layers=1,
        mixer_factory=_minimal_mixer,
    )
    source_names = {type(module).__module__ for module in stack.modules()}
    assert not any(name.startswith("state_memory_operator") for name in source_names)
    assert not any(name.startswith("mamba_ssm") for name in source_names)
    assert isinstance(stack.layers[0].mixer, SequenceKernel)


def test_dit_explicitly_aligns_autocast_normalization_dtype_at_mixer_boundary():
    class FloatNorm(nn.Module):
        def forward(self, x):
            return x.float()

    stack = DiTSequenceStack(
        d_model=32,
        ff_size=64,
        num_layers=1,
        mixer_factory=_minimal_mixer,
    ).to(dtype=torch.bfloat16)
    stack.layers[0].norm_mixer = FloatNorm()
    state = stack.init_state(batch_size=1, dtype=torch.bfloat16)
    x = torch.randn(1, 4, 32, dtype=torch.bfloat16)
    condition = torch.randn(1, 32, dtype=torch.bfloat16)
    output, _, _ = stack(x, condition, state=state)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
