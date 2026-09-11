from __future__ import annotations

import torch
from torch import nn

from ardy_adapter.stage09_structured import G0Correction, Stage09G0Block


class _Layer(nn.Module):
    def forward(self, x):
        return x + 0.1, None, {}


class _Stack(nn.Module):
    def __init__(self):
        super().__init__(); self.layers = nn.ModuleList([_Layer() for _ in range(4)])


class _Attention(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.sequence_stack = _Stack(); self.anchor = nn.Parameter(torch.ones(dim))
        self.pending = False; self.resets = 0; self.commits = 0

    def forward(self, x, valid, *_):
        hidden = x
        for layer in self.sequence_stack.layers:
            hidden = layer(hidden)[0]
        if self.pending:
            self.sequence_stack.layers[3](hidden[:, :2])
            self.pending = False; self.commits += 1
        return hidden * valid[..., None]

    def reset_memory(self, reset_mask=None): self.resets += 1
    def prepare_accepted_commit(self, selection, *, chunk_id, detach=True): self.pending = True
    def _stack_parameter(self): return self.anchor


class _Mamba(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.anchor = nn.Parameter(torch.ones(dim)); self.layers = ()
        self.captured_representations = {}; self.capture_layers = ()

    def enable_representation_capture(self, enabled=True, layers=None): self.capture_layers = layers

    def forward(self, x, *_):
        layer = self.capture_layers[0]
        self.captured_representations = {
            "token_hidden": {layer: x + 0.2},
            "ssm_state_summary": {layer: x.mean(1)[:, :, None]},
        }
        return x


def _inputs(batch=2, tokens=5, dim=8):
    x = torch.randn(batch, tokens, dim); valid = torch.ones(batch, tokens, dtype=torch.bool)
    return (x, valid, None, None, None, None, None)


def test_g0_zero_output_is_exact():
    correction = G0Correction(8, 8, 6); correction.zero_output()
    actual = correction(torch.randn(2, 5, 8), torch.randn(2, 5, 8))
    assert torch.equal(actual, torch.zeros_like(actual))


def test_stage09_wrapper_zero_init_delegation_and_short_commit_capture():
    attention = _Attention(8); mamba = _Mamba(8)
    block = Stage09G0Block(
        attention, mamba, layer=3, state_source="hidden", output_dim=8,
        token_mean=torch.zeros(8), token_std=torch.ones(8),
        state_mean=torch.zeros(8), state_std=torch.ones(8), zero_output=True,
    )
    inputs = _inputs()
    expected = attention(*inputs)
    block.prepare_accepted_commit(torch.ones(2, 5, dtype=torch.bool), chunk_id=0)
    actual = block(*inputs)
    assert torch.equal(actual, expected)
    assert attention.commits == 1
    assert block.attention_forwards == block.mamba_forwards == 1
    assert block.last_token.shape == block.last_state.shape == (2, 5, 8)
    block.reset_memory(); assert attention.resets == 1
    assert block._stack_parameter() is attention.anchor


def test_stage09_wrapper_backward_changes_only_correction():
    attention = _Attention(8); mamba = _Mamba(8)
    block = Stage09G0Block(
        attention, mamba, layer=3, state_source="hidden", output_dim=8,
        token_mean=torch.zeros(8), token_std=torch.ones(8),
        state_mean=torch.zeros(8), state_std=torch.ones(8), zero_output=True,
    )
    output = block(*_inputs()); output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in block.correction.parameters())
    assert all(parameter.grad is None for parameter in attention.parameters())
    assert all(parameter.grad is None for parameter in mamba.parameters())
