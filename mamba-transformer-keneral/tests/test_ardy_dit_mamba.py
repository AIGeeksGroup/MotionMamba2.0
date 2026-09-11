"""CUDA acceptance for ARDY DiT adapter with the real existing Mamba-3 path."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from conftest import requires_cuda


class _PE(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x)


class _NegativePE(_PE):
    def forward(self, x, token_index):
        return self.dropout(x + token_index[:, :, None].to(x.dtype) / 100.0)


class _Time(nn.Module):
    def __init__(self, d_model, _pe):
        super().__init__()
        self.embedding = nn.Embedding(100, d_model)

    def forward(self, timesteps):
        return self.embedding(timesteps.long())[:, None]


def _pad(x, mask, size):
    return x[:, :size], mask[:, :size]


@requires_cuda
def test_real_mamba_ardy_dit_read_only_commit_backward_and_reset(monkeypatch):
    import ardy_adapter.ardy_dit_block as module
    from ardy_adapter import ArdyDiTSequenceBlock

    monkeypatch.setattr(
        module,
        "_import_ardy_backbone",
        lambda: (_PE, _NegativePE, _Time, _pad),
    )
    torch.manual_seed(53)
    block = ArdyDiTSequenceBlock(
        input_dim=-1,
        output_dim=20,
        skeleton=SimpleNamespace(nbjoints=27),
        llm_shape=[1, 64],
        use_text_mask=False,
        latent_dim=64,
        ff_size=128,
        num_layers=1,
        num_heads=4,
        add_input_proj=False,
        positional_encoding_mode="learned_prefix_zero_at_first_generation",
        input_first_heading_angle=True,
        sequence_mode="hybrid",
        attention_backend="sdpa",
        window_size_chunks=2,
        num_sink_tokens=2,
        mamba_headdim=32,
        mamba_d_state=32,
        mamba_expand=2,
        mamba_chunk_size=16,
        mamba_dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ).cuda().train()

    batch, tokens = 2, 16
    x = torch.randn(batch, tokens, 64, device="cuda", requires_grad=True)
    valid = torch.ones(batch, tokens, dtype=torch.bool, device="cuda")
    text = torch.randn(batch, 1, 64, device="cuda")
    text_valid = torch.ones(batch, 1, dtype=torch.bool, device="cuda")
    timestep = torch.full((batch,), 7, dtype=torch.long, device="cuda")
    heading = torch.tensor([0.1, 0.2], device="cuda")
    token_index = torch.arange(tokens, device="cuda")[None].expand(batch, -1) - 8

    for _ in range(3):
        output = block(
            x, valid, text, text_valid, timestep, heading, token_index
        )
        assert block._sequence_state.layers[0].attention_cache.num_chunks == 0
    assert output.shape == (batch, tokens, 20)
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    layer = block.sequence_stack.layers[0]
    assert layer.mixer.mamba_path.in_proj.weight.grad is not None
    assert layer.mixer.attention.q_proj.weight.grad is not None

    block.eval()
    selection = torch.zeros(batch, tokens, dtype=torch.bool, device="cuda")
    selection[:, -8:] = True
    block.prepare_accepted_commit(selection, chunk_id=0)
    with torch.no_grad():
        block(
            x.detach(),
            valid,
            text,
            text_valid,
            torch.zeros_like(timestep),
            heading,
            token_index,
        )
    state = block._sequence_state.layers[0]
    assert state.attention_cache.chunk_ids == (0,)
    assert state.attention_cache.chunks[0].num_tokens == 8
    assert state.mamba_state.ssm_state.shape == (batch, 4, 32, 32)

    block.reset_memory(torch.tensor([True, False], device="cuda"))
    assert not bool(state.attention_cache.chunks[0].token_valid[0].any())
    assert bool(state.attention_cache.chunks[0].token_valid[1].any())
    assert torch.count_nonzero(state.mamba_state.ssm_state[0]) == 0
