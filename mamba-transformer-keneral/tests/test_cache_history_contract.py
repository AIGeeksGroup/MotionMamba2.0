"""Scientific cache-contract tests beyond lifecycle/rollback coverage."""
from types import SimpleNamespace

import torch

from ardy_adapter.history_commit import (
    build_canonical_history_layout,
    canonical_history_prime,
)
from tests.test_ardy_dit_sequence import _make_block, _patch_ardy


class _MaskConverter:
    num_frames_per_token = 4

    @staticmethod
    def convert_frame_mask_to_token_mask(history, generation, future, motion_mask):
        return tuple(value.reshape(value.shape[0], -1, 4).any(-1) for value in (history, generation, future))


class _FakeRaw:
    num_frames_per_token = 4
    motion_rep = SimpleNamespace(motion_rep_dim=6)
    hybrid = _MaskConverter()


def test_canonical_layout_uses_history_tokens_and_negative_origin():
    raw = _FakeRaw()
    history = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    layout = build_canonical_history_layout(
        raw, history, history_frames=12, next_generation_frames=8
    )
    assert layout.x.shape == (2, 5, 8)
    torch.testing.assert_close(layout.x[:, :3], history)
    assert layout.history_token_mask[:, :3].all()
    assert not layout.history_token_mask[:, 3:].any()
    assert layout.generation_token_mask[:, 3:].all()
    origin = torch.arange(5) - 3
    assert origin[:3].tolist() == [-3, -2, -1]
    assert origin[3:].tolist() == [0, 1]


def test_canonical_prime_commits_history_selection_and_negative_positions(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()

    class Raw(_FakeRaw, torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.root_model, self.body_model = root, body
            self.seen_indices = []

        def forward(self, x, history_len, generation_len, future_len,
                    history_mask, generation_mask, future_mask,
                    history_token_mask, generation_token_mask, future_token_mask,
                    text_feat, text_pad_mask, timesteps, first_heading_angle,
                    motion_mask, observed_motion):
            token_index = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
            token_index = token_index - history_len[:, None] // 4
            self.seen_indices.append(token_index.detach().clone())
            valid = history_token_mask | generation_token_mask | future_token_mask
            self.root_model(x, valid, text_feat, text_pad_mask, timesteps, None, token_index)
            self.body_model(x, valid, text_feat, text_pad_mask, timesteps, None, token_index)
            return x

    raw = Raw()
    history = torch.randn(2, 3, 32)
    text = torch.randn(2, 2, 24)
    mask = torch.ones(2, 2, dtype=torch.bool)
    canonical_history_prime(
        raw, history, text_feat=text, text_pad_mask=mask,
        first_heading_angle=torch.zeros(2), next_generation_frames=8, chunk_id=7,
    )
    assert raw.root_model.accepted_commits == raw.body_model.accepted_commits == 1
    assert raw.root_model._sequence_state.layers[0].attention_cache.chunk_ids == (7,)
    assert raw.root_model._sequence_state.layers[0].attention_cache.num_history_tokens == 3
    assert raw.seen_indices[-1][0].tolist() == [-3, -2, -1, 0, 1]


def test_canonical_prime_can_retain_history_write_gradients(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()

    class Raw(_FakeRaw, torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.root_model, self.body_model = root, body

        def forward(self, x, history_len, generation_len, future_len,
                    history_mask, generation_mask, future_mask,
                    history_token_mask, generation_token_mask, future_token_mask,
                    text_feat, text_pad_mask, timesteps, first_heading_angle,
                    motion_mask, observed_motion):
            valid = history_token_mask | generation_token_mask | future_token_mask
            token_index = torch.arange(x.shape[1])[None].expand(x.shape[0], -1)
            token_index = token_index - history_len[:, None] // 4
            self.root_model(x, valid, text_feat, text_pad_mask, timesteps, None, token_index)
            self.body_model(x, valid, text_feat, text_pad_mask, timesteps, None, token_index)
            return x

    raw = Raw()
    history = torch.randn(2, 3, 32, requires_grad=True)
    canonical_history_prime(
        raw,
        history,
        text_feat=torch.randn(2, 2, 24),
        text_pad_mask=torch.ones(2, 2, dtype=torch.bool),
        first_heading_angle=torch.zeros(2),
        next_generation_frames=8,
        chunk_id=0,
        detach=False,
    )
    cached = raw.root_model._sequence_state.layers[0].attention_cache.chunks[0]
    (cached.key.sum() + cached.value.sum()).backward()
    assert history.grad is not None
    assert torch.isfinite(history.grad).all()
    assert torch.count_nonzero(history.grad) > 0
