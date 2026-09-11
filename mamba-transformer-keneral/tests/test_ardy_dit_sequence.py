"""ARDY adapter tests with a tiny stand-in for the external ARDY package."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import ardy_adapter.ardy_dit_block as ardy_dit_module
from ardy_adapter import (
    ArdyDiffusionMemoryController,
    ArdySequenceKernelConfig,
    install_ardy_sequence_kernel,
)
from ardy_adapter.ardy_dit_block import ArdyDiTSequenceBlock


class FakePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x)


class FakeNegativeIndexEncoding(FakePositionalEncoding):
    def forward(self, x, token_index):
        return self.dropout(x + token_index[:, :, None].to(x.dtype) / 100.0)


class FakeTimestepEmbedder(nn.Module):
    def __init__(self, d_model, _positional_encoder):
        super().__init__()
        self.embedding = nn.Embedding(100, d_model)

    def forward(self, timesteps):
        return self.embedding(timesteps.long())[:, None]


def _pad(x, mask, size):
    if x.shape[1] >= size:
        return x[:, :size], mask[:, :size]
    count = size - x.shape[1]
    return (
        torch.nn.functional.pad(x, (0, 0, 0, count)),
        torch.nn.functional.pad(mask, (0, count)),
    )


def _patch_ardy(monkeypatch):
    monkeypatch.setattr(
        ardy_dit_module,
        "_import_ardy_backbone",
        lambda: (
            FakePositionalEncoding,
            FakeNegativeIndexEncoding,
            FakeTimestepEmbedder,
            _pad,
        ),
    )


def _make_block(monkeypatch) -> ArdyDiTSequenceBlock:
    _patch_ardy(monkeypatch)
    return ArdyDiTSequenceBlock(
        input_dim=-1,
        output_dim=32,
        skeleton=SimpleNamespace(nbjoints=27),
        llm_shape=[2, 24],
        use_text_mask=True,
        latent_dim=32,
        ff_size=64,
        num_layers=2,
        num_heads=2,
        add_input_proj=False,
        positional_encoding_mode="learned_prefix_zero_at_first_generation",
        sequence_mode="minimal",
        attention_backend="sdpa",
        window_size_chunks=2,
        num_sink_tokens=2,
    )


def _inputs(batch=2, tokens=5):
    return {
        "x": torch.randn(batch, tokens, 32),
        "x_pad_mask": torch.ones(batch, tokens, dtype=torch.bool),
        "text_feat": torch.randn(batch, 2, 24),
        "text_feat_pad_mask": torch.ones(batch, 2, dtype=torch.bool),
        "timesteps": torch.full((batch,), 7, dtype=torch.long),
        "first_heading_angle": None,
        "token_index": torch.arange(tokens)[None].expand(batch, -1) - 2,
    }


def test_ardy_dit_block_noisy_forwards_are_read_only_and_commit_selected_clean_tokens(monkeypatch):
    torch.manual_seed(41)
    block = _make_block(monkeypatch).eval()
    inputs = _inputs()
    for _ in range(6):
        output = block(**inputs)
        assert output.shape == inputs["x"].shape
        assert all(
            state.attention_cache.num_chunks == 0
            for state in block._sequence_state.layers
        )

    selection = torch.tensor(
        [[False, False, True, True, False], [False, False, True, True, False]]
    )
    block.prepare_accepted_commit(selection, chunk_id=3)
    block(**{**inputs, "timesteps": torch.zeros(2, dtype=torch.long)})

    assert block.accepted_commits == 1
    for state in block._sequence_state.layers:
        assert state.attention_cache.chunk_ids == (3,)
        assert state.attention_cache.chunks[0].num_tokens == 2


class FakeRawDenoiser(nn.Module):
    def __init__(self, root, body):
        super().__init__()
        self.root_model = root
        self.body_model = body

    def forward(
        self,
        x,
        history_len,
        generation_len,
        future_len,
        history_mask,
        generation_mask,
        future_mask,
        history_token_mask,
        generation_token_mask,
        future_token_mask,
        text_feat,
        text_feat_pad_mask,
        timesteps,
        first_heading_angle=None,
        motion_mask=None,
        observed_motion=None,
    ):
        token_index = torch.arange(x.shape[1])[None].expand(x.shape[0], -1)
        valid = history_token_mask | generation_token_mask | future_token_mask
        root = self.root_model(
            x,
            valid,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
            token_index,
        )
        body = self.body_model(
            x,
            valid,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
            token_index,
        )
        return 0.5 * (root + body)


class FakeArdy:
    def __init__(self, raw):
        self.denoiser = SimpleNamespace(model=raw)

    def denoising_step(
        self,
        x,
        history_len,
        generation_len,
        future_len,
        history_mask,
        generation_mask,
        future_mask,
        history_token_mask,
        generation_token_mask,
        future_token_mask,
        text_feat,
        text_pad_mask,
        t,
        first_heading_angle,
        motion_mask,
        observed_motion,
        num_denoising_steps,
        cfg_weight,
        target_motion=None,
        cfg_type=None,
    ):
        prediction = self.denoiser.model(
            x,
            history_len,
            generation_len,
            future_len,
            history_mask,
            generation_mask,
            future_mask,
            history_token_mask,
            generation_token_mask,
            future_token_mask,
            text_feat,
            text_pad_mask,
            t,
            first_heading_angle,
            motion_mask,
            observed_motion,
        )
        return prediction


class FakeSeparatedCfgArdy(FakeArdy):
    def denoising_step(
        self,
        x,
        history_len,
        generation_len,
        future_len,
        history_mask,
        generation_mask,
        future_mask,
        history_token_mask,
        generation_token_mask,
        future_token_mask,
        text_feat,
        text_pad_mask,
        t,
        first_heading_angle,
        motion_mask,
        observed_motion,
        num_denoising_steps,
        cfg_weight,
        target_motion=None,
        cfg_type=None,
    ):
        kwargs = locals()
        kwargs.pop("self")
        repeated = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == kwargs["x"].shape[0]:
                repeated[key] = torch.cat([value, value, value], dim=0)
            else:
                repeated[key] = value
        prediction = self.denoiser.model(
            repeated["x"],
            repeated["history_len"],
            repeated["generation_len"],
            repeated["future_len"],
            repeated["history_mask"],
            repeated["generation_mask"],
            repeated["future_mask"],
            repeated["history_token_mask"],
            repeated["generation_token_mask"],
            repeated["future_token_mask"],
            repeated["text_feat"],
            repeated["text_pad_mask"],
            repeated["t"],
            repeated["first_heading_angle"],
            repeated["motion_mask"],
            repeated["observed_motion"],
        )
        return prediction.chunk(3, dim=0)[0]


def _denoising_kwargs(timestep):
    batch, tokens = 2, 4
    history_token = torch.tensor(
        [[True, True, False, False], [True, True, False, False]]
    )
    generation_token = ~history_token
    return {
        "x": torch.randn(batch, tokens, 32),
        "history_len": torch.full((batch,), 2, dtype=torch.long),
        "generation_len": torch.full((batch,), 2, dtype=torch.long),
        "future_len": torch.zeros(batch, dtype=torch.long),
        "history_mask": history_token,
        "generation_mask": generation_token,
        "future_mask": torch.zeros_like(history_token),
        "history_token_mask": history_token,
        "generation_token_mask": generation_token,
        "future_token_mask": torch.zeros_like(history_token),
        "text_feat": torch.randn(batch, 2, 24),
        "text_pad_mask": torch.ones(batch, 2, dtype=torch.bool),
        "t": torch.full((batch,), timestep, dtype=torch.long),
        "first_heading_angle": None,
        "motion_mask": None,
        "observed_motion": None,
        "num_denoising_steps": torch.tensor([4]),
        "cfg_weight": 2.0,
    }


def test_diffusion_controller_commits_once_only_after_final_accepted_step(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()
    ardy = FakeArdy(FakeRawDenoiser(root, body))
    controller = ArdyDiffusionMemoryController(ardy).install()

    for timestep in (3, 2, 1):
        ardy.denoising_step(**_denoising_kwargs(timestep))
        assert controller.trace.accepted_commits == 0
        assert root.accepted_commits == body.accepted_commits == 0
        assert all(
            state.attention_cache.num_chunks == 0
            for state in root._sequence_state.layers
        )

    ardy.denoising_step(**_denoising_kwargs(0))
    assert controller.trace.accepted_commits == 1
    assert root.accepted_commits == body.accepted_commits == 1
    for block in (root, body):
        for state in block._sequence_state.layers:
            assert state.attention_cache.chunk_ids == (0,)
            assert state.attention_cache.chunks[0].num_tokens == 2

    for timestep in (3, 1, 0):
        ardy.denoising_step(**_denoising_kwargs(timestep))
    assert controller.trace.accepted_commits == 2
    for block in (root, body):
        for state in block._sequence_state.layers:
            assert state.attention_cache.chunk_ids == (0, 1)

    controller.reset()
    assert root._sequence_state is None and body._sequence_state is None
    assert controller.trace.accepted_commits == 0


def test_controller_records_per_layer_state_and_periodically_resets_memory(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()
    ardy = FakeArdy(FakeRawDenoiser(root, body))
    controller = ArdyDiffusionMemoryController(ardy).install()
    controller.configure_rollout_diagnostics(
        enabled=True, periodic_reset_windows=2
    )

    ardy.denoising_step(**_denoising_kwargs(0))
    assert root._sequence_state is not None
    first = controller.state_diagnostics[-1]
    assert first["event"] == "generated_window"
    assert first["periodic_reset_after_commit"] is False
    assert len(first["blocks"]["root"]) == 2
    assert first["blocks"]["root"][0]["attention_cache_chunks"] == 1

    ardy.denoising_step(**_denoising_kwargs(0))
    second = controller.state_diagnostics[-1]
    assert second["periodic_reset_after_commit"] is True
    assert second["all_finite"] is True
    assert root._sequence_state is None and body._sequence_state is None
    assert controller.trace.accepted_commits == 2


def test_controller_applies_depth_dependent_mamba_output_decay(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()
    ardy = FakeArdy(FakeRawDenoiser(root, body))
    controller = ArdyDiffusionMemoryController(ardy).install()
    controller.configure_mamba_output_decay(
        start_windows=0, half_life_windows=1, min_scale=0.4
    )

    ardy.denoising_step(**_denoising_kwargs(0))
    assert root.sequence_stack.layers[0].mixer.runtime_mamba_output_scale == 1.0
    ardy.denoising_step(**_denoising_kwargs(0))
    assert root.sequence_stack.layers[0].mixer.runtime_mamba_output_scale == 0.5
    ardy.denoising_step(**_denoising_kwargs(0))
    assert root.sequence_stack.layers[0].mixer.runtime_mamba_output_scale == 0.4
    controller.reset()
    assert root.sequence_stack.layers[0].mixer.runtime_mamba_output_scale == 1.0


def test_controller_captures_and_restores_selected_training_depths(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()
    ardy = FakeArdy(FakeRawDenoiser(root, body))
    controller = ArdyDiffusionMemoryController(ardy).install()

    controller.begin_training_snapshot_capture([1, 3])
    for _ in range(3):
        ardy.denoising_step(**_denoising_kwargs(0))
    snapshots, history_chunks = controller.end_training_snapshot_capture()

    assert sorted(snapshots) == [1, 3]
    assert history_chunks == []  # This lightweight double has no post-window hook.
    assert snapshots[1][0].layers[0].attention_cache.num_chunks == 1
    assert snapshots[3][0].layers[0].attention_cache.chunk_ids == (1, 2)
    controller.restore_training_snapshot(snapshots[1])
    assert root._sequence_state.layers[0].attention_cache.num_chunks == 1
    assert body._sequence_state.layers[0].attention_cache.num_chunks == 1
    with pytest.raises(ValueError, match="unique"):
        controller.begin_training_snapshot_capture([1, 1])


def test_training_snapshot_reflects_periodic_reset_state(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()
    ardy = FakeArdy(FakeRawDenoiser(root, body))
    controller = ArdyDiffusionMemoryController(ardy).install()
    controller.configure_rollout_diagnostics(enabled=False, periodic_reset_windows=2)

    controller.begin_training_snapshot_capture([2])
    for _ in range(2):
        ardy.denoising_step(**_denoising_kwargs(0))
    snapshots, history_chunks = controller.end_training_snapshot_capture()

    assert snapshots[2] == (None, None)
    assert history_chunks == []


def test_separated_cfg_branches_share_base_batch_history_and_commit_once(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()
    ardy = FakeSeparatedCfgArdy(FakeRawDenoiser(root, body))
    controller = ArdyDiffusionMemoryController(ardy).install()

    for timestep in (3, 2, 1):
        ardy.denoising_step(**_denoising_kwargs(timestep))
        assert controller.trace.accepted_commits == 0
        assert root._sequence_state.layers[0].attention_cache.batch_size == 2
        assert body._sequence_state.layers[0].attention_cache.batch_size == 2

    ardy.denoising_step(**_denoising_kwargs(0))
    assert controller.trace.accepted_commits == 1
    assert root.accepted_commits == body.accepted_commits == 1
    for block in (root, body):
        assert block._sequence_state.layers[0].attention_cache.chunk_ids == (0,)


class FakeOldEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 64)
        self.nhead = 2
        self.attn_dropout_p = 0.0


class FakeOldBlock(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.use_text_mask = True
        self.embed_text = nn.Linear(24, 32)
        self.sequence_pos_encoder = FakePositionalEncoding(32, 0.0)
        self.embed_timestep = FakeTimestepEmbedder(32, self.sequence_pos_encoder)
        self.input_linear = nn.Identity()
        self.output_linear = nn.Linear(32, output_dim)
        self.motion_token_embedding = FakeNegativeIndexEncoding(32, 0.0)
        self.seqTransEncoder = SimpleNamespace(layers=nn.ModuleList([FakeOldEncoderLayer(), FakeOldEncoderLayer()]))


class FakeInstallRaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.motion_rep = SimpleNamespace(skeleton=SimpleNamespace(nbjoints=27))
        self.llm_shape = [2, 24]
        self.latent_dim = 32
        self.positional_encoding_mode = "learned_prefix_zero_at_first_generation"
        self.sparsify_token_seq = False
        self.root_model = FakeOldBlock(32)
        self.body_model = FakeOldBlock(32)


def test_install_factory_exposes_original_dit_and_minimal_without_mamba(monkeypatch):
    _patch_ardy(monkeypatch)
    original_raw = FakeInstallRaw()
    original_ardy = FakeArdy(original_raw)
    original = install_ardy_sequence_kernel(
        original_ardy, ArdySequenceKernelConfig(mode="original")
    )
    assert original.controller is None
    assert original_raw.root_model.__class__ is FakeOldBlock

    for mode in ("dit", "minimal"):
        raw = FakeInstallRaw()
        ardy = FakeArdy(raw)
        modules_before = set(__import__("sys").modules)
        client = install_ardy_sequence_kernel(
            ardy,
            ArdySequenceKernelConfig(
                mode=mode,
                attention_backend="sdpa",
                window_size_chunks=2,
                num_sink_tokens=2,
            ),
        )
        assert isinstance(raw.root_model, ArdyDiTSequenceBlock)
        assert isinstance(raw.body_model, ArdyDiTSequenceBlock)
        assert client.controller is not None
        assert client.controller.commit_enabled is (mode == "minimal")
        assert next(raw.root_model.parameters()).device == raw.anchor.device
        assert next(raw.body_model.parameters()).device == raw.anchor.device
        new_modules = set(__import__("sys").modules) - modules_before
        assert not any(
            name == "mamba_ssm" or name.startswith("mamba_ssm.")
            for name in new_modules
        )
        module_names = [type(module).__name__.lower() for module in raw.modules()]
        assert not any("mamba" in name for name in module_names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA placement regression")
def test_install_factory_moves_full_minimal_blocks_to_ardy_cuda_device(monkeypatch):
    _patch_ardy(monkeypatch)
    raw = FakeInstallRaw().cuda()
    ardy = FakeArdy(raw)
    install_ardy_sequence_kernel(
        ardy,
        ArdySequenceKernelConfig(
            mode="minimal",
            attention_backend="sdpa",
            window_size_chunks=2,
            num_sink_tokens=2,
        ),
    )
    assert {parameter.device.type for parameter in raw.root_model.parameters()} == {
        "cuda"
    }
    assert {parameter.device.type for parameter in raw.body_model.parameters()} == {
        "cuda"
    }
    block_inputs = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in _inputs().items()
    }
    root_output = raw.root_model(**block_inputs)
    body_output = raw.body_model(**block_inputs)
    assert root_output.device.type == body_output.device.type == "cuda"
    assert torch.isfinite(root_output).all() and torch.isfinite(body_output).all()


def test_failed_body_commit_rolls_back_root_and_body_atomically(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()
    ardy = FakeArdy(FakeRawDenoiser(root, body))
    controller = ArdyDiffusionMemoryController(ardy).install()
    original_body_forward = body.forward

    def fail_only_when_armed(*args, **kwargs):
        if body._accepted_commit is not None:
            raise RuntimeError("synthetic body commit failure")
        return original_body_forward(*args, **kwargs)

    body.forward = fail_only_when_armed
    with pytest.raises(RuntimeError, match="synthetic body"):
        ardy.denoising_step(**_denoising_kwargs(0))

    assert controller.trace.accepted_commits == 0
    assert root.accepted_commits == body.accepted_commits == 0
    assert root._accepted_commit is None and body._accepted_commit is None
    for block in (root, body):
        assert all(
            state.attention_cache.num_chunks == 0
            for state in block._sequence_state.layers
        )


def test_missing_body_commit_is_detected_and_rolled_back(monkeypatch):
    root = _make_block(monkeypatch).eval()
    body = _make_block(monkeypatch).eval()
    ardy = FakeArdy(FakeRawDenoiser(root, body))
    controller = ArdyDiffusionMemoryController(ardy).install()
    original_body_forward = body.forward

    def suppress_body_commit_count(*args, **kwargs):
        commits_before = body.accepted_commits
        output = original_body_forward(*args, **kwargs)
        if commits_before != body.accepted_commits:
            body.accepted_commits = commits_before
        return output

    body.forward = suppress_body_commit_count
    with pytest.raises(RuntimeError, match="body memory exactly once"):
        ardy.denoising_step(**_denoising_kwargs(0))

    assert controller.trace.accepted_commits == 0
    assert root.accepted_commits == body.accepted_commits == 0
    assert root._accepted_commit is None and body._accepted_commit is None
    for block in (root, body):
        assert all(
            state.attention_cache.num_chunks == 0
            for state in block._sequence_state.layers
        )
