"""Milestone B validation (`docs/ardy_mamba_replacement_plan.md`): shape
parity between `ArdyMambaBlock` and ARDY's real `TransformerEncoderBlock`,
using the actual released "core" model's `denoiser` config values
(`latent_dim=1024`, `num_layers=8`, `llm_shape=[1,4096]`,
`positional_encoding_mode="learned_prefix_zero_at_first_generation"`,
`input_first_heading_angle=True`, `add_input_proj=False` -- matching how
`AutoLatentTwostageDenoiser` actually constructs `root_model`/`body_model`).

Requires `ardy` importable (PYTHONPATH set to the ARDY repo) -- run on the
server, not locally.

Run with:

PYTHONPATH=/data3/zts/ardy \
/data3/zts/conda_envs/ardy_env/bin/python -m pytest \
    tests/test_ardy_mamba_block.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

TRITON_OVERLAY = os.environ.get("MAMBA_TRITON_OVERLAY", "/tmp/mamba_op_triton")
MAMBA_REPO = os.environ.get("MAMBA_REPO", "/home/plote/mambamotion/basecode/mamba/state-spaces-mamba")
ARDY_ROOT = os.environ.get("ARDY_ROOT", "/data3/zts/ardy")

requires_ardy = pytest.mark.skipif(
    not Path(ARDY_ROOT).exists(), reason=f"ARDY repo not found at {ARDY_ROOT}"
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _setup_paths():
    if torch.cuda.is_available():
        torch.empty(1, device="cuda")
    for path in (TRITON_OVERLAY, MAMBA_REPO, ARDY_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


# Real values from the released ARDY-Core-RP-20FPS-Horizon40 denoiser config
# (config.yaml's `denoiser:` section) -- not made up.
DENOISER_KWARGS = dict(
    llm_shape=[1, 4096],
    use_text_mask=False,
    latent_dim=1024,
    ff_size=2048,
    num_layers=8,
    num_heads=8,
    activation="gelu",
    dropout=0.0,
    pe_dropout=0.0,
    norm_first=False,
    input_first_heading_angle=True,
    positional_encoding_mode="learned_prefix_zero_at_first_generation",
)
ROOT_OUTPUT_DIM = 20  # nframe_root_dim
BODY_OUTPUT_DIM = 128  # latent_embedding_dim


def _make_stub_skeleton():
    return type("StubSkeleton", (), {"nbjoints": 27})()


def _make_inputs(batch, num_tokens, latent_dim, llm_dim, device, dtype=torch.float32):
    torch.manual_seed(0)
    x = torch.randn(batch, num_tokens, latent_dim, device=device, dtype=dtype)
    x_pad_mask = torch.ones(batch, num_tokens, dtype=torch.bool, device=device)
    text_feat = torch.randn(batch, 1, llm_dim, device=device, dtype=dtype)
    text_feat_pad_mask = torch.ones(batch, 1, dtype=torch.bool, device=device)
    timesteps = torch.randint(0, 10, (batch,), device=device)
    first_heading_angle = torch.rand(batch, device=device) * 6.28
    token_index = torch.arange(num_tokens, device=device)[None, :].repeat(batch, 1) - num_tokens // 2
    return x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps, first_heading_angle, token_index


@requires_ardy
@requires_cuda
def test_shape_parity_root_stage():
    _setup_paths()
    from ardy.model.backbone import TransformerEncoderBlock

    from ardy_adapter.ardy_mamba_block import ArdyMambaBlock

    device = torch.device("cuda")
    batch, num_tokens = 2, 10

    transformer_block = TransformerEncoderBlock(
        add_input_proj=False,
        input_dim=-1,
        output_dim=ROOT_OUTPUT_DIM,
        skeleton=_make_stub_skeleton(),
        **DENOISER_KWARGS,
    ).to(device)

    mamba_block = ArdyMambaBlock(
        add_input_proj=False,
        input_dim=-1,
        output_dim=ROOT_OUTPUT_DIM,
        skeleton=_make_stub_skeleton(),
        device=device,
        **DENOISER_KWARGS,
    ).to(device)

    inputs = _make_inputs(batch, num_tokens, DENOISER_KWARGS["latent_dim"], DENOISER_KWARGS["llm_shape"][-1], device)

    with torch.no_grad():
        out_transformer = transformer_block(*inputs)
        out_mamba = mamba_block(*inputs)

    assert out_transformer.shape == (batch, num_tokens, ROOT_OUTPUT_DIM)
    assert out_mamba.shape == out_transformer.shape
    assert torch.isfinite(out_mamba).all()


@requires_ardy
@requires_cuda
def test_shape_parity_body_stage():
    _setup_paths()
    from ardy_adapter.ardy_mamba_block import ArdyMambaBlock

    device = torch.device("cuda")
    batch, num_tokens = 2, 10

    mamba_block = ArdyMambaBlock(
        add_input_proj=False,
        input_dim=-1,
        output_dim=BODY_OUTPUT_DIM,
        skeleton=_make_stub_skeleton(),
        device=device,
        **DENOISER_KWARGS,
    ).to(device)

    inputs = _make_inputs(batch, num_tokens, DENOISER_KWARGS["latent_dim"], DENOISER_KWARGS["llm_shape"][-1], device)
    with torch.no_grad():
        out = mamba_block(*inputs)
    assert out.shape == (batch, num_tokens, BODY_OUTPUT_DIM)
    assert torch.isfinite(out).all()


@requires_ardy
@requires_cuda
def test_representation_capture_is_prediction_preserving():
    _setup_paths()
    from ardy_adapter.ardy_mamba_block import ArdyMambaBlock

    device = torch.device("cuda")
    block = ArdyMambaBlock(
        add_input_proj=False, input_dim=-1, output_dim=ROOT_OUTPUT_DIM,
        skeleton=_make_stub_skeleton(), device=device, **DENOISER_KWARGS,
    ).to(device).eval()
    inputs = _make_inputs(
        1, 6, DENOISER_KWARGS["latent_dim"],
        DENOISER_KWARGS["llm_shape"][-1], device,
    )
    with torch.no_grad():
        reference = block(*inputs)
        block.enable_representation_capture(layers=(0, DENOISER_KWARGS["num_layers"] - 1))
        captured = block(*inputs)
    assert torch.equal(reference, captured)
    representations = block.captured_representations
    assert set(representations["token_hidden"]) == {0, DENOISER_KWARGS["num_layers"] - 1}
    assert set(representations["ssm_state_summary"]) == {
        0, DENOISER_KWARGS["num_layers"] - 1
    }
    assert representations["token_hidden"][DENOISER_KWARGS["num_layers"] - 1].shape == (
        1, 6, DENOISER_KWARGS["latent_dim"]
    )


@requires_ardy
@requires_cuda
def test_mask_parity_padded_tokens_do_not_crash():
    _setup_paths()
    from ardy_adapter.ardy_mamba_block import ArdyMambaBlock

    device = torch.device("cuda")
    batch, num_tokens = 2, 10

    mamba_block = ArdyMambaBlock(
        add_input_proj=False,
        input_dim=-1,
        output_dim=ROOT_OUTPUT_DIM,
        skeleton=_make_stub_skeleton(),
        device=device,
        **DENOISER_KWARGS,
    ).to(device)

    x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps, first_heading_angle, token_index = _make_inputs(
        batch, num_tokens, DENOISER_KWARGS["latent_dim"], DENOISER_KWARGS["llm_shape"][-1], device
    )
    # Mark the trailing half of each sequence as padding (the common
    # collation convention) -- must not crash and must still produce
    # finite, correctly-shaped output.
    x_pad_mask[:, num_tokens // 2 :] = False

    with torch.no_grad():
        out = mamba_block(x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps, first_heading_angle, token_index)
    assert out.shape == (batch, num_tokens, ROOT_OUTPUT_DIM)
    assert torch.isfinite(out).all()


@requires_ardy
@requires_cuda
def test_single_denoising_step_backward_smoke():
    _setup_paths()
    from ardy_adapter.ardy_mamba_block import ArdyMambaBlock

    device = torch.device("cuda")
    batch, num_tokens = 2, 10

    mamba_block = ArdyMambaBlock(
        add_input_proj=False,
        input_dim=-1,
        output_dim=ROOT_OUTPUT_DIM,
        skeleton=_make_stub_skeleton(),
        device=device,
        **DENOISER_KWARGS,
    ).to(device)

    x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps, first_heading_angle, token_index = _make_inputs(
        batch, num_tokens, DENOISER_KWARGS["latent_dim"], DENOISER_KWARGS["llm_shape"][-1], device
    )
    x.requires_grad_(True)

    out = mamba_block(x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps, first_heading_angle, token_index)
    loss = out.float().sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert mamba_block.output_linear.weight.grad is not None
    assert mamba_block.mamba_layers[0].in_proj.weight.grad is not None
