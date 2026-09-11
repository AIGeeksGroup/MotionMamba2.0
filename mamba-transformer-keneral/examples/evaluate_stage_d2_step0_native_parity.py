#!/usr/bin/env python3
"""Compare Native Hybrid 50k with its neutral D2 migration on BONES validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "ardy"))
for path in (
    os.environ.get("MAMBA_REPO", "/data3/zts/packages/state-spaces-mamba"),
    os.environ.get("MAMBA_TRITON_OVERLAY", "/data3/zts/packages/mamba_op_triton"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from ardy_adapter.history_commit import canonical_history_prime  # noqa: E402
from ardy_data import HumanML3DArdyDataset  # noqa: E402
from train_humanml3d_ardy_tiny import (  # noqa: E402
    _build_batch,
    _load_denoiser_weights,
    _replace_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9032026)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model", default="g1")
    parser.add_argument("--remediation", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _raw_forward(raw, fixed: dict) -> torch.Tensor:
    return raw(
        fixed["x"],
        fixed["history_len"],
        fixed["generation_len"],
        fixed["future_len"],
        fixed["history_mask"],
        fixed["generation_mask"],
        fixed["future_mask"],
        fixed["history_token_mask"],
        fixed["generation_token_mask"],
        fixed["future_token_mask"],
        fixed["text"],
        fixed["text_mask"],
        fixed["timestep"],
        fixed["heading"],
        fixed["motion_mask"],
        fixed["observed"],
    )


def _forward_with_block_capture(raw, fixed: dict) -> tuple[torch.Tensor, dict]:
    captured: dict[str, list[torch.Tensor]] = {"root": [], "body": []}
    handles = []
    for name, block in (("root", raw.root_model), ("body", raw.body_model)):
        handles.append(
            block.register_forward_hook(
                lambda _module, _inputs, output, key=name: captured[key].append(
                    output.detach().float().clone()
                )
            )
        )
    try:
        output = _raw_forward(raw, fixed)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured["root"]) != 1 or len(captured["body"]) != 1:
        raise RuntimeError("expected exactly one root and body block call")
    return output, {name: values[0] for name, values in captured.items()}


def _state_tensors(state: Any) -> list[torch.Tensor]:
    if state is None:
        return []
    if isinstance(state, torch.Tensor):
        return [state]
    tensors = []
    for name in ("angle_dt_state", "ssm_state", "k_state", "v_state"):
        value = getattr(state, name, None)
        if value is not None:
            tensors.extend(_state_tensors(value))
    return tensors


def _checksum(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _committed_tensors(block) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    attention = []
    mamba = []
    if block._sequence_state is None:
        return attention, mamba
    for layer in block._sequence_state.layers:
        for chunk in layer.attention_cache.chunks:
            attention.extend((chunk.key, chunk.value))
        mamba.extend(_state_tensors(layer.mamba_state))
    return attention, mamba


def _max_abs_difference(first: list[torch.Tensor], second: list[torch.Tensor]) -> float:
    if len(first) != len(second):
        raise RuntimeError("state tensor layouts differ")
    if not first:
        return 0.0
    return max(
        float((left.detach().float() - right.detach().float()).abs().max())
        for left, right in zip(first, second)
    )


def _denoising_losses(raw, prediction: torch.Tensor, fixed: dict) -> dict:
    selected_prediction = prediction[fixed["generation_token_mask"]].float()
    selected_target = fixed["target"][fixed["generation_token_mask"]].float()
    root_dim = int(raw.nframe_root_dim)
    root = F.mse_loss(
        selected_prediction[:, :root_dim], selected_target[:, :root_dim]
    )
    body = F.mse_loss(
        selected_prediction[:, root_dim:], selected_target[:, root_dim:]
    )
    return {"root": float(root), "body": float(body), "total": float(root + body)}


def _geometry_proxy(model, prediction: torch.Tensor, fixed: dict) -> dict:
    explicit = model.hybrid.get_explicit_motion_from_hybrid(
        prediction, fixed["pad_mask"], fixed["lengths"]
    )
    predicted_physical = model.motion_rep.unnormalize(explicit)
    target_physical = model.motion_rep.unnormalize(fixed["motion"])
    generation = fixed["generation_mask"]
    root_error = torch.linalg.vector_norm(
        predicted_physical[:, :, [0, 2]] - target_physical[:, :, [0, 2]], dim=-1
    )[generation].mean() * 100.0
    predicted_joints = model.motion_rep.inverse(
        explicit, is_normalized=True
    )["posed_joints"]
    target_joints = fixed["target_joints"]
    joint_error = torch.linalg.vector_norm(
        predicted_joints - target_joints, dim=-1
    )[generation].mean() * 100.0
    constraint_mask = fixed["motion_mask"].bool() & generation[:, :, None]
    constraint = (
        (explicit - fixed["motion"])[constraint_mask].abs().mean()
        if bool(constraint_mask.any())
        else explicit.new_zeros(())
    )
    return {
        "root_ade_cm": float(root_error),
        "joint_error_cm": float(joint_error),
        "normalized_constraint_l1": float(constraint),
    }


def _content_diagnostics(raw) -> dict:
    slot_norms = []
    read_scales = []
    address_scales = []
    gate_means = []
    for block in (raw.root_model, raw.body_model):
        for layer, state in zip(
            block.sequence_stack.layers, block._sequence_state.layers
        ):
            mixer = layer.mixer
            slot_norms.append(
                float(state.persistent_memory.detach().float().norm(dim=-1).mean())
            )
            read_scales.append(
                float(mixer.content_memory.read_residual_scale.detach())
            )
            address_scales.append(
                float(mixer.token_state_interaction.effective_scale.detach())
            )
            value = mixer.last_content_memory_diagnostics.get("write_gate_mean")
            if value is not None:
                gate_means.append(float(value))
    return {
        "memory_slot_norm_mean": sum(slot_norms) / len(slot_norms),
        "read_residual_scale_mean": sum(read_scales) / len(read_scales),
        "state_query_effective_scale_mean": sum(address_scales) / len(address_scales),
        "write_gate_mean": sum(gate_means) / len(gate_means),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    checkpoint_path = Path(args.checkpoint)
    expected_parent_sha = (
        "cfa9d44d9a804e3f71dbfde98cbad7faceed98bca83b5417c3afb5bca6071a4a"
    )
    parent_sha = _sha256(checkpoint_path)
    if parent_sha != expected_parent_sha:
        raise RuntimeError("healthy Native Hybrid parent SHA mismatch")

    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    def build(*, d2: bool):
        torch.manual_seed(args.seed)
        model = load_model(args.model, device=str(device), text_encoder=False)
        _replace_stats(model, args.stats_dir)
        client = install_ardy_sequence_kernel(
            model,
            ArdySequenceKernelConfig(
                mode="hybrid",
                attention_backend="sdpa",
                window_size_chunks=4,
                num_sink_tokens=4,
                token_state_interaction=("state_query" if d2 else "naive_gated_add"),
                state_summary_dim=128,
                use_state_delta=True,
                state_conditioning_scale_init=0.0,
                state_aware_output_mode=("native_gated_add" if d2 else "attention_only"),
                fusion_gate_logit_init=0.0,
                content_memory_enabled=d2,
                content_memory_slots=2,
                content_memory_use_state_delta=True,
                content_memory_write_gate_context="previous_memory",
                content_memory_decoder_init_scale=0.02,
                content_memory_write_gate_init_scale=0.02,
                content_memory_write_gate_bias_init=0.0,
                content_memory_read_scale_init=0.0,
                content_memory_conditioning=(
                    "text" if d2 and args.remediation else "none"
                ),
                content_memory_normalize_delta=bool(d2 and args.remediation),
                content_memory_normalize_slots=bool(d2 and args.remediation),
                content_memory_max_write_rate=(
                    0.25 if d2 and args.remediation else 1.0
                ),
            ),
        )
        _load_denoiser_weights(model.denoiser.model, checkpoint)
        model.denoiser.model.eval()
        return model, client

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    native_model, native_client = build(d2=False)
    d2_model, d2_client = build(d2=True)
    horizon = int(native_model.gen_horizon_len)
    if horizon != 52:
        raise RuntimeError(f"expected G1 52-frame horizon, got {horizon}")
    dataset = HumanML3DArdyDataset(
        args.manifest,
        args.stats_dir,
        text_cache_dir=args.text_cache,
        frames=horizon,
        horizon=horizon,
        condition_dropout=0.0,
        text_dropout=0.0,
        history_probability=0.0,
        seed=args.seed,
    )
    sample = dataset.sample(
        args.sample_index,
        args.seed,
        force_history=False,
        out_of_horizon=False,
    )
    batch = default_collate([sample])
    native_model.diffusion.calc_diffusion_vars(
        torch.arange(native_model.diffusion.num_base_steps, device=device)
    )
    fixed = _build_batch(
        native_model,
        batch,
        device,
        fixed_noise=True,
        boundary_loss_weight=0.0,
        root_velocity_loss_weight=0.0,
        constraint_loss_weight=0.0,
        fk_constraint_loss_weight=0.0,
        foot_skate_loss_weight=0.0,
    )

    with torch.no_grad():
        native_client.reset_memory()
        d2_client.reset_memory()
        native_output, native_blocks = _forward_with_block_capture(
            native_model.denoiser.model, fixed
        )
        d2_output, d2_blocks = _forward_with_block_capture(
            d2_model.denoiser.model, fixed
        )
        native_losses = _denoising_losses(
            native_model.denoiser.model, native_output, fixed
        )
        d2_losses = _denoising_losses(d2_model.denoiser.model, d2_output, fixed)
        native_geometry = _geometry_proxy(native_model, native_output, fixed)
        d2_geometry = _geometry_proxy(d2_model, d2_output, fixed)

        history = fixed["target"][fixed["generation_token_mask"]].reshape(
            1, -1, fixed["target"].shape[-1]
        )
        native_client.reset_memory()
        d2_client.reset_memory()
        for raw in (native_model.denoiser.model, d2_model.denoiser.model):
            canonical_history_prime(
                raw,
                history,
                text_feat=fixed["text"],
                text_pad_mask=fixed["text_mask"],
                first_heading_angle=fixed["heading"],
                next_generation_frames=horizon,
                chunk_id=0,
                detach=True,
            )

    state_comparisons = {}
    for block_name in ("root", "body"):
        native_block = getattr(native_model.denoiser.model, f"{block_name}_model")
        d2_block = getattr(d2_model.denoiser.model, f"{block_name}_model")
        native_attention, native_mamba = _committed_tensors(native_block)
        d2_attention, d2_mamba = _committed_tensors(d2_block)
        state_comparisons[block_name] = {
            "attention_cache_max_abs_diff": _max_abs_difference(
                native_attention, d2_attention
            ),
            "mamba_state_max_abs_diff": _max_abs_difference(native_mamba, d2_mamba),
            "native_attention_checksum": _checksum(native_attention),
            "d2_attention_checksum": _checksum(d2_attention),
            "native_mamba_checksum": _checksum(native_mamba),
            "d2_mamba_checksum": _checksum(d2_mamba),
        }

    raw_diffs = {
        "combined_max_abs": float(
            (native_output.float() - d2_output.float()).abs().max()
        ),
        "root_block_max_abs": float(
            (native_blocks["root"] - d2_blocks["root"]).abs().max()
        ),
        "body_block_max_abs": float(
            (native_blocks["body"] - d2_blocks["body"]).abs().max()
        ),
    }
    loss_diff = abs(native_losses["total"] - d2_losses["total"])
    geometry_max_diff = max(
        abs(native_geometry[key] - d2_geometry[key]) for key in native_geometry
    )
    gate = {
        "raw_output_max_abs_at_most_1e-6": max(raw_diffs.values()) <= 1e-6,
        "denoising_loss_abs_diff_at_most_1e-7": loss_diff <= 1e-7,
        "attention_cache_max_abs_diff_at_most_1e-6": max(
            value["attention_cache_max_abs_diff"]
            for value in state_comparisons.values()
        ) <= 1e-6,
        "mamba_state_max_abs_diff_at_most_1e-6": max(
            value["mamba_state_max_abs_diff"]
            for value in state_comparisons.values()
        ) <= 1e-6,
        "geometry_proxy_max_abs_diff_at_most_1e-5": geometry_max_diff <= 1e-5,
    }
    gate["status"] = "PASS" if all(gate.values()) else "FAIL"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": (
            "d2_remediation_step0_native_parity_v1"
            if args.remediation
            else "d2_step0_native_parity_v1"
        ),
        "git_sha": _git_sha(),
        "parent_checkpoint": str(checkpoint_path.resolve()),
        "parent_checkpoint_sha256": parent_sha,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "sample_index": args.sample_index,
        "sample_id": sample["id"],
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "raw_output": raw_diffs,
        "denoising_loss": {
            "native": native_losses,
            "d2_initial": d2_losses,
            "total_abs_diff": loss_diff,
        },
        "geometry_52_frame_proxy": {
            "native": native_geometry,
            "d2_initial": d2_geometry,
            "max_abs_diff": geometry_max_diff,
        },
        "committed_state": state_comparisons,
        "d2_content": _content_diagnostics(d2_model.denoiser.model),
        "gate": gate,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if gate["status"] != "PASS":
        raise RuntimeError("D2 step-0 Native parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
