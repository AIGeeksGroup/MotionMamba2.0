"""Real-data/control tiny-overfit gate for Minimal or Hybrid ARDY mixers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate

REPO = Path(__file__).resolve().parents[1]
ARDY_ROOT = REPO.parents[0] / "ardy"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ARDY_ROOT))
for path in (
    os.environ.get("MAMBA_REPO", "/data3/zts/packages/state-spaces-mamba"),
    os.environ.get("MAMBA_TRITON_OVERLAY", "/data3/zts/packages/mamba_op_triton"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from ardy_data import HumanML3DArdyDataset  # noqa: E402
from ardy_adapter.history_commit import canonical_history_prime  # noqa: E402
from ardy_adapter.differentiable_rollout import (  # noqa: E402
    differentiable_rollout_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--text-root", default=None)
    parser.add_argument("--text-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["minimal", "hybrid"], required=True)
    parser.add_argument("--model", default="core")
    parser.add_argument(
        "--train-scope",
        choices=["mixer", "denoiser", "state_memory"],
        default="denoiser",
    )
    parser.add_argument("--phase", choices=["tiny", "pilot"], default="tiny")
    parser.add_argument("--dev-manifest", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tiny-samples", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--history-probability", type=float, default=0.0)
    parser.add_argument("--constraint-dropout", type=float, default=0.1)
    parser.add_argument("--text-dropout", type=float, default=0.1)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--root-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--constraint-loss-weight", type=float, default=0.0)
    parser.add_argument("--fk-constraint-loss-weight", type=float, default=0.0)
    parser.add_argument("--foot-skate-loss-weight", type=float, default=0.0)
    parser.add_argument("--differentiable-rollout-probability", type=float, default=0.0)
    parser.add_argument("--differentiable-rollout-windows", type=int, default=0)
    parser.add_argument("--differentiable-rollout-diffusion-steps", type=int, default=1)
    parser.add_argument("--differentiable-rollout-warm-prefix-windows", type=int, default=0)
    parser.add_argument("--long-slope-loss-weight", type=float, default=0.0)
    parser.add_argument("--long-auc-loss-weight", type=float, default=0.0)
    parser.add_argument("--long-final-loss-weight", type=float, default=0.0)
    parser.add_argument("--long-slope-margin-cm-per-window", type=float, default=0.0)
    parser.add_argument("--rollout-gradient-checkpointing", action="store_true")
    parser.add_argument("--attention-window-chunks", type=int, default=4)
    parser.add_argument(
        "--token-state-interaction",
        choices=["naive_gated_add", "state_query", "state_kv_modulation"],
        default="naive_gated_add",
    )
    parser.add_argument("--state-summary-dim", type=int, default=128)
    parser.add_argument(
        "--use-state-delta", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--state-conditioning-scale-init", type=float, default=0.0)
    parser.add_argument(
        "--state-aware-output-mode",
        choices=["attention_only", "native_gated_add"],
        default="attention_only",
    )
    parser.add_argument("--content-memory-enabled", action="store_true")
    parser.add_argument("--content-memory-slots", type=int, default=1)
    parser.add_argument(
        "--content-memory-use-state-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--content-memory-write-gate-context",
        choices=["previous_memory", "proposal"],
        default="previous_memory",
    )
    parser.add_argument("--content-memory-decoder-init-scale", type=float, default=0.02)
    parser.add_argument("--content-memory-write-gate-init-scale", type=float, default=0.02)
    parser.add_argument("--content-memory-write-gate-bias-init", type=float, default=0.0)
    parser.add_argument("--content-memory-read-scale-init", type=float, default=0.0)
    parser.add_argument(
        "--minimal-ff-size", type=int, default=None,
        help="Override Minimal FFN width; 5248 is the registered Hybrid parameter match.",
    )
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def _replace_stats(model, stats_dir: str):
    from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep

    old = model.motion_rep
    rep = ArdyMotionRep(old.skeleton, old.fps, stats_path=stats_dir)
    rep.skeleton = old.skeleton
    model.motion_rep = rep
    raw = model.denoiser.model
    raw.motion_rep = rep
    model.autoencoder._motion_rep = rep
    for converter in (model.hybrid, raw.hybrid):
        converter._motion_rep = rep
        converter.__dict__.pop("motion_rep", None)
    return rep


def _hashed_text(texts: list[str], dim: int, device: torch.device) -> torch.Tensor:
    values = []
    for caption in texts:
        seed = int.from_bytes(hashlib.sha256(caption.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(dim).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        values.append(torch.from_numpy(vector))
    return torch.stack(values)[:, None].to(device)


def _build_batch(
    model,
    batch,
    device: torch.device,
    fixed_noise: bool,
    boundary_loss_weight: float = 0.0,
    root_velocity_loss_weight: float = 0.0,
    constraint_loss_weight: float = 0.0,
    fk_constraint_loss_weight: float = 0.0,
    foot_skate_loss_weight: float = 0.0,
    sequential_cache: bool = False,
    sequential_rollout_windows: int = 2,
    rollout_total_frames: int | None = None,
):
    from ardy.model.ardy_model import get_three_mask_from_len

    motion = batch["motion"].to(device)
    observed = batch["observed_motion"].to(device)
    motion_mask = batch["motion_mask"].to(device)
    if rollout_total_frames is not None:
        if rollout_total_frames <= 0 or rollout_total_frames > motion.shape[1]:
            raise ValueError("rollout_total_frames must fit inside the sampled motion")
        motion = motion[:, :rollout_total_frames]
        observed = observed[:, :rollout_total_frames]
        motion_mask = motion_mask[:, :rollout_total_frames]
    physical_motion = model.motion_rep.unnormalize(motion)
    rollout_motion = motion
    rollout_observed = observed
    rollout_motion_mask = motion_mask
    rollout_physical_motion = physical_motion
    batch_size, frames = motion.shape[:2]
    horizon = int(model.gen_horizon_len)
    if frames % model.hybrid.num_frames_per_token:
        raise ValueError("frame count must be divisible by the tokenizer patch size")
    lengths = torch.full((batch_size,), frames, dtype=torch.long, device=device)
    pad = torch.ones((batch_size, frames), dtype=torch.bool, device=device)
    with torch.no_grad():
        clean, _ = model.hybrid.get_hybrid_motion_from_explicit(motion, lengths, pad)
    original_history_len = batch.get("history_frames", torch.zeros(batch_size, dtype=torch.long)).to(device)
    if sequential_cache:
        if sequential_rollout_windows < 2:
            raise ValueError("sequential_rollout_windows must be at least 2")
        expected_history_frames = (sequential_rollout_windows - 1) * horizon
        if (
            not bool((original_history_len == expected_history_frames).all())
            or frames < sequential_rollout_windows * horizon
        ):
            raise ValueError(
                "cache-only sequential batches require a uniform "
                f"{expected_history_frames}-frame teacher history and "
                f"{sequential_rollout_windows * horizon} total frames"
            )
        history_chunks = []
        chunk_tokens = horizon // model.hybrid.num_frames_per_token
        center_index = torch.full(
            (batch_size,), horizon - 1, dtype=torch.long, device=device
        )
        for window_index in range(sequential_rollout_windows - 1):
            token_start = window_index * chunk_tokens
            token_end = token_start + chunk_tokens
            chunk = clean[:, token_start:token_end].detach()
            recentered, _, _ = model._recenter_history(
                chunk, center_index, requantize=False
            )
            history_chunks.append(recentered.detach())
        history_hybrid = torch.stack(history_chunks, dim=1)
        heading_indices = torch.arange(
            horizon - 1,
            expected_history_frames,
            horizon,
            device=device,
        )
        history_headings = torch.atan2(
            physical_motion[:, heading_indices, 4],
            physical_motion[:, heading_indices, 3],
        )
        current_start = expected_history_frames
        current_end = current_start + horizon
        motion = motion[:, current_start:current_end]
        observed = observed[:, current_start:current_end]
        motion_mask = motion_mask[:, current_start:current_end]
        clean = clean[:, current_start // model.hybrid.num_frames_per_token : current_end // model.hybrid.num_frames_per_token]
        frames = motion.shape[1]
        history_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        generation_len = torch.full((batch_size,), frames, dtype=torch.long, device=device)
        future_len = torch.zeros_like(history_len)
        lengths = torch.full((batch_size,), frames, dtype=torch.long, device=device)
        pad = torch.ones((batch_size, frames), dtype=torch.bool, device=device)
    else:
        history_len = original_history_len
        generation_len = torch.full((batch_size,), horizon, dtype=torch.long, device=device)
        future_len = frames - history_len - generation_len
    if bool((future_len < 0).any()):
        raise ValueError("history + generation exceeds the training window")
    history_mask, generation_mask, future_mask = get_three_mask_from_len(
        history_len, generation_len, future_len, frames, device
    )
    history_token_mask, generation_token_mask, future_token_mask = model.hybrid.convert_frame_mask_to_token_mask(
        history_mask, generation_mask, future_mask, motion_mask
    )
    if "text_feature" in batch:
        text = batch["text_feature"].to(device)[:, None]
    else:
        text = _hashed_text(list(batch["text"]), model.denoiser.model.llm_shape[-1], device)
    text_dropped = batch.get(
        "text_dropped", torch.zeros(batch_size, dtype=torch.bool)
    ).to(device)
    text_mask = (~text_dropped)[:, None].expand(text.shape[:2])
    # Heading is an angle in the physical [cos, sin] representation; compute
    # it before normalization rather than applying atan2 to z-scored values.
    heading_index = int(original_history_len[0]) - 1 if sequential_cache else 0
    heading = torch.atan2(physical_motion[:, heading_index, 4], physical_motion[:, heading_index, 3])
    with torch.no_grad():
        target_geometry = model.motion_rep.inverse(motion, is_normalized=True)
    if fixed_noise:
        timestep = torch.full((batch_size,), model.diffusion.num_base_steps // 2, dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(901)
        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
    else:
        timestep = torch.randint(0, model.diffusion.num_base_steps, (batch_size,), device=device)
        noise = torch.randn_like(clean)
    noisy = model.diffusion.q_sample(clean, timestep, noise)
    x = torch.zeros_like(clean)
    x[history_token_mask] = clean[history_token_mask]
    x[generation_token_mask] = noisy[generation_token_mask]
    return {
        "x": x, "target": clean,
        "history_len": history_len, "generation_len": generation_len, "future_len": future_len,
        "history_mask": history_mask, "generation_mask": generation_mask, "future_mask": future_mask,
        "history_token_mask": history_token_mask, "generation_token_mask": generation_token_mask,
        "future_token_mask": future_token_mask, "text": text, "text_mask": text_mask,
        "timestep": timestep, "heading": heading, "motion_mask": motion_mask,
        "observed": observed, "boundary_loss_weight": boundary_loss_weight,
        "root_velocity_loss_weight": root_velocity_loss_weight,
        "constraint_loss_weight": constraint_loss_weight,
        "fk_constraint_loss_weight": fk_constraint_loss_weight,
        "foot_skate_loss_weight": foot_skate_loss_weight,
        "sequential_cache": sequential_cache,
        "history_hybrid": history_hybrid if sequential_cache else None,
        "history_headings": history_headings if sequential_cache else None,
        "sequential_rollout_windows": sequential_rollout_windows if sequential_cache else 1,
        "motion": motion,
        "lengths": lengths,
        "pad_mask": pad,
        "hybrid_converter": model.hybrid,
        "target_joints": target_geometry["posed_joints"],
        "target_foot_contacts": target_geometry["foot_contacts"],
        "joint_position_mask": motion_mask[
            :, :, model.motion_rep.slice_dict["local_joints_positions"]
        ].reshape(batch_size, frames, model.skeleton.nbjoints - 1, 3).any(dim=-1)
        & generation_mask[:, :, None],
        "foot_indices": list(model.skeleton.foot_joint_idx),
        "next_generation_frames": horizon,
        "noise": noise,
        "rollout_motion": rollout_motion,
        "rollout_observed": rollout_observed,
        "rollout_motion_mask": rollout_motion_mask,
        "rollout_physical_motion": rollout_physical_motion,
    }


def _unwrap_block(block):
    return getattr(block, "module", block)


def _unwrap_raw(raw):
    return getattr(raw, "module", raw)


def _denoiser_aux_state(raw) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in _unwrap_raw(raw).state_dict().items()
        if not key.startswith(("root_model.", "body_model."))
    }


def _load_expanded_state(module, state: dict[str, torch.Tensor]) -> None:
    """Load an FFN-expanded checkpoint while preserving the old function."""
    current = module.state_dict()
    unknown = set(state) - set(current)
    if unknown:
        raise ValueError(f"checkpoint has unknown keys: {sorted(unknown)}")
    expanded = dict(current)
    for key, source in state.items():
        target = current[key]
        if source.shape == target.shape:
            expanded[key] = source
            continue
        if ".mlp.0." not in key and ".mlp.3.weight" not in key:
            raise ValueError(
                f"unsupported checkpoint expansion for {key}: "
                f"{tuple(source.shape)} -> {tuple(target.shape)}"
            )
        if source.ndim != target.ndim or any(
            old > new for old, new in zip(source.shape, target.shape)
        ):
            raise ValueError(f"checkpoint tensor cannot expand into {key}")
        value = target.clone()
        slices = tuple(slice(0, size) for size in source.shape)
        if ".mlp.3.weight" in key:
            value.zero_()
        value[slices] = source
        expanded[key] = value
    module.load_state_dict(expanded, strict=True)


def _load_denoiser_weights(raw, checkpoint: dict) -> None:
    model = _unwrap_raw(raw)
    _load_expanded_state(_unwrap_block(model.root_model), checkpoint["root_model"])
    _load_expanded_state(_unwrap_block(model.body_model), checkpoint["body_model"])
    auxiliary = checkpoint.get("denoiser_aux")
    if auxiliary is not None:
        current = model.state_dict()
        unknown = set(auxiliary) - set(current)
        if unknown:
            raise ValueError(f"checkpoint has unknown denoiser_aux keys: {sorted(unknown)}")
        with torch.no_grad():
            for key, value in auxiliary.items():
                current[key].copy_(value.to(device=current[key].device, dtype=current[key].dtype))


def _prime_teacher_history(raw, fixed, client=None) -> None:
    """Commit teacher history chunks through the canonical history protocol."""
    model = _unwrap_raw(raw)
    histories = fixed["history_hybrid"]
    headings = fixed["history_headings"]
    if histories.ndim != 4 or headings.shape != histories.shape[:2]:
        raise ValueError("multiwindow teacher history must be [B,W,T,D] with [B,W] headings")
    root_training = model.root_model.training
    body_training = model.body_model.training
    model.root_model.eval()
    model.body_model.eval()
    try:
        for chunk_id in range(histories.shape[1]):
            if (
                fixed.get("training_state_intervention") == "fresh_each_window"
                and client is not None
                and getattr(client, "controller", None) is not None
            ):
                client.controller.reset_mamba_only()
            canonical_history_prime(
                model,
                histories[:, chunk_id],
                text_feat=fixed["text"],
                text_pad_mask=fixed["text_mask"],
                first_heading_angle=headings[:, chunk_id],
                next_generation_frames=fixed["next_generation_frames"],
                chunk_id=chunk_id,
            )
        if (
            fixed.get("training_state_intervention") == "fresh_each_window"
            and client is not None
            and getattr(client, "controller", None) is not None
        ):
            client.controller.reset_mamba_only()
    finally:
        model.root_model.train(root_training)
        model.body_model.train(body_training)


def select_generated_history_supervision_depths(
    generated_windows: int,
    max_points: int,
) -> list[int]:
    """Select generated-prefix depths covering early, window five, and final."""
    if generated_windows <= 0:
        raise ValueError("generated_windows must be positive")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if generated_windows <= max_points:
        return list(range(1, generated_windows + 1))
    if max_points == 1:
        return [generated_windows]
    if max_points == 2:
        depths = {min(4, generated_windows), generated_windows}
        if len(depths) < 2:
            depths.add(1)
        return sorted(depths)

    depths = {1, min(4, generated_windows), generated_windows}
    # More than three points is supported for ablations. Fill remaining slots
    # uniformly without changing the three scientifically meaningful anchors.
    for index in range(1, max_points + 1):
        if len(depths) >= max_points:
            break
        candidate = 1 + round(
            index * (generated_windows - 1) / max(max_points - 1, 1)
        )
        depths.add(min(candidate, generated_windows))
    for candidate in range(1, generated_windows + 1):
        if len(depths) >= max_points:
            break
        depths.add(candidate)
    return sorted(depths)[:max_points]


def _generated_supervision_batch(
    model,
    fixed: dict,
    *,
    mode: str,
    depth: int,
    generated_full: torch.Tensor,
    physical: torch.Tensor,
    full_mask: torch.Tensor,
    seed_physical: torch.Tensor,
    seed_hybrid,
    seed_heading: torch.Tensor,
    replay_history_chunks,
    state_snapshot,
    bptt_start_depth: int,
    bptt_start_snapshot,
    teacher_state_snapshot,
    max_recovery_meters: float,
) -> dict:
    """Build the next-window target for one accepted generated prefix."""
    from ardy.model.ardy_model import get_three_mask_from_len

    horizon = int(model.gen_horizon_len)
    batch_size = fixed["text"].shape[0]
    device = fixed["text"].device
    prefix_frames = (depth + 1) * horizon
    target_end = prefix_frames + horizon
    if generated_full.shape[1] < prefix_frames or physical.shape[1] < target_end:
        raise ValueError("supervision depth exceeds generated/source rollout")

    generated_prefix = generated_full[:, :prefix_frames]
    generated_physical = model.motion_rep.unnormalize(generated_prefix)
    generated_endpoint = generated_physical[:, -1, [0, 2]].clone()
    teacher_endpoint = physical[:, prefix_frames - 1, [0, 2]].clone()
    recovery = teacher_endpoint - generated_endpoint
    raw_recovery_norm = torch.linalg.vector_norm(recovery, dim=-1)
    if max_recovery_meters > 0:
        scale = torch.clamp(
            max_recovery_meters / raw_recovery_norm.clamp_min(1e-8), max=1.0
        )
        recovery = recovery * scale[:, None]
    applied_recovery_norm = torch.linalg.vector_norm(recovery, dim=-1)

    target_physical = physical[:, prefix_frames:target_end].clone()
    target_physical[:, :, 0] -= teacher_endpoint[:, None, 0]
    target_physical[:, :, 2] -= teacher_endpoint[:, None, 1]
    target_physical[:, :, 0] += recovery[:, None, 0]
    target_physical[:, :, 2] += recovery[:, None, 1]
    target_motion = model.motion_rep.normalize(target_physical)
    target_mask = full_mask[:, prefix_frames:target_end]

    if mode == "original":
        generated_local = generated_physical.clone()
        generated_local[:, :, 0] -= generated_endpoint[:, None, 0]
        generated_local[:, :, 2] -= generated_endpoint[:, None, 1]
        explicit_motion = torch.cat(
            (model.motion_rep.normalize(generated_local), target_motion), dim=1
        )
        motion_mask = torch.cat((full_mask[:, :prefix_frames], target_mask), dim=1)
        history_len = torch.full(
            (batch_size,), prefix_frames, dtype=torch.long, device=device
        )
        heading = torch.atan2(seed_physical[:, 0, 4], seed_physical[:, 0, 3])
    else:
        explicit_motion = target_motion
        motion_mask = target_mask
        history_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        heading = torch.atan2(generated_physical[:, -1, 4], generated_physical[:, -1, 3])

    frames = explicit_motion.shape[1]
    generation_len = torch.full(
        (batch_size,), horizon, dtype=torch.long, device=device
    )
    future_len = torch.zeros_like(history_len)
    history_mask, generation_mask, future_mask = get_three_mask_from_len(
        history_len, generation_len, future_len, frames, device
    )
    history_token_mask, generation_token_mask, future_token_mask = (
        model.hybrid.convert_frame_mask_to_token_mask(
            history_mask, generation_mask, future_mask, motion_mask
        )
    )
    lengths = torch.full((batch_size,), frames, dtype=torch.long, device=device)
    pad = torch.ones(batch_size, frames, dtype=torch.bool, device=device)
    clean, _ = model.hybrid.get_hybrid_motion_from_explicit(
        explicit_motion, lengths, pad
    )
    noise = fixed["noise"][:, : clean.shape[1]]
    if noise.shape != clean.shape:
        raise RuntimeError(
            f"generated target shape changed: noise={tuple(noise.shape)} "
            f"clean={tuple(clean.shape)}"
        )
    noisy = model.diffusion.q_sample(clean, fixed["timestep"], noise)
    x = torch.zeros_like(clean)
    x[history_token_mask] = clean[history_token_mask]
    x[generation_token_mask] = noisy[generation_token_mask]
    observed_motion = explicit_motion * motion_mask.to(explicit_motion.dtype)
    target_geometry = model.motion_rep.inverse(explicit_motion, is_normalized=True)

    supervision = dict(fixed)
    supervision.update(
        {
            "x": x,
            "target": clean,
            "history_len": history_len,
            "generation_len": generation_len,
            "future_len": future_len,
            "history_mask": history_mask,
            "generation_mask": generation_mask,
            "future_mask": future_mask,
            "history_token_mask": history_token_mask,
            "generation_token_mask": generation_token_mask,
            "future_token_mask": future_token_mask,
            "heading": heading,
            "motion": explicit_motion,
            "motion_mask": motion_mask.to(dtype=fixed["motion_mask"].dtype),
            "observed": observed_motion,
            "lengths": lengths,
            "pad_mask": pad,
            "target_joints": target_geometry["posed_joints"],
            "target_foot_contacts": target_geometry["foot_contacts"],
            "joint_position_mask": motion_mask[
                :, :, model.motion_rep.slice_dict["local_joints_positions"]
            ].reshape(batch_size, frames, model.skeleton.nbjoints - 1, 3).any(dim=-1)
            & generation_mask[:, :, None],
            "generated_history_state_ready": mode != "original",
            "generated_history_state_snapshot": state_snapshot,
            "generated_history_bptt_start_depth": bptt_start_depth,
            "generated_history_bptt_start_snapshot": bptt_start_snapshot,
            "generated_history_teacher_state_snapshot": teacher_state_snapshot,
            "generated_history_seed_hybrid": seed_hybrid,
            "generated_history_seed_heading": seed_heading,
            "generated_history_replay_chunks": replay_history_chunks,
            "generated_history_supervision_depth": depth,
            "generated_history_endpoint_xz": generated_endpoint,
            "generated_history_raw_recovery_norm": raw_recovery_norm,
            "generated_history_applied_recovery_norm": applied_recovery_norm,
        }
    )
    supervision.pop("generated_history_supervision_batches", None)
    return supervision


@torch.no_grad()
def _prepare_generated_history(
    model,
    raw,
    client,
    fixed,
    *,
    mode: str,
    diffusion_steps: int,
    cfg_weight_text: float,
    cfg_weight_constraint: float,
    max_recovery_meters: float = 0.0,
    max_supervision_points: int = 1,
    bptt_windows: int = 0,
    attach_seed_write: bool = False,
    state_consistency_weight: float = 0.0,
    root_endpoint_loss_weight: float = 0.0,
) -> dict:
    """Create next-window targets at selected depths of one on-policy rollout."""
    rollout_windows = int(fixed["sequential_rollout_windows"])
    if rollout_windows < 3:
        raise ValueError("generated history requires a seed, a generated window, and a target")
    if diffusion_steps <= 0:
        raise ValueError("generated-history diffusion steps must be positive")
    if max_recovery_meters < 0:
        raise ValueError("generated-history recovery cap must be non-negative")
    if bptt_windows < 0:
        raise ValueError("generated-history BPTT windows must be non-negative")
    if state_consistency_weight < 0:
        raise ValueError("generated-history state consistency weight must be non-negative")
    if state_consistency_weight and not bptt_windows:
        raise ValueError("state consistency requires generated-history BPTT")
    if root_endpoint_loss_weight < 0:
        raise ValueError("generated-history root endpoint loss weight must be non-negative")
    supervision_depths = select_generated_history_supervision_depths(
        rollout_windows - 2, max_supervision_points
    )
    bptt_start_depths = {
        depth: max(0, depth - bptt_windows) for depth in supervision_depths
    }
    snapshot_depths = sorted(
        set(supervision_depths)
        | {depth for depth in bptt_start_depths.values() if depth > 0}
    )
    horizon = int(model.gen_horizon_len)
    prefix_frames = (rollout_windows - 1) * horizon
    batch_size = fixed["text"].shape[0]
    device = fixed["text"].device
    physical = fixed["rollout_physical_motion"].clone()
    if physical.shape[1] < rollout_windows * horizon:
        raise ValueError("generated-history source is shorter than the rollout")

    # Anchor the rollout world frame at the end of the GT seed window. This is
    # the same initial-history convention used by official Table 5.
    seed_center = physical[:, horizon - 1, [0, 2]].clone()
    physical[:, :, 0] -= seed_center[:, None, 0]
    physical[:, :, 2] -= seed_center[:, None, 1]
    normalized = model.motion_rep.normalize(physical)
    full_mask = fixed["rollout_motion_mask"].to(dtype=torch.bool)
    observed = normalized * full_mask.to(normalized.dtype)
    seed_motion = normalized[:, :horizon]
    seed_physical = physical[:, :horizon]
    seed_heading = torch.atan2(seed_physical[:, -1, 4], seed_physical[:, -1, 3])
    texts = [""] * batch_size

    denoiser = _unwrap_raw(raw)
    root_training = denoiser.root_model.training
    body_training = denoiser.body_model.training
    denoiser.root_model.eval()
    denoiser.body_model.eval()
    seed_hybrid = None
    history_chunks = []
    teacher_state_snapshots = {}
    try:
        if mode == "original":
            lengths = torch.full(
                (batch_size,), prefix_frames, dtype=torch.long, device=device
            )
            generated_full = model(
                texts,
                prefix_frames,
                num_denoising_steps=diffusion_steps,
                pad_mask=torch.ones(
                    batch_size, prefix_frames, dtype=torch.bool, device=device
                ),
                first_heading_angle=None,
                motion_mask=full_mask[:, :prefix_frames],
                observed_motion=observed[:, :prefix_frames],
                cfg_weight=(cfg_weight_text, cfg_weight_constraint),
                crop_history_length=None,
                init_history_sequence=seed_motion,
                text_feat=fixed["text"],
                text_pad_mask=fixed["text_mask"],
                progress_bar=lambda values: values,
            )
        else:
            if not hasattr(client, "controller") or client.controller is None:
                raise RuntimeError("persistent generated history requires a memory controller")
            client.reset_memory()
            seed_lengths = torch.full(
                (batch_size,), horizon, dtype=torch.long, device=device
            )
            seed_hybrid, _ = model.hybrid.get_hybrid_motion_from_explicit(
                seed_motion,
                seed_lengths,
                torch.ones(batch_size, horizon, dtype=torch.bool, device=device),
            )
            controller = client.controller
            controller.prime_accepted_history(
                seed_hybrid,
                text_feat=fixed["text"],
                text_pad_mask=fixed["text_mask"],
                first_heading_angle=seed_heading,
            )
            generated_frames = prefix_frames - horizon
            controller.begin_training_snapshot_capture(snapshot_depths)
            try:
                generated_tail = model(
                    texts,
                    generated_frames,
                    num_denoising_steps=diffusion_steps,
                    pad_mask=torch.ones(
                        batch_size, generated_frames, dtype=torch.bool, device=device
                    ),
                    first_heading_angle=seed_heading,
                    motion_mask=full_mask[:, horizon:prefix_frames],
                    observed_motion=observed[:, horizon:prefix_frames],
                    cfg_weight=(cfg_weight_text, cfg_weight_constraint),
                    crop_history_length=0,
                    text_feat=fixed["text"],
                    text_pad_mask=fixed["text_mask"],
                    progress_bar=lambda values: values,
                )
            finally:
                state_snapshots, history_chunks = (
                    controller.end_training_snapshot_capture()
                )
            generated_full = torch.cat((seed_motion, generated_tail), dim=1)
            expected_commits = rollout_windows - 1
            if client.controller.trace.accepted_commits != expected_commits:
                raise RuntimeError(
                    "generated-history rollout did not commit the seed and every accepted window"
                )
            if set(state_snapshots) != set(snapshot_depths):
                raise RuntimeError(
                    "generated-history rollout did not capture every requested depth: "
                    f"requested={snapshot_depths} captured={sorted(state_snapshots)}"
                )
            if len(history_chunks) != rollout_windows - 2:
                raise RuntimeError(
                    "generated-history rollout did not capture every accepted history chunk"
                )
            if state_consistency_weight:
                client.reset_memory()
                teacher_histories = fixed["history_hybrid"]
                teacher_headings = fixed["history_headings"]
                max_depth = max(supervision_depths)
                for chunk_id in range(max_depth + 1):
                    client.controller.set_runtime_mamba_output_scale_for_depth(
                        max(0, chunk_id - 1)
                    )
                    canonical_history_prime(
                        denoiser,
                        teacher_histories[:, chunk_id],
                        text_feat=fixed["text"],
                        text_pad_mask=fixed["text_mask"],
                        first_heading_angle=teacher_headings[:, chunk_id],
                        next_generation_frames=fixed["next_generation_frames"],
                        chunk_id=chunk_id,
                        detach=True,
                    )
                    if chunk_id in supervision_depths:
                        teacher_state_snapshots[chunk_id] = (
                            denoiser.root_model._sequence_state.clone(),
                            denoiser.body_model._sequence_state.clone(),
                        )
    finally:
        denoiser.root_model.train(root_training)
        denoiser.body_model.train(body_training)

    # Sampling switches the diffusion object to a spaced schedule. Restore the
    # base schedule before constructing the supervised noisy target.
    model.diffusion.calc_diffusion_vars(
        torch.arange(model.diffusion.num_base_steps, device=device)
    )
    state_snapshots = {} if mode == "original" else state_snapshots
    supervision_batches = [
        _generated_supervision_batch(
            model,
            fixed,
            mode=mode,
            depth=depth,
            generated_full=generated_full,
            physical=physical,
            full_mask=full_mask,
            seed_physical=seed_physical,
            seed_hybrid=seed_hybrid,
            seed_heading=seed_heading,
            replay_history_chunks=history_chunks[:depth],
            state_snapshot=state_snapshots.get(depth),
            bptt_start_depth=bptt_start_depths[depth],
            bptt_start_snapshot=state_snapshots.get(bptt_start_depths[depth]),
            teacher_state_snapshot=teacher_state_snapshots.get(depth),
            max_recovery_meters=max_recovery_meters,
        )
        for depth in supervision_depths
    ]
    fixed.update(supervision_batches[-1])
    fixed["generated_history_supervision_batches"] = supervision_batches
    fixed["generated_history_supervision_depths"] = supervision_depths
    for supervision in supervision_batches:
        supervision["generated_history_bptt_windows"] = bptt_windows
    fixed["generated_history_bptt_windows"] = bptt_windows
    for supervision in supervision_batches:
        supervision["generated_history_attach_seed_write"] = attach_seed_write
    fixed["generated_history_attach_seed_write"] = attach_seed_write
    for supervision in supervision_batches:
        supervision["generated_history_state_consistency_weight"] = (
            state_consistency_weight
        )
    fixed["generated_history_state_consistency_weight"] = state_consistency_weight
    for supervision in supervision_batches:
        supervision["generated_history_root_endpoint_loss_weight"] = (
            root_endpoint_loss_weight
        )
    fixed["generated_history_root_endpoint_loss_weight"] = root_endpoint_loss_weight
    return fixed


def _prime_generated_history_bptt(raw, client, fixed) -> None:
    """Replay accepted chunks and retain gradients through the newest writes."""
    bptt_windows = int(fixed.get("generated_history_bptt_windows", 0))
    chunks = fixed.get("generated_history_replay_chunks", [])
    if bptt_windows <= 0 or not chunks:
        raise ValueError("differentiable history replay requires chunks and a positive depth")
    seed_hybrid = fixed.get("generated_history_seed_hybrid")
    seed_heading = fixed.get("generated_history_seed_heading")
    if seed_hybrid is None or seed_heading is None:
        raise RuntimeError("differentiable history replay is missing the seed")
    if not hasattr(client, "controller") or client.controller is None:
        raise RuntimeError("differentiable history replay requires a memory controller")

    model = _unwrap_raw(raw)
    root_training = model.root_model.training
    body_training = model.body_model.training
    model.root_model.eval()
    model.body_model.eval()
    start_depth = int(fixed.get("generated_history_bptt_start_depth", 0))
    start_snapshot = fixed.get("generated_history_bptt_start_snapshot")
    if start_depth < 0 or start_depth > len(chunks):
        raise RuntimeError("invalid differentiable history replay start depth")
    client.reset_memory()
    try:
        if start_depth == 0:
            client.controller.set_runtime_mamba_output_scale_for_depth(0)
            canonical_history_prime(
                model,
                seed_hybrid,
                text_feat=fixed["text"],
                text_pad_mask=fixed["text_mask"],
                first_heading_angle=seed_heading,
                next_generation_frames=fixed["next_generation_frames"],
                chunk_id=0,
                detach=not bool(
                    fixed.get("generated_history_attach_seed_write", False)
                ),
            )
        else:
            if start_snapshot is None:
                raise RuntimeError("truncated BPTT start snapshot is missing")
            client.controller.restore_training_snapshot(start_snapshot)
        for index in range(start_depth, len(chunks)):
            history_hybrid, heading = chunks[index]
            client.controller.set_runtime_mamba_output_scale_for_depth(index)
            canonical_history_prime(
                model,
                history_hybrid,
                text_feat=fixed["text"],
                text_pad_mask=fixed["text_mask"],
                first_heading_angle=heading,
                next_generation_frames=fixed["next_generation_frames"],
                chunk_id=index + 1,
                detach=False,
            )
        client.controller.set_runtime_mamba_output_scale_for_depth(len(chunks))
    finally:
        model.root_model.train(root_training)
        model.body_model.train(body_training)


def _rollout_raw_forward(raw, x: torch.Tensor, args: tuple, checkpoint: bool):
    if not checkpoint:
        return raw(x, *args)
    # Native explicit-history ARDY is functional across this call. Persistent
    # state is intentionally not checkpointed because recomputation after a
    # later state commit would read a different state snapshot.
    return torch.utils.checkpoint.checkpoint(
        lambda value: raw(value, *args), x, use_reentrant=False
    )


def _differentiable_rollout_surrogate(
    model,
    raw,
    client,
    fixed: dict,
    *,
    mode: str,
    windows: int,
    diffusion_steps: int,
    warm_prefix_windows: int,
    slope_weight: float,
    auc_weight: float,
    final_weight: float,
    margin_cm_per_window: float,
    gradient_checkpointing: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable deterministic rollout with production recenter semantics."""
    from ardy.model.ardy_model import get_three_mask_from_len

    if not 5 <= windows <= 8:
        raise ValueError("differentiable rollout requires 5-8 total windows")
    if diffusion_steps not in {1, 2}:
        raise ValueError("differentiable rollout diffusion steps must be 1 or 2")
    if gradient_checkpointing and mode != "original":
        raise ValueError(
            "gradient checkpointing is not state-safe for persistent rollout; "
            "use the bf16/low-probability path"
        )
    physical = fixed["rollout_physical_motion"]
    full_mask = fixed["rollout_motion_mask"].to(dtype=torch.bool)
    horizon = int(model.gen_horizon_len)
    required_frames = windows * horizon
    if physical.shape[1] < required_frames:
        raise ValueError(
            f"differentiable rollout needs {required_frames} frames, got {physical.shape[1]}"
        )
    physical = physical[:, :required_frames]
    full_mask = full_mask[:, :required_frames]
    batch = physical.shape[0]
    device = physical.device
    nfpt = int(model.hybrid.num_frames_per_token)
    tokens = horizon // nfpt
    raw_model = _unwrap_raw(raw)
    client.reset_memory() if hasattr(client, "reset_memory") else None

    seed_physical = physical[:, :horizon].clone()
    seed_origin = seed_physical[:, -1, [0, 2]].clone()
    seed_local = seed_physical.clone()
    seed_local[:, :, 0] -= seed_origin[:, None, 0]
    seed_local[:, :, 2] -= seed_origin[:, None, 1]
    seed_motion = model.motion_rep.normalize(seed_local)
    lengths = torch.full((batch,), horizon, dtype=torch.long, device=device)
    pad = torch.ones(batch, horizon, dtype=torch.bool, device=device)
    seed_hybrid, _ = model.hybrid.get_hybrid_motion_from_explicit(
        seed_motion, lengths, pad
    )
    center_index = torch.full(
        (batch,), horizon - 1, dtype=torch.long, device=device
    )
    seed_hybrid, _, seed_root = model._recenter_history(
        seed_hybrid, center_index, requantize=False
    )
    heading = model.motion_rep.get_root_heading_angle(seed_root)[:, -1]
    if mode == "original":
        explicit_history = seed_hybrid
    else:
        canonical_history_prime(
            raw_model,
            seed_hybrid,
            text_feat=fixed["text"],
            text_pad_mask=fixed["text_mask"],
            first_heading_angle=heading,
            next_generation_frames=horizon,
            chunk_id=0,
            detach=warm_prefix_windows > 0,
        )
        if fixed.get("training_state_intervention") == "fresh_each_window":
            client.controller.reset_mamba_only()
        explicit_history = None

    predicted_world_origin = seed_origin
    root_errors = []
    constraint_errors = []
    endpoint_tensors = []
    gradient_endpoints = []
    for window in range(1, windows):
        if mode == "hybrid":
            client.controller.set_runtime_mamba_output_scale_for_depth(window - 1)
        start = window * horizon
        end = start + horizon
        target_world = physical[:, start:end]
        target_origin = physical[:, start - 1, [0, 2]]
        target_local = target_world.clone()
        target_local[:, :, 0] -= target_origin[:, None, 0]
        target_local[:, :, 2] -= target_origin[:, None, 1]
        target_motion = model.motion_rep.normalize(target_local)
        target_hybrid, _ = model.hybrid.get_hybrid_motion_from_explicit(
            target_motion, lengths, pad
        )
        if mode == "original":
            history_tokens = explicit_history.shape[1]
            clean = torch.cat((explicit_history, target_hybrid), dim=1)
            history_frames = history_tokens * nfpt
            total_frames = history_frames + horizon
            history_len = torch.full(
                (batch,), history_frames, dtype=torch.long, device=device
            )
            generation_len = torch.full(
                (batch,), horizon, dtype=torch.long, device=device
            )
            future_len = torch.zeros_like(history_len)
            motion_mask = torch.cat(
                (
                    torch.zeros(
                        batch, history_frames, full_mask.shape[-1],
                        dtype=torch.bool, device=device,
                    ),
                    full_mask[:, start:end],
                ),
                dim=1,
            )
            observed = torch.zeros(
                batch, total_frames, target_motion.shape[-1],
                dtype=target_motion.dtype, device=device,
            )
            observed[:, history_frames:][full_mask[:, start:end]] = target_motion[
                full_mask[:, start:end]
            ]
        else:
            clean = target_hybrid
            total_frames = horizon
            history_len = torch.zeros(batch, dtype=torch.long, device=device)
            generation_len = torch.full_like(history_len, horizon)
            future_len = torch.zeros_like(history_len)
            motion_mask = full_mask[:, start:end]
            observed = torch.zeros_like(target_motion)
            observed[motion_mask] = target_motion[motion_mask]
        history_mask, generation_mask, future_mask = get_three_mask_from_len(
            history_len, generation_len, future_len, total_frames, device
        )
        history_token_mask, generation_token_mask, future_token_mask = (
            model.hybrid.convert_frame_mask_to_token_mask(
                history_mask, generation_mask, future_mask, motion_mask
            )
        )
        step_tensors = [model.diffusion.num_base_steps // 2]
        if diffusion_steps == 2:
            step_tensors.append(0)
        noise = torch.randn_like(clean)
        x = model.diffusion.q_sample(
            clean,
            torch.full((batch,), step_tensors[0], dtype=torch.long, device=device),
            noise,
        )
        x = torch.where(history_token_mask[:, :, None], clean, x)
        args = (
            history_len, generation_len, future_len,
            history_mask, generation_mask, future_mask,
            history_token_mask, generation_token_mask, future_token_mask,
            fixed["text"], fixed["text_mask"],
            None, heading, motion_mask.to(dtype=target_motion.dtype), observed,
        )
        prediction = None
        for denoise_index, timestep in enumerate(step_tensors):
            call_args = list(args)
            call_args[11] = torch.full(
                (batch,), timestep, dtype=torch.long, device=device
            )
            prediction = _rollout_raw_forward(
                raw,
                x,
                tuple(call_args),
                gradient_checkpointing,
            )
            if denoise_index + 1 < len(step_tensors):
                x = torch.where(
                    history_token_mask[:, :, None], clean, prediction
                )
        accepted = prediction[:, -tokens:]
        accepted_motion = model.hybrid.get_explicit_motion_from_hybrid(
            accepted, pad, lengths
        )
        accepted_physical = model.motion_rep.unnormalize(accepted_motion)
        predicted_world_root = accepted_physical[:, :, [0, 2]] + predicted_world_origin[:, None]
        target_world_root = target_world[:, :, [0, 2]]
        root_error = torch.linalg.vector_norm(
            predicted_world_root - target_world_root, dim=-1
        ).mean(dim=1) * 100.0
        generated_geometry = model.motion_rep.inverse(
            accepted_motion, is_normalized=True
        )
        target_geometry = model.motion_rep.inverse(
            target_motion, is_normalized=True
        )
        joint_error = torch.linalg.vector_norm(
            generated_geometry["posed_joints"] - target_geometry["posed_joints"],
            dim=-1,
        ).mean(dim=(1, 2)) * 100.0
        root_errors.append(root_error)
        constraint_errors.append(joint_error)
        predicted_world_origin = predicted_world_root[:, -1]
        endpoint_tensors.append(predicted_world_origin)
        warm_prefix = window <= warm_prefix_windows
        if warm_prefix:
            root_errors[-1] = root_errors[-1].detach()
            constraint_errors[-1] = constraint_errors[-1].detach()
            predicted_world_origin = predicted_world_origin.detach()
            endpoint_tensors[-1] = endpoint_tensors[-1].detach()
            accepted = accepted.detach()
        else:
            predicted_world_origin.retain_grad()
            gradient_endpoints.append(predicted_world_origin)

        if mode == "original":
            explicit_history = torch.cat((explicit_history, accepted), dim=1)
            full_center = torch.full(
                (batch,), explicit_history.shape[1] * nfpt - 1,
                dtype=torch.long, device=device,
            )
            explicit_history, _, recentered_root = model._recenter_history(
                explicit_history, full_center, requantize=False
            )
            heading = model.motion_rep.get_root_heading_angle(recentered_root)[:, -1]
        else:
            recentered, _, recentered_root = model._recenter_history(
                accepted, center_index, requantize=False
            )
            heading = model.motion_rep.get_root_heading_angle(recentered_root)[:, -1]
            detach = warm_prefix
            canonical_history_prime(
                raw_model,
                recentered,
                text_feat=fixed["text"],
                text_pad_mask=fixed["text_mask"],
                first_heading_angle=heading,
                next_generation_frames=horizon,
                chunk_id=window,
                detach=detach,
            )
            if fixed.get("training_state_intervention") == "fresh_each_window":
                client.controller.reset_mamba_only()

    # Include a zero-valued seed slot so start_window=5 means window six in
    # one-based reporting and the tensor always covers the registered region.
    zero = torch.zeros_like(root_errors[0])
    root_matrix = torch.stack((zero, *root_errors), dim=1)
    constraint_matrix = torch.stack((zero, *constraint_errors), dim=1)
    root_loss, root_parts = differentiable_rollout_loss(
        root_matrix,
        slope_weight=slope_weight,
        auc_weight=auc_weight,
        final_weight=final_weight,
        margin_cm_per_window=margin_cm_per_window,
    )
    constraint_loss, constraint_parts = differentiable_rollout_loss(
        constraint_matrix,
        slope_weight=slope_weight,
        auc_weight=auc_weight,
        final_weight=final_weight,
        margin_cm_per_window=margin_cm_per_window,
    )
    loss = 0.5 * (root_loss + constraint_loss)
    return loss, {
        "root_errors_cm": root_matrix,
        "constraint_errors_cm": constraint_matrix,
        "root_slope_cm_per_window": root_parts["slope_cm_per_window"],
        "constraint_slope_cm_per_window": constraint_parts["slope_cm_per_window"],
        "root_auc_cm_windows": root_parts["auc_cm_windows"],
        "constraint_auc_cm_windows": constraint_parts["auc_cm_windows"],
        "root_final_cm": root_parts["final_error_cm"],
        "constraint_final_cm": constraint_parts["final_error_cm"],
        "endpoints": torch.stack(endpoint_tensors, dim=1),
        "gradient_endpoints": gradient_endpoints,
        "state_graph_depth": torch.tensor(
            windows - max(warm_prefix_windows, 1), device=device
        ),
    }


def _generated_history_state_consistency(client, fixed) -> torch.Tensor:
    weight = float(fixed.get("generated_history_state_consistency_weight", 0.0))
    teacher = fixed.get("generated_history_teacher_state_snapshot")
    if weight <= 0:
        return fixed["x"].new_zeros((), dtype=torch.float32)
    if teacher is None:
        raise RuntimeError("state consistency target is missing")
    losses = []
    for block, teacher_state in zip(
        (client.controller.root_block, client.controller.body_block), teacher
    ):
        current_state = block._sequence_state
        if current_state is None or teacher_state is None:
            raise RuntimeError("state consistency requires materialized sequence states")
        for current_layer, teacher_layer in zip(
            current_state.layers, teacher_state.layers
        ):
            current_tensors = client.controller._state_tensors(
                current_layer.mamba_state
            )
            teacher_tensors = client.controller._state_tensors(
                teacher_layer.mamba_state
            )
            if len(current_tensors) != len(teacher_tensors):
                raise RuntimeError("generated/teacher Mamba state layouts differ")
            for current_tensor, teacher_tensor in zip(
                current_tensors, teacher_tensors
            ):
                target = teacher_tensor.detach().float()
                scale = target.square().mean().clamp_min(1e-6)
                losses.append((current_tensor.float() - target).square().mean() / scale)
    if not losses:
        return fixed["x"].new_zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def _loss(raw, client, fixed, return_components: bool = False):
    # Original ARDY has no adapter-owned persistent cache; reset is a no-op.
    if hasattr(client, "reset_memory") and not fixed.get(
        "generated_history_state_ready", False
    ):
        client.reset_memory()
    if fixed.get("sequential_cache", False) and not fixed.get(
        "generated_history_state_ready", False
    ):
        _prime_teacher_history(raw, fixed, client)
    bptt_windows = int(fixed.get("generated_history_bptt_windows", 0))
    if bptt_windows > 0 and fixed.get("generated_history_state_ready", False):
        _prime_generated_history_bptt(raw, client, fixed)
    snapshot = fixed.get("generated_history_state_snapshot")
    if snapshot is not None and bptt_windows == 0:
        client.controller.restore_training_snapshot(snapshot)
        client.controller.set_runtime_mamba_output_scale_for_depth(
            int(fixed["generated_history_supervision_depth"])
        )
    state_consistency_loss = _generated_history_state_consistency(client, fixed)
    prediction = raw(
        fixed["x"], fixed["history_len"], fixed["generation_len"], fixed["future_len"],
        fixed["history_mask"], fixed["generation_mask"], fixed["future_mask"],
        fixed["history_token_mask"], fixed["generation_token_mask"], fixed["future_token_mask"],
        fixed["text"], fixed["text_mask"], fixed["timestep"], fixed["heading"],
        fixed["motion_mask"], fixed["observed"],
    )
    selection = fixed["generation_token_mask"]
    predicted = prediction[selection].float()
    target = fixed["target"][selection].float()
    model = _unwrap_raw(raw)
    root_dim = model.nframe_root_dim
    root_loss = F.mse_loss(predicted[:, :root_dim], target[:, :root_dim])
    body_loss = F.mse_loss(predicted[:, root_dim:], target[:, root_dim:])
    rows = torch.arange(prediction.shape[0], device=prediction.device)
    first_generation = selection.long().argmax(dim=1)
    predicted_boundary = prediction[rows, first_generation][:, [0, 2]].float()
    target_boundary = fixed["target"][rows, first_generation][:, [0, 2]].float()
    boundary_loss = F.mse_loss(predicted_boundary, target_boundary)
    nfpt = model.num_frames_per_token
    root_pred = prediction[:, :, :root_dim].float().reshape(
        prediction.shape[0], prediction.shape[1] * nfpt, -1
    )
    root_target = fixed["target"][:, :, :root_dim].float().reshape(
        prediction.shape[0], prediction.shape[1] * nfpt, -1
    )
    last_generation = fixed["generation_mask"].long().sum(dim=1) - 1
    last_generation = last_generation + fixed["history_len"]
    endpoint_prediction = root_pred[rows, last_generation][:, [0, 2]]
    endpoint_target = root_target[rows, last_generation][:, [0, 2]]
    root_endpoint_loss = F.mse_loss(endpoint_prediction, endpoint_target)
    # Include the history-to-generation transition. This is the actual
    # autoregressive window boundary whose velocity must remain continuous.
    velocity_mask = fixed["generation_mask"][:, 1:] & (
        fixed["history_mask"][:, :-1] | fixed["generation_mask"][:, :-1]
    )
    predicted_velocity = root_pred[:, 1:, [0, 2]] - root_pred[:, :-1, [0, 2]]
    target_velocity = root_target[:, 1:, [0, 2]] - root_target[:, :-1, [0, 2]]
    velocity_loss = F.mse_loss(
        predicted_velocity[velocity_mask], target_velocity[velocity_mask]
    )
    # Future constraints condition the current window but the ARDY denoiser
    # deliberately zeroes future outputs. They cannot be direct loss targets.
    constraint_mask = (
        fixed["motion_mask"].bool()
        & fixed["generation_mask"][:, :, None]
    )
    if constraint_mask.any():
        explicit_prediction = fixed["hybrid_converter"].get_explicit_motion_from_hybrid(
            prediction, fixed["pad_mask"], fixed["lengths"]
        )
        constraint_loss = F.mse_loss(
            explicit_prediction[constraint_mask].float(),
            fixed["motion"][constraint_mask].float(),
        )
    else:
        constraint_loss = prediction.new_zeros(())
        explicit_prediction = None
    needs_geometry = (
        fixed["fk_constraint_loss_weight"] > 0
        or fixed["foot_skate_loss_weight"] > 0
    )
    if needs_geometry:
        if explicit_prediction is None:
            explicit_prediction = fixed["hybrid_converter"].get_explicit_motion_from_hybrid(
                prediction, fixed["pad_mask"], fixed["lengths"]
            )
        generated_geometry = model.motion_rep.inverse(
            explicit_prediction, is_normalized=True
        )
        generated_joints = generated_geometry["posed_joints"]
        joint_mask = (
            fixed["joint_position_mask"]
            & fixed["generation_mask"][:, :, None]
        )
        if joint_mask.any():
            fk_constraint_loss = F.mse_loss(
                generated_joints[:, :, 1:][joint_mask].float(),
                fixed["target_joints"][:, :, 1:][joint_mask].float(),
            )
        else:
            fk_constraint_loss = prediction.new_zeros(())
        feet = generated_joints[:, :, fixed["foot_indices"]]
        horizontal_velocity = feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]]
        contact = (
            fixed["target_foot_contacts"][:, 1:].bool()
            & velocity_mask[:, :, None]
        )
        if contact.any():
            foot_skate_loss = horizontal_velocity[contact].square().mean()
        else:
            foot_skate_loss = prediction.new_zeros(())
    else:
        fk_constraint_loss = prediction.new_zeros(())
        foot_skate_loss = prediction.new_zeros(())
    total = (
        root_loss + body_loss
        + fixed["boundary_loss_weight"] * boundary_loss
        + fixed["root_velocity_loss_weight"] * velocity_loss
        + fixed["constraint_loss_weight"] * constraint_loss
        + fixed["fk_constraint_loss_weight"] * fk_constraint_loss
        + fixed["foot_skate_loss_weight"] * foot_skate_loss
        + fixed.get("generated_history_state_consistency_weight", 0.0)
        * state_consistency_loss
        + fixed.get("generated_history_root_endpoint_loss_weight", 0.0)
        * root_endpoint_loss
    )
    if return_components:
        return (
            total, root_loss, body_loss, boundary_loss, velocity_loss,
            constraint_loss, fk_constraint_loss, foot_skate_loss,
            state_consistency_loss, root_endpoint_loss,
        )
    return total


@torch.no_grad()
def _evaluate(raw, client, fixed):
    model = _unwrap_raw(raw)
    root_training = model.root_model.training
    body_training = model.body_model.training
    model.root_model.eval()
    model.body_model.eval()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return float(_loss(raw, client, fixed))
    finally:
        model.root_model.train(root_training)
        model.body_model.train(body_training)


def _evaluate_many(raw, client, batches) -> float:
    return float(np.mean([_evaluate(raw, client, batch) for batch in batches]))


def _save_resume_checkpoint(
    path: Path,
    *,
    args,
    raw,
    optimizer,
    step: int,
    losses: list[float],
    initial_eval: float,
    all_finite: bool,
    data_generator: torch.Generator,
    elapsed_seconds: float,
    device: torch.device,
    differentiable_rollout_steps: int,
) -> None:
    checkpoint = {
        "format_version": 2,
        "mode": args.mode,
        "phase": args.phase,
        "step": step,
        "root_model": _unwrap_block(_unwrap_raw(raw).root_model).state_dict(),
        "body_model": _unwrap_block(_unwrap_raw(raw).body_model).state_dict(),
        "denoiser_aux": _denoiser_aux_state(raw),
        "optimizer": optimizer.state_dict(),
        "losses": losses,
        "initial_eval_loss": initial_eval,
        "all_gradients_finite": all_finite,
        "data_generator_state": data_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
        "elapsed_seconds": elapsed_seconds,
        "differentiable_rollout_steps": differentiable_rollout_steps,
        "config": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.differentiable_rollout_probability <= 1.0:
        raise ValueError("differentiable-rollout-probability must be in [0, 1]")
    if args.differentiable_rollout_probability:
        if not 6 <= args.differentiable_rollout_windows <= 8:
            raise ValueError("long-loss tiny rollout requires 6-8 windows")
    if args.train_scope == "state_memory":
        if args.mode != "hybrid" or not args.content_memory_enabled:
            raise ValueError("state_memory train scope requires Hybrid content memory")
        if args.token_state_interaction not in {"state_query", "state_kv_modulation"}:
            raise ValueError("state_memory train scope requires a state-aware interaction")
        if not args.differentiable_rollout_probability:
            raise ValueError(
                "tiny state_memory scope requires a differentiable rollout task"
            )
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model(args.model, device=str(device), text_encoder=False)
    horizon = int(model.gen_horizon_len)
    if (
        args.differentiable_rollout_probability
        and args.frames < args.differentiable_rollout_windows * horizon
    ):
        raise ValueError("tiny frames are shorter than differentiable rollout")
    _replace_stats(model, args.stats_dir)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    client = install_ardy_sequence_kernel(
        model,
        ArdySequenceKernelConfig(
            mode=args.mode,
            attention_backend="sdpa",
            window_size_chunks=args.attention_window_chunks,
            num_sink_tokens=4,
            token_state_interaction=args.token_state_interaction,
            state_summary_dim=args.state_summary_dim,
            use_state_delta=args.use_state_delta,
            state_conditioning_scale_init=args.state_conditioning_scale_init,
            state_aware_output_mode=args.state_aware_output_mode,
            content_memory_enabled=args.content_memory_enabled,
            content_memory_slots=args.content_memory_slots,
            content_memory_use_state_delta=args.content_memory_use_state_delta,
            content_memory_write_gate_context=args.content_memory_write_gate_context,
            content_memory_decoder_init_scale=args.content_memory_decoder_init_scale,
            content_memory_write_gate_init_scale=args.content_memory_write_gate_init_scale,
            content_memory_write_gate_bias_init=args.content_memory_write_gate_bias_init,
            content_memory_read_scale_init=args.content_memory_read_scale_init,
            ff_size=args.minimal_ff_size,
        ),
    )
    raw = model.denoiser.model
    if args.train_scope == "denoiser":
        for parameter in raw.parameters():
            parameter.requires_grad_(True)
    elif args.train_scope == "mixer":
        for parameter in list(raw.root_model.parameters()) + list(raw.body_model.parameters()):
            parameter.requires_grad_(True)
    else:
        for name, parameter in raw.named_parameters():
            parameter.requires_grad_(
                any(
                    marker in name
                    for marker in (
                        ".state_projector.",
                        ".token_state_interaction.",
                        ".content_memory.",
                    )
                )
            )
    trainable = [parameter for parameter in raw.parameters() if parameter.requires_grad]
    raw.root_model.train()
    raw.body_model.train()
    model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps, device=device))
    dataset = HumanML3DArdyDataset(
        args.manifest, args.stats_dir, text_root=args.text_root, frames=args.frames, horizon=horizon,
        condition_dropout=args.constraint_dropout, seed=args.seed, text_cache_dir=args.text_cache,
        history_probability=args.history_probability,
        text_dropout=args.text_dropout,
    )
    data_generator = torch.Generator().manual_seed(args.seed + 555)
    if args.phase == "tiny":
        training_data = Subset(dataset, range(min(args.tiny_samples, len(dataset))))
        loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=False)
        batch = next(iter(loader))
    else:
        batch = default_collate([dataset[index] for index in range(args.batch_size)])
    fixed = _build_batch(
        model, batch, device, fixed_noise=True,
        boundary_loss_weight=args.boundary_loss_weight,
        root_velocity_loss_weight=args.root_velocity_loss_weight,
        constraint_loss_weight=args.constraint_loss_weight,
        fk_constraint_loss_weight=args.fk_constraint_loss_weight,
        foot_skate_loss_weight=args.foot_skate_loss_weight,
    )
    dev_fixed = []
    if args.dev_manifest:
        dev = HumanML3DArdyDataset(
            args.dev_manifest, args.stats_dir, text_root=args.text_root, frames=args.frames, horizon=horizon,
            condition_dropout=0.0, seed=args.seed + 10000, text_cache_dir=args.text_cache,
            history_probability=args.history_probability,
            text_dropout=0.0,
        )
        for batch_index in range(args.eval_batches):
            start = batch_index * args.batch_size
            samples = [dev[(start + offset) % len(dev)] for offset in range(args.batch_size)]
            dev_fixed.append(
                _build_batch(
                    model, default_collate(samples), device, fixed_noise=True,
                    boundary_loss_weight=args.boundary_loss_weight,
                    root_velocity_loss_weight=args.root_velocity_loss_weight,
                    constraint_loss_weight=args.constraint_loss_weight,
                    fk_constraint_loss_weight=args.fk_constraint_loss_weight,
                    foot_skate_loss_weight=args.foot_skate_loss_weight,
                )
            )
    eval_batches = dev_fixed or [fixed]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"
    start_step = 0
    losses = []
    all_finite = True
    elapsed_before = 0.0
    differentiable_rollout_steps = 0
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("mode") != args.mode:
            raise ValueError("initial checkpoint mode does not match requested run")
        _load_denoiser_weights(raw, initial)
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("mode") != args.mode or resume.get("phase") != args.phase:
            raise ValueError("resume checkpoint mode/phase does not match requested run")
        _load_denoiser_weights(raw, resume)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])
        losses = list(resume["losses"])
        all_finite = bool(resume["all_gradients_finite"])
        data_generator.set_state(resume["data_generator_state"].cpu())
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(resume["cuda_rng_state"].cpu(), device)
        elapsed_before = float(resume.get("elapsed_seconds", 0.0))
        differentiable_rollout_steps = int(
            resume.get("differentiable_rollout_steps", 0)
        )
        initial_eval = float(resume["initial_eval_loss"])
        if args.steps <= start_step:
            raise ValueError(f"--steps must exceed resumed step {start_step}")
    else:
        initial_eval = _evaluate_many(raw, client, eval_batches)
        if log_path.exists():
            raise FileExistsError(f"refusing to overwrite existing log without --resume: {log_path}")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        if args.phase == "pilot":
            indices = torch.randint(len(dataset), (args.batch_size,), generator=data_generator).tolist()
            seeds = torch.randint(
                0, 2**63 - 1, (args.batch_size,), generator=data_generator
            ).tolist()
            batch = default_collate(
                [dataset.sample(index, seed) for index, seed in zip(indices, seeds)]
            )
            current = _build_batch(
                model, batch, device, fixed_noise=False,
                boundary_loss_weight=args.boundary_loss_weight,
                root_velocity_loss_weight=args.root_velocity_loss_weight,
                constraint_loss_weight=args.constraint_loss_weight,
                fk_constraint_loss_weight=args.fk_constraint_loss_weight,
                foot_skate_loss_weight=args.foot_skate_loss_weight,
            )
        else:
            current = fixed
        optimizer.zero_grad(set_to_none=True)
        use_differentiable_rollout = bool(
            torch.rand((), generator=data_generator)
            < args.differentiable_rollout_probability
        )
        differentiable_metrics = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                loss, root_loss, body_loss, boundary_loss, velocity_loss,
                constraint_loss, fk_constraint_loss, foot_skate_loss,
                state_consistency_loss, root_endpoint_loss,
            ) = _loss(raw, client, current, return_components=True)
            if use_differentiable_rollout:
                surrogate, differentiable_metrics = _differentiable_rollout_surrogate(
                    model, raw, client, current,
                    mode=args.mode,
                    windows=args.differentiable_rollout_windows,
                    diffusion_steps=args.differentiable_rollout_diffusion_steps,
                    warm_prefix_windows=args.differentiable_rollout_warm_prefix_windows,
                    slope_weight=args.long_slope_loss_weight,
                    auc_weight=args.long_auc_loss_weight,
                    final_weight=args.long_final_loss_weight,
                    margin_cm_per_window=args.long_slope_margin_cm_per_window,
                    gradient_checkpointing=args.rollout_gradient_checkpointing,
                )
                loss = loss + surrogate
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        finite = bool(gradients) and all(torch.isfinite(grad).all() for grad in gradients)
        all_finite = all_finite and finite
        if not finite:
            raise RuntimeError(f"missing/non-finite gradients at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        differentiable_rollout_steps += int(use_differentiable_rollout)
        losses.append(float(loss))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": float(loss),
                "root_loss": float(root_loss),
                "body_loss": float(body_loss),
                "boundary_loss": float(boundary_loss),
                "root_velocity_loss": float(velocity_loss),
                "constraint_loss": float(constraint_loss),
                "fk_constraint_loss": float(fk_constraint_loss),
                "foot_skate_loss": float(foot_skate_loss),
                "generated_history_state_consistency_loss": float(
                    state_consistency_loss
                ),
                "generated_history_root_endpoint_loss": float(root_endpoint_loss),
                "grad_norm": float(grad_norm),
                "differentiable_rollout": use_differentiable_rollout,
                "differentiable_root_errors_cm": (
                    differentiable_metrics["root_errors_cm"].detach().float().mean(0).cpu().tolist()
                    if differentiable_metrics is not None else []
                ),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            eval_loss = _evaluate_many(raw, client, eval_batches)
            record = {"step": step, "eval_loss": eval_loss}
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if args.save_every and step % args.save_every == 0:
            _save_resume_checkpoint(
                output / f"checkpoint_step_{step:06d}.pt",
                args=args, raw=raw, optimizer=optimizer, step=step,
                losses=losses, initial_eval=initial_eval, all_finite=all_finite,
                data_generator=data_generator,
                elapsed_seconds=elapsed_before + time.perf_counter() - started,
                device=device,
                differentiable_rollout_steps=differentiable_rollout_steps,
            )
    final_eval = _evaluate_many(raw, client, eval_batches)
    first_window = float(np.mean(losses[:min(50, len(losses))]))
    last_window = float(np.mean(losses[-min(50, len(losses)):]))
    passed = (
        final_eval < 0.5 * initial_eval
        if args.phase == "tiny"
        else all_finite and last_window < first_window and final_eval < initial_eval
    )
    metrics = {
        "status": "PASS" if passed else "FAIL",
        "mode": args.mode,
        "model": args.model,
        "phase": args.phase,
        "steps": args.steps,
        "samples": len(batch["id"]),
        "ids": list(batch["id"]),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction": 1.0 - losses[-1] / losses[0],
        "first_50_mean_loss": first_window,
        "last_50_mean_loss": last_window,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "all_gradients_finite": all_finite,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "constraint_elements": [int(value) for value in fixed["motion_mask"].sum((1, 2))],
        "out_of_horizon": [bool(value) for value in batch["out_of_horizon"]],
        "history_frames": [int(value) for value in batch["history_frames"]],
        "text_backend": "llm2vec_cache" if args.text_cache else "deterministic_hash_smoke_only",
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "differentiable_rollout_steps": differentiable_rollout_steps,
        "differentiable_rollout_probability": args.differentiable_rollout_probability,
        "differentiable_rollout_windows": args.differentiable_rollout_windows,
        "train_scope": args.train_scope,
        "token_state_interaction": args.token_state_interaction,
        "state_aware_output_mode": args.state_aware_output_mode,
        "content_memory_enabled": args.content_memory_enabled,
        "content_memory_slots": args.content_memory_slots,
        "content_memory_write_gate_context": args.content_memory_write_gate_context,
    }
    checkpoint = {
        "mode": args.mode,
        "model": args.model,
        "root_model": _unwrap_block(_unwrap_raw(raw).root_model).state_dict(),
        "body_model": _unwrap_block(_unwrap_raw(raw).body_model).state_dict(),
        "denoiser_aux": _denoiser_aux_state(raw),
        "metrics": metrics,
    }
    torch.save(checkpoint, output / f"{args.mode}_humanml3d_{args.phase}.pt")
    if not args.save_every or args.steps % args.save_every != 0:
        _save_resume_checkpoint(
            output / f"checkpoint_step_{args.steps:06d}.pt",
            args=args, raw=raw, optimizer=optimizer, step=args.steps,
            losses=losses, initial_eval=initial_eval, all_finite=all_finite,
            data_generator=data_generator, elapsed_seconds=metrics["elapsed_seconds"],
            device=device,
            differentiable_rollout_steps=differentiable_rollout_steps,
        )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if metrics["status"] != "PASS":
        raise RuntimeError(f"{args.phase} training gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
