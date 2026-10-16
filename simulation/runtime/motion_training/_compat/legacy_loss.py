"""Preserved paired numerical contract; no CLI, path injection or import side effects."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from ardy_adapter.history_commit import canonical_history_prime
from ardy_adapter.differentiable_rollout import differentiable_rollout_loss

def _replace_stats(model, stats_dir: str):
    from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
    from g1_motion_contract import inspect_stats, validate_codec_stats, install_decoder_history

    old = model.motion_rep
    rep = ArdyMotionRep(old.skeleton, old.fps, stats_path=stats_dir)
    metadata = inspect_stats(stats_dir)
    validate_codec_stats(old, rep, metadata)
    rep.skeleton = old.skeleton
    model.motion_rep = rep
    raw = model.denoiser.model
    raw.motion_rep = rep
    model.autoencoder._motion_rep = rep
    for converter in (model.hybrid, raw.hybrid):
        converter._motion_rep = rep
        converter.__dict__.pop("motion_rep", None)
    model.motion_contract = metadata['version']
    model.motion_contract_metadata = metadata
    raw.motion_contract_metadata = metadata
    if metadata['version'] != 'legacy':
        install_decoder_history(model, metadata['decoder_history_frames'])
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
    decoded_body_loss_weight: float = 1.0,
    kinematic_consistency_loss_weight: float = 1.0,
    joint_limit_loss_weight: float = 0.0,
    joint_limit_spec: str | None = None,
    accepted_boundary_loss_weight: float = 0.0,
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
    motion_contract = getattr(model, 'motion_contract', 'legacy')
    decoder_history_hybrid = None
    accepted_history_tail = None
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
    if motion_contract != 'legacy' and 'length' in batch:
        lengths = batch['length'].to(device).clamp(max=frames)
        pad = torch.arange(frames, device=device)[None] < lengths[:, None]
        motion_mask = motion_mask * pad[:, :, None]
    with torch.no_grad():
        if sequential_cache:
            from g1_bounded_codec import encode_training_history
            clean, _ = encode_training_history(model, motion, lengths, pad)
        else:
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
        if motion_contract != 'legacy':
            decoder_history_hybrid = clean[:, (current_start-horizon)//model.hybrid.num_frames_per_token:current_start//model.hybrid.num_frames_per_token].detach()
            if accepted_boundary_loss_weight > 0:
                accepted_history_tail = motion[:,current_start-2:current_start].detach()
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
        from g1_training_noise import sample_training_timesteps
        timestep = sample_training_timesteps(batch_size, model.diffusion.num_base_steps, device,
            getattr(model, 'training_noise_policy', 'uniform'))
        noise = torch.randn_like(clean)
    noisy = model.diffusion.q_sample(clean, timestep, noise)
    x = torch.zeros_like(clean)
    x[history_token_mask] = clean[history_token_mask]
    x[generation_token_mask] = noisy[generation_token_mask]
    return {
        "motion_contract": motion_contract,
        "training_codec_contract": getattr(model, 'training_codec_contract', 'legacy'),
        "decoded_body_loss_weight": decoded_body_loss_weight,
        "kinematic_consistency_loss_weight": kinematic_consistency_loss_weight,
        "joint_limit_loss_weight": joint_limit_loss_weight,
        "joint_limit_spec": joint_limit_spec,
        "accepted_boundary_loss_weight": accepted_boundary_loss_weight,
        "accepted_history_tail": accepted_history_tail,
        "decoder_history_hybrid": decoder_history_hybrid,
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
    if hasattr(model, 'motion_contract_metadata'):
        from g1_motion_contract import validate_checkpoint
        validate_checkpoint(checkpoint, model.motion_contract_metadata)
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
    """Commit teacher history chunks through the selected write contract."""
    model = _unwrap_raw(raw)
    histories = fixed["history_hybrid"]
    headings = fixed["history_headings"]
    if histories.ndim != 4 or headings.shape != histories.shape[:2]:
        raise ValueError("multiwindow teacher history must be [B,W,T,D] with [B,W] headings")
    root_training = model.root_model.training
    body_training = model.body_model.training
    model.root_model.eval()
    model.body_model.eval()
    contract = fixed.get("history_write_contract", "legacy_synthetic_tail")
    if contract not in {"legacy_synthetic_tail", "accepted_history_v3_history_only"}:
        raise ValueError(f"unknown history_write_contract: {contract}")
    include_synthetic_generation = contract == "legacy_synthetic_tail"
    fixed['history_write_calls'] = []
    try:
        for chunk_id in range(histories.shape[1]):
            if (
                fixed.get("training_state_intervention") == "fresh_each_window"
                and client is not None
                and getattr(client, "controller", None) is not None
            ):
                client.controller.reset_mamba_only()
            layout = canonical_history_prime(
                model,
                histories[:, chunk_id],
                text_feat=fixed["text"],
                text_pad_mask=fixed["text_mask"],
                first_heading_angle=headings[:, chunk_id],
                next_generation_frames=fixed["next_generation_frames"],
                chunk_id=chunk_id,
                include_synthetic_generation=include_synthetic_generation,
            )
            fixed['history_write_calls'].append(dict(chunk_id=chunk_id,
                include_synthetic_generation=include_synthetic_generation,
                input_tokens=layout.x.shape[1],history_tokens=int(layout.history_token_mask[0].sum()),
                generation_frames=int(layout.generation_len[0]),detach=True))
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
    target_frame: str = "world_recovery",
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
    heading_alignment = torch.zeros_like(raw_recovery_norm)
    phase_search_starts = None
    phase_search_costs = None
    if target_frame in {"local_continuation", "phase_compatible_continuation"}:
        from g1_continuation_target import continuation_target
        if full_mask.any():
            raise ValueError('Local continuation cannot relabel absolute spatial constraints')
        if target_frame == "local_continuation":
            target_physical, heading_alignment = continuation_target(
                model.motion_rep, physical[:, prefix_frames:target_end],
                physical[:, prefix_frames-1], generated_physical[:, -1])
        else:
            from g1_continuation_target import phase_compatible_continuation_target
            target_physical, phase_search_starts, phase_search_costs = (
                phase_compatible_continuation_target(
                    model.motion_rep, physical, prefix_frames,
                    generated_physical[:, -2:], horizon)
            )
            rows_for_phase = torch.arange(batch_size, device=device)
            # The phase helper returns the first target-frame index.  Heading
            # alignment uses the preceding frame, which is the actual seam
            # boundary used by continuation_target().
            phase_teacher_boundary = physical[
                rows_for_phase, phase_search_starts - 1
            ]
            teacher_heading = torch.atan2(
                phase_teacher_boundary[:, 4], phase_teacher_boundary[:, 3]
            )
            generated_heading = torch.atan2(
                generated_physical[:, -1, 4], generated_physical[:, -1, 3]
            )
            heading_alignment = torch.atan2(
                torch.sin(generated_heading - teacher_heading),
                torch.cos(generated_heading - teacher_heading),
            )
        applied_recovery_norm = torch.zeros_like(raw_recovery_norm)
    elif target_frame != "world_recovery":
        raise ValueError(f'Unknown generated-history target frame: {target_frame}')
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
    if 'matched_noise_seed' in fixed:
        from main3_training import tensor_hash
        generator = torch.Generator(device=device).manual_seed(fixed['matched_noise_seed'] + 2)
        target_noise = torch.randn((batch_size, horizon // model.num_frames_per_token, clean.shape[-1]),
                                   device=device, dtype=clean.dtype, generator=generator)
        noise = torch.zeros_like(clean)
        noise[:, -target_noise.shape[1]:] = target_noise
        fixed['matched_target_noise_sha256'] = tensor_hash(target_noise)
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
    if fixed.get('motion_contract', 'legacy') != 'legacy' and mode != 'original':
        # Encode accepted generated poses in the same origin as the next target;
        # preserve a causal encoder prefix, then retain one decoder window.
        generated_local = generated_physical.clone()
        generated_local[:, :, 0] -= generated_endpoint[:, None, 0]
        generated_local[:, :, 2] -= generated_endpoint[:, None, 1]
        combined = torch.cat((model.motion_rep.normalize(generated_local), target_motion), dim=1)
        combined_pad = torch.ones(combined.shape[:2], device=device, dtype=torch.bool)
        with torch.no_grad():
            from g1_bounded_codec import encode_training_history
            encoded, _ = encode_training_history(model, combined, combined_pad.sum(1), combined_pad)
        nt = horizon // model.hybrid.num_frames_per_token
        clean = encoded[:, -nt:]
        x = model.diffusion.q_sample(clean, fixed['timestep'], noise)
        supervision['decoder_history_hybrid'] = encoded[:, -2*nt:-nt].detach()
        if fixed.get('accepted_boundary_loss_weight', 0) > 0:
            supervision['accepted_history_tail'] = model.motion_rep.normalize(generated_local[:,-2:]).detach()
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
            "generated_history_target_frame": target_frame,
            "generated_history_heading_alignment": heading_alignment,
            "generated_history_phase_search_starts": phase_search_starts,
            "generated_history_phase_search_costs": phase_search_costs,
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
    target_frame: str = "world_recovery",
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
    if target_frame not in {
        "world_recovery", "local_continuation", "phase_compatible_continuation"
    }:
        raise ValueError('Unknown generated-history target frame')
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
    if target_frame in {"local_continuation", "phase_compatible_continuation"}:
        # This is an explicitly text-only task. Preserve world-constrained
        # supervision in the separate short/teacher branches, never silently
        # move an externally specified target with the generated trajectory.
        full_mask = torch.zeros_like(full_mask)
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
                from g1_motion_contract import generate_with_committed_decode
                generated_tail = generate_with_committed_decode(
                    model,
                    texts,
                    generated_frames,
                    decoder_prefix=seed_hybrid[:, -horizon//model.hybrid.num_frames_per_token:],
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
            target_frame=target_frame,
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
    from g1_motion_contract import CONTRACT, decode_training_prediction
    corrected = fixed.get('motion_contract', 'legacy') == CONTRACT
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
    from g1_training_precision import prediction_loss_context
    with prediction_loss_context(raw, prediction):
        return _prediction_loss(raw, fixed, prediction, state_consistency_loss, return_components)


def _prediction_loss(raw, fixed, prediction, state_consistency_loss, return_components=False):
    from g1_motion_contract import CONTRACT, decode_training_prediction
    corrected = fixed.get("motion_contract", "legacy") == CONTRACT
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
        explicit_prediction = decode_training_prediction(prediction, fixed)
        constraint_loss = F.mse_loss(
            explicit_prediction[constraint_mask].float(),
            fixed["observed"][constraint_mask].float(),
        )
    else:
        constraint_loss = prediction.new_zeros(())
        explicit_prediction = None
    needs_geometry = (
        fixed["fk_constraint_loss_weight"] > 0
        or fixed["foot_skate_loss_weight"] > 0
        or corrected
    )
    if needs_geometry:
        if explicit_prediction is None:
            explicit_prediction = decode_training_prediction(prediction, fixed)
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
        if corrected:
            horizontal_velocity = horizontal_velocity * model.motion_rep.fps
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
    decoded_body_loss = prediction.new_zeros(())
    consistency_loss = prediction.new_zeros(())
    if corrected:
        valid = fixed['generation_mask']
        body_slice = model.motion_rep.body_slice
        decoded_body_loss = F.smooth_l1_loss(explicit_prediction[:, :, body_slice][valid].float(),
                                           fixed['motion'][:, :, body_slice][valid].float())
        position_geometry = model.motion_rep.inverse(explicit_prediction, is_normalized=True, posed_joints_from='positions')
        # The positions inverse does not define pelvis height. Compare only
        # nonroot joints; keep the official inverse and all weights unchanged.
        consistency_loss = F.mse_loss(position_geometry['posed_joints'][:, :, 1:][valid].float(),
                                      generated_joints[:, :, 1:][valid].float())
    fixed['contract_loss_components'] = (decoded_body_loss.detach(), consistency_loss.detach())
    limit_loss=prediction.new_zeros(())
    if fixed.get('joint_limit_loss_weight',0)>0:
        if not corrected:raise ValueError('Joint-limit objective requires the corrected G1 contract')
        from g1_joint_limits import joint_limit_loss
        limit_loss=joint_limit_loss(generated_geometry,model.motion_rep.skeleton,
                                   fixed['joint_limit_spec'],fixed['generation_mask'])
    fixed['joint_limit_loss_value']=limit_loss.detach()
    accepted_boundary_loss = prediction.new_zeros(())
    if fixed.get('accepted_boundary_loss_weight', 0) > 0:
        if not corrected:
            raise ValueError('Accepted boundary objective requires corrected physical geometry')
        from g1_boundary_kinematics import accepted_boundary_kinematics
        first = fixed['generation_mask'].long().argmax(dim=1)
        history_tail = fixed.get('accepted_history_tail')
        if history_tail is None:
            # Mixed explicit-history / cold rows. Cold rows are masked out.
            indices = (first[:,None] + first.new_tensor([-2,-1])[None]).clamp_min(0)
            history_tail = fixed['motion'][rows[:,None],indices].detach()
            valid_boundary = fixed['history_len'] >= 2
        else:
            valid_boundary = torch.ones(len(first),dtype=torch.bool,device=prediction.device)
        with torch.no_grad():
            accepted_geometry = model.motion_rep.inverse(history_tail,is_normalized=True)
        parts = accepted_boundary_kinematics(generated_joints[rows,first].float(),
            accepted_geometry['posed_joints'].float(),accepted_geometry['foot_contacts'][:,-1],
            fixed['target_foot_contacts'][rows,first],fixed['foot_indices'],valid_boundary,
            fps=model.motion_rep.fps)
        accepted_boundary_loss = parts['total']
        fixed['accepted_boundary_components'] = {k:v.detach() for k,v in parts.items()}
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
        + fixed.get('decoded_body_loss_weight', 1.0) * decoded_body_loss
        + fixed.get('kinematic_consistency_loss_weight', 1.0) * consistency_loss
        + fixed.get('joint_limit_loss_weight', 0.0) * limit_loss
        + fixed.get('accepted_boundary_loss_weight', 0.0) * accepted_boundary_loss
    )
    if return_components:
        return (
            total, root_loss, body_loss, boundary_loss, velocity_loss,
            constraint_loss, fk_constraint_loss, foot_skate_loss,
            state_consistency_loss, root_endpoint_loss,
        )
    return total


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

