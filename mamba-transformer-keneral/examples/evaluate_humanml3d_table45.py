"""Evaluate reproducible Table 4/5 proxy metrics for trained ARDY adapters.

This checkout does not include the paper's HumanML3D retrieval/FID evaluator.
The script therefore reports only directly measurable geometry, constraint,
latency, and memory metrics and labels the missing learned metrics explicitly.
"""

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
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))
for path in (
    os.environ.get("MAMBA_REPO", "/data3/zts/packages/state-spaces-mamba"),
    os.environ.get("MAMBA_TRITON_OVERLAY", "/data3/zts/packages/mamba_op_triton"),
):
    sys.path.insert(0, path)

from ardy_data import HumanML3DArdyDataset  # noqa: E402
from train_humanml3d_ardy_tiny import _load_denoiser_weights, _replace_stats  # noqa: E402
from evaluation_checkpoint_config import (  # noqa: E402
    add_sequence_checkpoint_arguments,
    sequence_config_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "minimal", "hybrid"], required=True)
    parser.add_argument("--model", default="core")
    parser.add_argument("--dataset-name", default="HumanML3D")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "HumanML3D-trained sequence checkpoint. Required for Minimal/Hybrid; "
            "optional for Original (omitted Original uses Released ARDY native weights)."
        ),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--text-root", default=None)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--protocol", choices=["table4", "table5_in", "table5_out"], required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--cfg-weight-text", type=float, default=2.0)
    parser.add_argument("--cfg-weight-constraint", type=float, default=2.0)
    parser.add_argument(
        "--memory-mode",
        choices=["persistent", "explicit_history", "hybrid_debug"],
        default="persistent",
        help="persistent=paper cache-only; explicit_history=diagnostic; hybrid_debug explicitly permits both.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=None,
        help="Explicit autoregressive history; defaults to 0 for Table 4 and 40 for Table 5.",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--artifacts-output",
        default=None,
        help="Optional compressed same-generation geometry cache for secondary evaluators.",
    )
    parser.add_argument("--panel", default=None)
    parser.add_argument("--protocol-lock", default=None)
    parser.add_argument("--per-sample-output", default=None)
    parser.add_argument("--attention-window-chunks", type=int, default=4)
    parser.add_argument("--minimal-ff-size", type=int, default=None)
    add_sequence_checkpoint_arguments(parser)
    return parser.parse_args()


def _build_panel_sample(dataset, record: dict, panel: dict, motion_rep) -> dict:
    features = torch.from_numpy(np.load(record["feature_path"])).float()
    start = int(panel["crop_start"])
    end = int(panel["crop_end"])
    if end - start != dataset.frames or start < 0 or end > len(features):
        raise ValueError("Table 4 panel crop is invalid")
    physical = features[start:end].clone()
    normalized = (physical - dataset.mean) / torch.sqrt(dataset.std.square() + 1e-5)
    observed = torch.zeros_like(normalized)
    mask = torch.zeros_like(normalized, dtype=torch.bool)
    if len(panel["goal_frames"]) != 1 or len(panel["goal_joints"]) != 1:
        raise ValueError("Table 4 panel requires exactly one goal")
    frame = int(panel["goal_frames"][0])
    joint = int(panel["goal_joints"][0])
    root_slice = motion_rep.slice_dict["root_pos"]
    heading_slice = motion_rep.slice_dict["global_root_heading"]
    local_slice = motion_rep.slice_dict["local_joints_positions"]
    if joint == 0:
        mask[frame, root_slice.start] = True
        mask[frame, root_slice.start + 2] = True
    else:
        mask[frame, root_slice] = True
        mask[frame, heading_slice] = True
        offset = local_slice.start + (joint - 1) * 3
        mask[frame, offset:offset + 3] = True
    observed[mask] = normalized[mask]
    output = {
        "id": record["id"],
        "panel_id": panel["panel_id"],
        "actor_uid": record.get("actor_uid"),
        "noise_seed": int(panel["noise_seed"]),
        "motion": normalized,
        "observed_motion": observed,
        "motion_mask": mask.float(),
        "text": panel["caption"],
        "goal_frame": frame,
        "goal_joint": joint,
    }
    if dataset.text_cache is not None:
        row = dataset.text_cache_index[record["id"]]
        output["text_feature"] = torch.from_numpy(
            np.array(dataset.text_cache[row], copy=True)
        ).float()
    return output


def _select_indices(dataset: HumanML3DArdyDataset, frames: int, count: int, seed: int) -> list[int]:
    eligible = []
    for index, record in enumerate(dataset.records):
        motion = np.load(record["feature_path"], mmap_mode="r")
        if len(motion) >= frames:
            eligible.append(index)
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} records have at least {frames} frames")
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.choice(eligible, size=count, replace=False)]


def _joint_from_mask(mask: torch.Tensor, motion_rep) -> tuple[int, int]:
    joint_mask = mask[:, motion_rep.slice_dict["local_joints_positions"]]
    positions = torch.nonzero(joint_mask.any(dim=-1), as_tuple=False).flatten()
    if len(positions) != 1:
        raise RuntimeError(f"expected one constrained frame, got {positions.tolist()}")
    frame = int(positions[0])
    columns = torch.nonzero(joint_mask[frame], as_tuple=False).flatten()
    return frame, int(columns[0] // 3) + 1


def _skate_counts(
    posed: torch.Tensor,
    foot_contacts: torch.Tensor,
    foot_indices: list[int],
    fps: int,
) -> tuple[int, int]:
    feet = posed[:, :, foot_indices]
    horizontal_speed = torch.linalg.vector_norm(
        feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]], dim=-1
    ) * fps
    contact = foot_contacts[:, 1:].bool()
    return int(((horizontal_speed > 0.15) & contact).sum()), contact.numel()


def _rigid_pa_mpjpe(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rigid (rotation + translation, no scale) Procrustes MPJPE in metres."""
    source = predicted.reshape(-1, 3).float()
    reference = target.reshape(-1, 3).float()
    source_centered = source - source.mean(0, keepdim=True)
    reference_centered = reference - reference.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(source_centered.T @ reference_centered)
    rotation = vh.T @ u.T
    if torch.linalg.det(rotation) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.T @ u.T
    aligned = source_centered @ rotation.T + reference.mean(0, keepdim=True)
    return torch.linalg.vector_norm(aligned - reference, dim=-1).mean()


def _angle_mae_degrees(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = predicted.float().transpose(-1, -2) @ target.float()
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine)).mean()


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch size must be positive")
    if args.mode != "original" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for Minimal/Hybrid evaluation")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    from ardy.model import load_model
    from ardy.motion_rep.tools import length_to_mask
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model(args.model, device=str(device), text_encoder=False)
    horizon = int(model.gen_horizon_len)
    dataset_stats_dir = args.stats_dir
    if args.mode != "original" or args.checkpoint is not None:
        # A HumanML3D-trained Original checkpoint must use HumanML3D stats.
        # Only the no-checkpoint path is the separate Released ARDY reference.
        _replace_stats(model, args.stats_dir)
    else:
        # Released ARDY weights were trained with the bundled Rigplay stats.
        # Re-encode HumanML3D physical motion with those stats for a valid
        # released-checkpoint reference instead of mutating its normalization.
        dataset_stats_dir = model.motion_rep.stats.folder
    raw = model.denoiser.model
    sequence_architecture = None
    if args.mode == "original":
        client = model
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            if checkpoint.get("model", args.model) != args.model:
                raise ValueError(
                    f"checkpoint model={checkpoint.get('model')!r}, requested model={args.model!r}"
                )
            if checkpoint.get("mode") != "original":
                raise ValueError(
                    f"HumanML3D original checkpoint has mode={checkpoint.get('mode')!r}"
                )
            raw = model.denoiser.model
            _load_denoiser_weights(raw, checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sequence_kwargs, sequence_architecture = sequence_config_kwargs(
            args, checkpoint
        )
        client = install_ardy_sequence_kernel(
            model,
            ArdySequenceKernelConfig(
                mode=args.mode,
                attention_backend="sdpa",
                window_size_chunks=args.attention_window_chunks,
                num_sink_tokens=4,
                ff_size=args.minimal_ff_size,
                **sequence_kwargs,
            ),
        )
        if checkpoint.get("model", args.model) != args.model:
            raise ValueError(
                f"checkpoint model={checkpoint.get('model')!r}, requested model={args.model!r}"
            )
        raw = model.denoiser.model
        _load_denoiser_weights(raw, checkpoint)
        if args.memory_mode == "explicit_history":
            client.controller.uninstall()
    raw.root_model.eval()
    raw.body_model.eval()

    frames = horizon if args.protocol == "table4" else 2 * horizon
    history_frames = args.history_frames
    if history_frames is None:
        history_frames = 0 if args.memory_mode in {"persistent", "hybrid_debug"} else (
            0 if args.protocol == "table4" else horizon
        )
    if args.mode != "original" and args.memory_mode == "persistent" and history_frames != 0:
        raise ValueError("paper persistent evaluation is cache-only; use --memory-mode hybrid_debug for explicit history")
    if args.mode != "original" and args.memory_mode == "hybrid_debug" and history_frames == 0:
        raise ValueError("hybrid_debug requires explicit history_frames > 0")
    out_of_horizon = args.protocol == "table5_out"
    dataset = HumanML3DArdyDataset(
        args.manifest,
        dataset_stats_dir,
        text_root=args.text_root,
        frames=frames,
        horizon=horizon,
        condition_dropout=0.0,
        seed=args.seed,
        text_cache_dir=args.text_cache,
        history_probability=0.0,
    )
    if args.panel:
        panel_rows = [
            json.loads(line)
            for line in Path(args.panel).read_text().splitlines()
            if line.strip()
        ]
        if len(panel_rows) != args.samples:
            raise ValueError("panel row count differs from --samples")
        if args.protocol != "table4":
            raise ValueError("immutable panel support in this evaluator is Table 4 only")
        if len({row["panel_id"] for row in panel_rows}) != len(panel_rows):
            raise ValueError("panel IDs must be unique")
        if args.protocol_lock:
            lock = json.loads(Path(args.protocol_lock).read_text())
            split_role = panel_rows[0]["split_role"]
            expected = lock["panel_sha256"][f"{split_role}/table4"]
            actual = hashlib.sha256(Path(args.panel).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("Table 4 panel hash mismatch")
            if int(lock["diffusion_steps"]) != args.diffusion_steps:
                raise ValueError("protocol lock diffusion steps mismatch")
            if int(lock["batch_size"]) != args.batch_size:
                raise ValueError("protocol lock batch size mismatch")
            cfg = lock["cfg"]
            if (
                float(cfg["text"]) != args.cfg_weight_text
                or float(cfg["constraint"]) != args.cfg_weight_constraint
            ):
                raise ValueError("protocol lock CFG mismatch")
            allowed_attention_windows = {
                int(value)
                for value in lock.get(
                    "attention_window_variants",
                    [lock["attention_window_chunks"]],
                )
            }
            if args.attention_window_chunks not in allowed_attention_windows:
                raise ValueError("protocol lock attention window mismatch")
        by_id = {record["id"]: record for record in dataset.records}
        samples = [
            _build_panel_sample(dataset, by_id[row["id"]], row, model.motion_rep)
            for row in panel_rows
        ]
    else:
        indices = _select_indices(dataset, frames, args.samples, args.seed)
        samples = [
            dataset.sample(
                index,
                args.seed + 1000003 * offset,
                control="joint_position",
                out_of_horizon=out_of_horizon,
            )
            for offset, index in enumerate(indices)
        ]

    constraint_errors = []
    root_speeds = []
    boundary_root_speeds = []
    max_root_speed_event = {"speed": float("-inf"), "sample_id": None, "frame": None}
    latencies = []
    sliding = 0
    contacts = 0
    peak_memory = 0.0
    foot_indices = list(model.skeleton.foot_joint_idx)
    fps = int(model.motion_rep.fps)
    root_ade_cm = []
    root_fde_cm = []
    pa_mpjpe_cm = []
    jitter_m_per_frame2 = []
    angle_mae_deg = []
    artifact_generated_joints = []
    artifact_target_joints = []
    artifact_generated_root = []
    artifact_target_root = []
    artifact_ids = []
    artifact_texts = []
    per_sample_rows = []
    torch.cuda.reset_peak_memory_stats(device)

    for start in range(0, len(samples), args.batch_size):
        items = samples[start:start + args.batch_size]
        batch = default_collate(items)
        size = len(items)
        lengths = torch.full((size,), frames, dtype=torch.long, device=device)
        pad_mask = length_to_mask(lengths)
        motion_mask = batch["motion_mask"].to(device)
        observed = batch["observed_motion"].to(device)
        text_feat = batch["text_feature"].to(device)[:, None]
        text_mask = torch.ones(text_feat.shape[:2], dtype=torch.bool, device=device)
        batch_noise_seed = int.from_bytes(
            hashlib.sha256(
                ":".join(str(item.get("noise_seed", args.seed)) for item in items).encode()
            ).digest()[:8], "little"
        ) % (2**63 - 1)
        torch.manual_seed(batch_noise_seed)
        torch.cuda.manual_seed_all(batch_noise_seed)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = client(
                list(batch["text"]),
                frames,
                num_denoising_steps=args.diffusion_steps,
                pad_mask=pad_mask,
                first_heading_angle=torch.zeros(size, device=device),
                motion_mask=motion_mask,
                observed_motion=observed,
                cfg_weight=(args.cfg_weight_text, args.cfg_weight_constraint),
                crop_history_length=history_frames,
                text_feat=text_feat,
                text_pad_mask=text_mask,
                progress_bar=lambda values: values,
            )
        torch.cuda.synchronize(device)
        latencies.extend([(time.perf_counter() - started) / size] * size)
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device) / 2**20)

        generated = model.motion_rep.inverse(prediction, is_normalized=True)
        target = model.motion_rep.inverse(batch["motion"].to(device), is_normalized=True)
        generated_joints = generated["posed_joints"]
        target_joints = target["posed_joints"]
        generated_root = generated["root_positions"]
        target_root = target["root_positions"]
        generated_rotations = generated["local_rot_mats"]
        target_rotations = target["local_rot_mats"]
        for row, item in enumerate(items):
            if "goal_frame" in item:
                frame, joint = int(item["goal_frame"]), int(item["goal_joint"])
            else:
                frame, joint = _joint_from_mask(item["motion_mask"], model.motion_rep)
            if joint == model.skeleton.root_idx:
                error_cm = torch.linalg.vector_norm(
                    generated_root[row, frame, [0, 2]] - target_root[row, frame, [0, 2]]
                ) * 100.0
            else:
                error_cm = torch.linalg.vector_norm(
                    generated_joints[row, frame, joint] - target_joints[row, frame, joint]
                ) * 100.0
            constraint_errors.append(float(error_cm))
            root_distance = torch.linalg.vector_norm(
                generated_root[row] - target_root[row], dim=-1
            )
            root_ade_cm.append(float(root_distance.mean() * 100.0))
            root_fde_cm.append(float(root_distance[-1] * 100.0))
            pa_mpjpe_cm.append(
                float(_rigid_pa_mpjpe(generated_joints[row], target_joints[row]) * 100.0)
            )
            second_difference = (
                generated_joints[row, 2:]
                - 2.0 * generated_joints[row, 1:-1]
                + generated_joints[row, :-2]
            )
            jitter_m_per_frame2.append(
                float(torch.linalg.vector_norm(second_difference, dim=-1).mean())
            )
            angle_mae_deg.append(
                float(_angle_mae_degrees(generated_rotations[row], target_rotations[row]))
            )
            sample_bad, sample_contacts = _skate_counts(
                generated_joints[row : row + 1],
                generated["foot_contacts"][row : row + 1],
                foot_indices,
                fps,
            )
            sample_speed = torch.linalg.vector_norm(
                generated_root[row, 1:] - generated_root[row, :-1], dim=-1
            ) * fps
            per_sample_rows.append({
                "panel_id": item.get("panel_id", item["id"]),
                "motion_id": item["id"],
                "actor_uid": item.get("actor_uid"),
                "protocol": args.protocol,
                "constraint_error_cm": float(error_cm),
                "root_ade_cm": float(root_distance.mean() * 100.0),
                "root_fde_cm": float(root_distance[-1] * 100.0),
                "pa_mpjpe_cm": float(
                    _rigid_pa_mpjpe(generated_joints[row], target_joints[row]) * 100.0
                ),
                "jitter_m_per_frame2": float(
                    torch.linalg.vector_norm(second_difference, dim=-1).mean()
                ),
                "foot_skate_percent_proxy": (
                    100.0 * sample_bad / max(sample_contacts, 1)
                ),
                "max_root_speed_m_s": float(sample_speed.max()),
            })
            if args.artifacts_output:
                artifact_generated_joints.append(generated_joints[row].float().cpu().numpy())
                artifact_target_joints.append(target_joints[row].float().cpu().numpy())
                artifact_generated_root.append(generated_root[row].float().cpu().numpy())
                artifact_target_root.append(target_root[row].float().cpu().numpy())
                artifact_ids.append(item.get("panel_id", item["id"]))
                artifact_texts.append(item["text"])
        bad, total = _skate_counts(
            generated_joints, generated["foot_contacts"], foot_indices, fps
        )
        sliding += bad
        contacts += total
        root = generated["root_positions"]
        speed = torch.linalg.vector_norm(root[:, 1:] - root[:, :-1], dim=-1) * fps
        root_speeds.extend(float(value) for value in speed.flatten())
        if frames > model.gen_horizon_len:
            boundary_root_speeds.extend(
                float(value) for value in speed[:, model.gen_horizon_len - 1]
            )
        for row, item in enumerate(items):
            row_speed, frame = speed[row].max(dim=0)
            if float(row_speed) > max_root_speed_event["speed"]:
                max_root_speed_event = {
                    "speed": float(row_speed),
                    "sample_id": item["id"],
                    "frame": int(frame) + 1,
                }

    result = {
        "status": "PASS" if all(np.isfinite(constraint_errors + root_speeds + latencies)) else "FAIL",
        "metric_scope": f"ARDY Table 4/5 proxy on {args.dataset_name}; learned official evaluator unavailable",
        "mode": args.mode,
        "model": args.model,
        "protocol": args.protocol,
        "samples": len(samples),
        "frames": frames,
        "diffusion_steps": args.diffusion_steps,
        "cfg_weight_text": args.cfg_weight_text,
        "cfg_weight_constraint": args.cfg_weight_constraint,
        "history_frames": history_frames,
        "attention_window_chunks": args.attention_window_chunks,
        "minimal_ff_size": args.minimal_ff_size,
        "sequence_architecture": sequence_architecture,
        # Original ARDY carries generated motion through its autoregressive
        # history sequence; it has no adapter-owned persistent KV/Mamba state.
        "memory_mode": "explicit_history" if args.mode == "original" else args.memory_mode,
        "protocol_guard": (
            "native_explicit_history" if args.mode == "original" else
            ("paper_cache_only" if args.memory_mode == "persistent" else args.memory_mode)
        ),
        "r_precision": None,
        "fid": None,
        "constraint_error_cm_mean": float(np.mean(constraint_errors)),
        "constraint_error_cm_median": float(np.median(constraint_errors)),
        "mobench_root_ade_cm_mean": float(np.mean(root_ade_cm)),
        "mobench_root_fde_cm_mean": float(np.mean(root_fde_cm)),
        "mobench_pa_mpjpe_cm_mean": float(np.mean(pa_mpjpe_cm)),
        "mobench_jitter_m_per_frame2_mean": float(np.mean(jitter_m_per_frame2)),
        "mobench_angle_mae_deg_mean": float(np.mean(angle_mae_deg)),
        "skate_percent_proxy": 100.0 * sliding / max(contacts, 1),
        "latency_seconds_per_sample": float(np.mean(latencies)),
        "peak_gpu_memory_mib": peak_memory,
        "mean_root_speed": float(np.mean(root_speeds)),
        "max_root_speed": float(np.max(root_speeds)),
        "mean_window_boundary_root_speed": (
            float(np.mean(boundary_root_speeds)) if boundary_root_speeds else None
        ),
        "max_window_boundary_root_speed": (
            float(np.max(boundary_root_speeds)) if boundary_root_speeds else None
        ),
        "max_root_speed_sample_id": max_root_speed_event["sample_id"],
        "max_root_speed_frame": max_root_speed_event["frame"],
        "active_parameters": sum(
            parameter.numel()
            for parameter in list(raw.root_model.parameters()) + list(raw.body_model.parameters())
        ),
        "checkpoint": (
            "released_ardy_core"
            if args.checkpoint is None
            else str(Path(args.checkpoint).resolve())
        ),
        "normalization_stats": str(Path(dataset_stats_dir).resolve()),
        "sample_ids": [item.get("panel_id", item["id"]) for item in samples],
        "motion_ids": [item["id"] for item in samples],
        "actor_uids": [item.get("actor_uid") for item in samples],
        "panel": str(Path(args.panel).resolve()) if args.panel else None,
        "protocol_lock": str(Path(args.protocol_lock).resolve()) if args.protocol_lock else None,
        "panel_sha256": (
            hashlib.sha256(Path(args.panel).read_bytes()).hexdigest()
            if args.panel else None
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.per_sample_output:
        per_sample_path = Path(args.per_sample_output)
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        per_sample_path.write_text(
            "".join(json.dumps(record) + "\n" for record in per_sample_rows)
        )
    if args.artifacts_output:
        artifact_path = Path(args.artifacts_output)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            generated_joints=np.stack(artifact_generated_joints),
            target_joints=np.stack(artifact_target_joints),
            generated_root=np.stack(artifact_generated_root),
            target_root=np.stack(artifact_target_root),
            sample_ids=np.asarray(artifact_ids),
            texts=np.asarray(artifact_texts),
        )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
