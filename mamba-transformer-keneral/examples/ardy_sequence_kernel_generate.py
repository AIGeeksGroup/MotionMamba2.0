"""Generate ARDY motion in original, DiT, minimal, or hybrid mode.

The released original ARDY checkpoint works only with ``--original``. The
three DiT modes have new mixer parameters and therefore require a checkpoint
trained for this adapter via ``--sequence-checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from ardy_mamba_generate_for_viser import (
    _prepend_python_paths,
    _resolve_output_base,
    _save_motion_npz,
    _select_sample,
    _single_file_path,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARDY_ROOT = Path("/home/plote/mambamotion/basecode/ardy")
TRITON_OVERLAY = os.environ.get("MAMBA_TRITON_OVERLAY", "/tmp/mamba_op_triton")
MAMBA_REPO = os.environ.get(
    "MAMBA_REPO", "/home/plote/mambamotion/basecode/mamba/state-spaces-mamba"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ARDY with original/DiT/minimal/hybrid sequence mixers."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--original", action="store_true", help="Legacy ARDY baseline.")
    modes.add_argument(
        "--dit",
        action="store_true",
        help="DiT modernization baseline; no persistent KV, sinks, or Mamba.",
    )
    modes.add_argument(
        "--minimal",
        action="store_true",
        help="Recommended standard attention baseline; sliding KV + sinks; Mamba off.",
    )
    modes.add_argument(
        "--hybrid",
        action="store_true",
        help="Experimental MotionMamba integration scaffold; Mamba enabled.",
    )
    parser.add_argument("--sequence-checkpoint", help="Checkpoint with root_model/body_model state dicts.")
    parser.add_argument("--attention-backend", choices=["sdpa", "flash2", "flash4"], default="flash2")
    parser.add_argument("--window-size-chunks", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--prompt", default="A person jumps.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--history-frames", type=int, default=None)
    parser.add_argument(
        "--constraints",
        default=None,
        help="Saved ARDY constraint list passed through as motion conditioning.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-weight", type=float, nargs="+", default=[2.0, 2.0])
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument(
        "--stats-dir",
        default=None,
        help="Train-only ARDY dual-root stats used by the sequence checkpoint.",
    )
    parser.add_argument("--ardy-root", default=str(DEFAULT_ARDY_ROOT))
    parser.add_argument("--text-encoder-base-mirror", default=None)
    parser.add_argument("--text-encoder-mntp-adapter", default=None)
    parser.add_argument("--text-encoder-supervised-adapter", default=None)
    parser.add_argument(
        "--text-feature-sample",
        default=None,
        help="Teacher-cache .pt sample containing exact text_feat/mask for its prompt.",
    )
    parser.add_argument("--output", default="outputs/ardy_sequence_kernel")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--launch-viser", action="store_true")
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip optional motion_correction foot-skate cleanup.",
    )
    parser.add_argument("--port", type=int, default=2334)
    return parser.parse_args()


def _mode(args: argparse.Namespace) -> str:
    if args.dit:
        return "dit"
    if args.minimal:
        return "minimal"
    if args.hybrid:
        return "hybrid"
    return "original"


def main() -> int:
    args = parse_args()
    mode = _mode(args)
    ardy_root = Path(args.ardy_root).resolve()
    if not (ardy_root / "scripts" / "visualize.py").exists():
        raise FileNotFoundError(f"ARDY checkout not found or incomplete: {ardy_root}")
    if mode != "original" and args.sequence_checkpoint is None:
        raise ValueError(
            f"--{mode} has new DiT/mixer weights and requires --sequence-checkpoint"
        )
    if mode == "original" and args.sequence_checkpoint is not None:
        raise ValueError("--sequence-checkpoint is only for DiT/minimal/hybrid modes")
    if args.attention_backend == "flash4":
        raise NotImplementedError("FlashAttention-4 is reserved, not implemented")

    python_paths = [ardy_root, REPO_ROOT / "src"]
    if mode == "hybrid":
        python_paths = [TRITON_OVERLAY, MAMBA_REPO, *python_paths]
    _prepend_python_paths(*python_paths)

    from ardy.model import DEFAULT_MODEL, load_model
    from ardy.constraints import load_constraints_lst
    from ardy.model.loading import get_env_var
    from ardy.model.registry import resolve_model_name
    from ardy.motion_rep.tools import length_to_mask
    from ardy.postprocess import post_process_motion
    from ardy.skeleton import SOMASkeleton30
    from ardy.tools import seed_everything, to_numpy

    from ardy_adapter import (
        ArdySequenceKernelConfig,
        ExactArdyTextEncoder,
        install_ardy_sequence_kernel,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("The released ARDY generation path requires CUDA")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if len(args.cfg_weight) == 1:
        cfg_weight = float(args.cfg_weight[0])
    elif len(args.cfg_weight) == 2:
        cfg_weight = (float(args.cfg_weight[0]), float(args.cfg_weight[1]))
    else:
        raise ValueError("--cfg-weight expects one or two values")

    device = "cuda:0"
    torch.empty(1, device=device)
    seed_everything(args.seed)
    checkpoints_dir = args.checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
    resolved_model = resolve_model_name(
        args.model or DEFAULT_MODEL, checkpoints_dir=checkpoints_dir
    )
    encoder_paths = (
        args.text_encoder_base_mirror,
        args.text_encoder_mntp_adapter,
        args.text_encoder_supervised_adapter,
    )
    if any(encoder_paths) and not all(encoder_paths):
        raise ValueError("all three --text-encoder-* paths must be provided together")
    if args.text_feature_sample and any(encoder_paths):
        raise ValueError("--text-feature-sample and --text-encoder-* are exclusive")
    text_encoder = None
    if args.text_feature_sample:
        text_encoder = False
    elif all(encoder_paths):
        text_encoder = ExactArdyTextEncoder(
            base_mirror=args.text_encoder_base_mirror,
            mntp_adapter=args.text_encoder_mntp_adapter,
            supervised_adapter=args.text_encoder_supervised_adapter,
            device=device,
        )
    model = load_model(
        resolved_model,
        device=device,
        checkpoints_dir=checkpoints_dir,
        text_encoder=text_encoder,
    )
    if args.stats_dir:
        from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep

        old_rep = model.motion_rep
        motion_rep = ArdyMotionRep(
            old_rep.skeleton,
            old_rep.fps,
            stats_path=str(Path(args.stats_dir).resolve()),
        )
        motion_rep.skeleton = old_rep.skeleton
        model.motion_rep = motion_rep
        raw_denoiser = model.denoiser.model
        raw_denoiser.motion_rep = motion_rep
        model.autoencoder._motion_rep = motion_rep
        for converter in (model.hybrid, raw_denoiser.hybrid):
            converter._motion_rep = motion_rep
            converter.__dict__.pop("motion_rep", None)
    client = install_ardy_sequence_kernel(
        model,
        ArdySequenceKernelConfig(
            mode=mode,
            attention_backend=args.attention_backend,
            window_size_chunks=args.window_size_chunks,
            num_sink_tokens=args.sink_tokens,
        ),
    )
    if mode != "original":
        checkpoint_path = Path(args.sequence_checkpoint).resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        raw = model.denoiser.model
        raw.root_model.load_state_dict(checkpoint["root_model"], strict=True)
        raw.body_model.load_state_dict(checkpoint["body_model"], strict=True)
        raw.root_model.eval()
        raw.body_model.eval()
        saved_mode = checkpoint.get("mode")
        if saved_mode is not None and saved_mode != mode:
            raise ValueError(
                f"checkpoint mode is {saved_mode!r}, requested {mode!r}"
            )

    fps = model.motion_rep.fps
    num_frames = int(args.duration * fps)
    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = args.diffusion_steps or num_base_steps
    if not 1 <= diffusion_steps <= num_base_steps:
        raise ValueError("invalid --diffusion-steps")
    history_frames = args.history_frames
    if history_frames is None:
        patch = model.num_frames_per_token
        max_window = (int(10 * fps) // patch) * patch
        history_frames = ((max_window - model.gen_horizon_len) // patch) * patch

    constraint_lst = []
    if args.constraints:
        constraint_lst = load_constraints_lst(args.constraints, model.skeleton)
        max_frame_idx = max(int(c.frame_indices.max()) for c in constraint_lst)
        if max_frame_idx >= num_frames:
            raise ValueError(
                f"constraint frame index {max_frame_idx} exceeds requested motion length {num_frames}"
            )

    texts = [args.prompt.strip()] * args.num_samples
    text_feat = None
    text_feat_pad_mask = None
    if args.text_feature_sample:
        text_sample = torch.load(
            Path(args.text_feature_sample).resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if text_sample["prompt"].strip() != args.prompt.strip():
            raise ValueError(
                "--text-feature-sample prompt does not match --prompt: "
                f"{text_sample['prompt']!r} != {args.prompt!r}"
            )
        text_feat = text_sample["text_feat"].to(device).repeat(
            args.num_samples, 1, 1
        )
        text_feat_pad_mask = text_sample["text_feat_pad_mask"].to(device).repeat(
            args.num_samples, 1
        )
    lengths = torch.tensor([num_frames] * args.num_samples, device=device)
    pad_mask = length_to_mask(lengths)
    heading = torch.zeros(args.num_samples, device=device)
    observed_motion = None
    motion_mask = None
    if constraint_lst:
        observed_motion, motion_mask = (
            model.motion_rep.create_conditions_from_constraints_batched(
                constraint_lst,
                lengths,
                to_normalize=True,
                device=device,
            )
        )
    print(
        f"Generating mode={mode}, backend={args.attention_backend}, "
        f"frames={num_frames}, steps={diffusion_steps}"
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        motion = client(
            texts,
            num_frames,
            num_denoising_steps=diffusion_steps,
            pad_mask=pad_mask,
            first_heading_angle=heading,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=cfg_weight,
            crop_history_length=history_frames,
            text_feat=text_feat,
            text_pad_mask=text_feat_pad_mask,
        )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - started
        output = model.motion_rep.inverse(motion, is_normalized=True)

    generation_metrics = {
        "mode": mode,
        "prompt": args.prompt,
        "seed": args.seed,
        "frames": num_frames,
        "chunks": int((num_frames + model.gen_horizon_len - 1) // model.gen_horizon_len),
        "diffusion_steps": diffusion_steps,
        "generation_seconds": generation_seconds,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    if client.controller is not None:
        trace = client.controller.trace
        generation_metrics["memory_trace"] = {
            "completed_windows": trace.completed_windows,
            "accepted_commits": trace.accepted_commits,
            "last_sampling_timestep": trace.last_sampling_timestep,
        }
        raw = model.denoiser.model
        for stage, block in (("root", raw.root_model), ("body", raw.body_model)):
            if block._sequence_state is None:
                generation_metrics[f"{stage}_cache"] = {
                    "num_chunks": 0,
                    "chunk_ids": [],
                    "sink_count": args.sink_tokens if mode in {"minimal", "hybrid"} else 0,
                    "oldest": None,
                    "newest": None,
                }
            else:
                cache = block._sequence_state.layers[0].attention_cache
                generation_metrics[f"{stage}_cache"] = {
                    "num_chunks": cache.num_chunks,
                    "chunk_ids": list(cache.chunk_ids),
                    "sink_count": cache.num_sink_tokens,
                    "oldest": cache.oldest_chunk_id,
                    "newest": cache.newest_chunk_id,
                }

    if not args.no_postprocess and "g1" not in resolved_model.lower():
        output.update(
            post_process_motion(
                output["local_rot_mats"],
                output["root_positions"],
                output["foot_contacts"],
                model.skeleton,
                constraint_lst=constraint_lst or None,
            )
        )
    if isinstance(model.skeleton, SOMASkeleton30):
        output = model.skeleton.output_to_SOMASkeleton77(output)
    output = to_numpy(output)

    output_base = _resolve_output_base(args.output)
    if args.num_samples == 1:
        output_path = _single_file_path(output_base, ".npz")
        _save_motion_npz(
            output_path,
            _select_sample(output, 0, args.num_samples),
            fps,
            args.prompt,
        )
        visualize_path = output_path
    else:
        output_dir = output_base.with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.num_samples):
            output_path = output_dir / f"{output_dir.name}_{index:02d}.npz"
            _save_motion_npz(
                output_path,
                _select_sample(output, index, args.num_samples),
                fps,
                args.prompt,
            )
        visualize_path = output_dir
    print(f"Saved {mode} motion: {visualize_path}")
    generation_metrics["output"] = str(visualize_path)
    print(json.dumps(generation_metrics, indent=2))
    if args.metrics_output:
        metrics_path = Path(args.metrics_output).resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(generation_metrics, indent=2) + "\n")

    if args.launch_viser:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ardy_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        return subprocess.run(
            [
                sys.executable,
                str(ardy_root / "scripts" / "visualize.py"),
                str(visualize_path),
                "--port",
                str(args.port),
            ],
            env=env,
            check=False,
        ).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
