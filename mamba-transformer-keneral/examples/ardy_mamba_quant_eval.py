"""Quantitative teacher-vs-ArdyMambaBlock evaluation for Milestone C.

This script keeps ARDY's diffusion/model path intact, swaps only the root/body
denoiser blocks, and compares generated motion outputs from:

- the original transformer teacher;
- the trained ArdyMambaBlock checkpoint.

It records simple but useful first-pass metrics over multiple prompts:
joint-position L2 error, root-position L2 error, finite checks, wall-clock
latency, and peak CUDA memory. This is not a final motion-quality benchmark;
it is the next sanity layer after the single-prompt viser qualitative check.

Run on the server:

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data3/zts/cache/huggingface \
MAMBA_TRITON_OVERLAY=/data3/zts/packages/mamba_op_triton \
MAMBA_REPO=/data3/zts/packages/state-spaces-mamba \
PYTHONPATH=/data3/zts/ardy \
/data3/zts/conda_envs/ardy_env/bin/python examples/ardy_mamba_quant_eval.py \
    --checkpoint /data3/zts/results/ardy_mamba_distill/mamba_blocks_step1999.pt \
    --out-dir /data3/zts/results/ardy_mamba_quant_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

TRITON_OVERLAY = os.environ.get("MAMBA_TRITON_OVERLAY", "/tmp/mamba_op_triton")
MAMBA_REPO = os.environ.get("MAMBA_REPO", "/home/plote/mambamotion/basecode/mamba/state-spaces-mamba")

DEFAULT_PROMPTS = [
    "A person jumps.",
    "A person dances happily.",
    "A person walks forward.",
    "A person turns around and waves.",
    "A person squats down and stands up.",
    "A person kicks with one leg.",
    "A person jogs in place.",
    "A person steps sideways and turns.",
    "A person raises both arms.",
    "A person walks in a circle.",
    "A person bends down then stands upright.",
    "A person takes several quick steps backward.",
]


def _load_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = [line.strip() for line in Path(path).read_text().splitlines()]
    prompts = [p for p in prompts if p and not p.startswith("#")]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _squeeze_batch(output: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v)
        for k, v in output.items()
    }


def _finite_dict(output: dict[str, Any]) -> bool:
    for value in output.values():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            if not np.isfinite(value).all():
                return False
    return True


def _mean_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=-1).mean())


def _max_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=-1).max())


def _metrics_for_outputs(teacher: dict[str, Any], mamba: dict[str, Any]) -> dict[str, float | bool]:
    metrics: dict[str, float | bool] = {
        "teacher_finite": _finite_dict(teacher),
        "mamba_finite": _finite_dict(mamba),
    }
    if "posed_joints" in teacher and "posed_joints" in mamba:
        metrics["posed_joints_mean_l2"] = _mean_l2(teacher["posed_joints"], mamba["posed_joints"])
        metrics["posed_joints_max_l2"] = _max_l2(teacher["posed_joints"], mamba["posed_joints"])
    if "root_positions" in teacher and "root_positions" in mamba:
        metrics["root_positions_mean_l2"] = _mean_l2(teacher["root_positions"], mamba["root_positions"])
        metrics["root_positions_max_l2"] = _max_l2(teacher["root_positions"], mamba["root_positions"])
    if "foot_contacts" in teacher and "foot_contacts" in mamba:
        teacher_contacts = teacher["foot_contacts"].astype(np.float64)
        mamba_contacts = mamba["foot_contacts"].astype(np.float64)
        metrics["foot_contact_mean_abs_diff"] = float(np.abs(teacher_contacts - mamba_contacts).mean())
    return metrics


def _save_npz(path: Path, output: dict[str, Any], fps: float, text: str) -> None:
    arrays = {k: np.asarray(v) for k, v in output.items()}
    arrays["fps"] = np.asarray(fps)
    arrays["text"] = np.asarray(text)
    np.savez(path, **arrays)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"prompt_index", "seed"}
        }
    )
    summary: dict[str, Any] = {"num_prompts": len(rows)}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        if values.size:
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_max"] = float(values.max())
    summary["all_teacher_finite"] = bool(all(row.get("teacher_finite", False) for row in rows))
    summary["all_mamba_finite"] = bool(all(row.get("mamba_finite", False) for row in rows))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="/data3/zts/results/ardy_mamba_quant_eval")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--no-warmup", action="store_true", help="Disable one unmeasured teacher/Mamba warmup run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        return 2

    sys.path.insert(0, TRITON_OVERLAY)
    sys.path.insert(0, MAMBA_REPO)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from ardy.model import DEFAULT_MODEL, load_model
    from ardy.model.loading import get_env_var
    from ardy.model.registry import resolve_model_name
    from ardy.motion_rep.tools import length_to_mask
    from ardy.tools import seed_everything, to_numpy

    from ardy_adapter import ArdyMambaBlock

    device = "cuda:0"
    prompts = _load_prompts(args.prompts_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = get_env_var("CHECKPOINTS_DIR")
    resolved_model = resolve_model_name(DEFAULT_MODEL, checkpoints_dir=checkpoints_dir)
    print(f"Loading {resolved_model} ...")
    model = load_model(resolved_model, device=device, checkpoints_dir=checkpoints_dir)
    print("Loaded.")

    raw_denoiser = model.denoiser.model
    common_kwargs = dict(
        input_dim=-1,
        add_input_proj=False,
        skeleton=raw_denoiser.motion_rep.skeleton,
        llm_shape=raw_denoiser.llm_shape,
        use_text_mask=raw_denoiser.root_model.use_text_mask,
        latent_dim=raw_denoiser.latent_dim,
        num_layers=8,
        input_first_heading_angle=hasattr(raw_denoiser.root_model, "linear_first_heading_angle"),
        positional_encoding_mode=raw_denoiser.positional_encoding_mode,
        device=device,
    )
    mamba_root = ArdyMambaBlock(output_dim=raw_denoiser.root_model.output_linear.out_features, **common_kwargs).to(device)
    mamba_body = ArdyMambaBlock(output_dim=raw_denoiser.body_model.output_linear.out_features, **common_kwargs).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    mamba_root.load_state_dict(ckpt["root_model"])
    mamba_body.load_state_dict(ckpt["body_model"])
    mamba_root.eval()
    mamba_body.eval()
    print(f"Loaded trained checkpoint from step {ckpt.get('step', 'unknown')}.")

    original_root, original_body = raw_denoiser.root_model, raw_denoiser.body_model
    gen_horizon_len = model.gen_horizon_len
    fps = model.motion_rep.fps
    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = args.diffusion_steps or num_base_steps

    def generate(prompt: str, root_model, body_model, seed: int) -> tuple[dict[str, Any], float, int]:
        raw_denoiser.root_model = root_model
        raw_denoiser.body_model = body_model
        seed_everything(seed)
        lengths = torch.tensor([gen_horizon_len], device=device)
        pad_mask = length_to_mask(lengths)
        first_heading_angle = torch.zeros(1, device=device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            motion = model(
                [prompt],
                gen_horizon_len,
                num_denoising_steps=diffusion_steps,
                pad_mask=pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=None,
                observed_motion=None,
                cfg_weight=(2.0, 2.0),
                crop_history_length=None,
            )
            output = model.motion_rep.inverse(motion, is_normalized=True)
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - start
        peak_bytes = int(torch.cuda.max_memory_allocated())
        return _squeeze_batch(to_numpy(output)), latency_s, peak_bytes

    rows: list[dict[str, Any]] = []
    try:
        if not args.no_warmup:
            warmup_prompt = prompts[0]
            warmup_seed = args.seed + 100000
            print(f"Warmup, not recorded: {warmup_prompt!r}")
            generate(warmup_prompt, original_root, original_body, warmup_seed)
            generate(warmup_prompt, mamba_root, mamba_body, warmup_seed)

        for i, prompt in enumerate(prompts):
            seed = args.seed + i
            print(f"[{i + 1}/{len(prompts)}] {prompt!r}")
            teacher_output, teacher_latency, teacher_peak = generate(prompt, original_root, original_body, seed)
            mamba_output, mamba_latency, mamba_peak = generate(prompt, mamba_root, mamba_body, seed)
            row: dict[str, Any] = {
                "prompt_index": i,
                "prompt": prompt,
                "seed": seed,
                "teacher_latency_s": teacher_latency,
                "mamba_latency_s": mamba_latency,
                "teacher_peak_memory_bytes": teacher_peak,
                "mamba_peak_memory_bytes": mamba_peak,
                "latency_ratio_mamba_over_teacher": mamba_latency / teacher_latency if teacher_latency > 0 else float("nan"),
            }
            row.update(_metrics_for_outputs(teacher_output, mamba_output))
            rows.append(row)

            if args.save_npz:
                stem = f"{i:02d}"
                _save_npz(out_dir / f"teacher_{stem}.npz", teacher_output, fps, prompt)
                _save_npz(out_dir / f"trained_mamba_{stem}.npz", mamba_output, fps, prompt)
    finally:
        raw_denoiser.root_model, raw_denoiser.body_model = original_root, original_body

    summary = _summarize(rows)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": ckpt.get("step", None),
        "model": resolved_model,
        "num_base_steps": num_base_steps,
        "diffusion_steps": diffusion_steps,
        "gen_horizon_len": int(gen_horizon_len),
        "fps": float(fps),
        "summary": summary,
        "rows": rows,
    }
    json_path = out_dir / "metrics.json"
    csv_path = out_dir / "metrics.csv"
    json_path.write_text(json.dumps(result, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics: {json_path}")
    print(f"Saved metrics: {csv_path}")
    print(json.dumps(summary, indent=2))
    print("ARDY_MAMBA_QUANT_EVAL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
