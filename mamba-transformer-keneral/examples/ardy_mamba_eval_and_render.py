"""Milestone C follow-up: load the trained ArdyMambaBlock checkpoint, run
REAL multi-step generation (the full `model.__call__` / DDIM path, not just
the single training objective) on a training prompt AND a held-out
(unseen) prompt, compare against the original transformer teacher on the
same prompts, and render each result to an MP4 (stick-figure skeleton
animation from `posed_joints` + skeleton parent links) so results are
viewable without needing the interactive viser viewer.

Run with (on the server):

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data3/zts/cache/huggingface \
MAMBA_TRITON_OVERLAY=/data3/zts/packages/mamba_op_triton \
MAMBA_REPO=/data3/zts/packages/state-spaces-mamba \
PYTHONPATH=/data3/zts/ardy \
/data3/zts/conda_envs/ardy_env/bin/python \
    examples/ardy_mamba_eval_and_render.py \
    --checkpoint /data3/zts/results/ardy_mamba_distill/mamba_blocks_step1999.pt \
    --out-dir /data3/zts/results/ardy_mamba_eval
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

TRITON_OVERLAY = os.environ.get("MAMBA_TRITON_OVERLAY", "/tmp/mamba_op_triton")
MAMBA_REPO = os.environ.get("MAMBA_REPO", "/home/plote/mambamotion/basecode/mamba/state-spaces-mamba")

TRAIN_PROMPT = "A person jumps."          # was in the 16-prompt distillation set
HELDOUT_PROMPT = "A person dances happily."  # NOT in the distillation set


def render_mp4(posed_joints: np.ndarray, joint_parents: np.ndarray, fps: float, out_path: Path, title: str) -> None:
    """posed_joints: (T, J, 3). joint_parents: (J,) int, -1 for root."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = posed_joints.shape[0]
    # Fixed axis limits across all frames so the animation doesn't jitter-zoom.
    mins = posed_joints.reshape(-1, 3).min(axis=0)
    maxs = posed_joints.reshape(-1, 3).max(axis=0)
    center = (mins + maxs) / 2
    half_range = max((maxs - mins).max() / 2, 0.3)

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        joints = posed_joints[t]
        for j, p in enumerate(joint_parents):
            if p < 0:
                continue
            xs = [joints[j, 0], joints[p, 0]]
            ys = [joints[j, 1], joints[p, 1]]
            zs = [joints[j, 2], joints[p, 2]]
            ax.plot(xs, zs, ys, color="#2a78d6", linewidth=2)
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], color="#e34948", s=8)
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[2] - half_range, center[2] + half_range)
        ax.set_zlim(center[1] - half_range, center[1] + half_range)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(buf)
        plt.close(fig)

    imageio.mimwrite(str(out_path), frames, fps=fps, macro_block_size=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="/data3/zts/results/ardy_mamba_eval")
    parser.add_argument("--diffusion-steps", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        return 2
    torch.empty(1, device="cuda")

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
    checkpoints_dir = get_env_var("CHECKPOINTS_DIR")
    resolved_model = resolve_model_name(DEFAULT_MODEL, checkpoints_dir=checkpoints_dir)
    print(f"Loading {resolved_model} ...")
    model = load_model(resolved_model, device=device, checkpoints_dir=checkpoints_dir)
    print("Loaded.")

    raw_denoiser = model.denoiser.model  # NOT model.denoiser -- see Milestone B correction
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
    print(f"Loaded trained checkpoint from step {ckpt['step']}.")

    original_root, original_body = raw_denoiser.root_model, raw_denoiser.body_model

    gen_horizon_len = model.gen_horizon_len
    fps = model.motion_rep.fps
    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = args.diffusion_steps or num_base_steps

    def generate(prompt: str):
        seed_everything(0)
        lengths = torch.tensor([gen_horizon_len], device=device)
        pad_mask = length_to_mask(lengths)
        first_heading_angle = torch.zeros(1, device=device)
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
        return to_numpy(output)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skeleton = model.skeleton
    joint_parents = skeleton.joint_parents.cpu().numpy()

    variants = [
        ("teacher", original_root, original_body),
        ("trained_mamba", mamba_root, mamba_body),
    ]
    prompts = [("train_prompt", TRAIN_PROMPT), ("heldout_prompt", HELDOUT_PROMPT)]

    for variant_name, root_m, body_m in variants:
        raw_denoiser.root_model = root_m
        raw_denoiser.body_model = body_m
        assert model.denoiser.model.root_model is root_m
        for prompt_name, prompt_text in prompts:
            print(f"Generating [{variant_name}] x [{prompt_name}]: {prompt_text!r}")
            output = generate(prompt_text)
            # Squeeze the batch dim like scripts/generate.py's _select_sample
            # does before saving -- ARDY's own scripts/visualize.py expects
            # per-sample arrays (posed_joints: (T, J, 3)), not a batch axis
            # (T here would be misread as a joint count and crash the
            # skeleton lookup).
            output = {
                k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v)
                for k, v in output.items()
            }
            stem = f"{variant_name}_{prompt_name}"
            npz_path = out_dir / f"{stem}.npz"
            arrays = {k: np.asarray(v) for k, v in output.items()}
            arrays["fps"] = np.asarray(fps)
            arrays["text"] = np.asarray(prompt_text)
            np.savez(npz_path, **arrays)
            mp4_path = out_dir / f"{stem}.mp4"
            render_mp4(output["posed_joints"], joint_parents, fps, mp4_path, f"{variant_name}: {prompt_text}")
            print(f"  saved {npz_path.name}, {mp4_path.name}")

    raw_denoiser.root_model, raw_denoiser.body_model = original_root, original_body
    print("EVAL_AND_RENDER_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
