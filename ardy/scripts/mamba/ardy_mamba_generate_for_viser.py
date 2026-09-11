"""Generate ARDY motion with trained ArdyMambaBlock and open ARDY's viser viewer.

This is the report-time path:

1. Load the released ARDY model.
2. Replace the denoiser root/body TransformerEncoderBlock modules with the
   trained ArdyMambaBlock checkpoint.
3. Generate a normal ARDY `.npz` motion file.
4. Optionally launch `basecode/ardy/scripts/visualize.py` on that `.npz`.

Example:

CUDA_VISIBLE_DEVICES=0 \
CHECKPOINTS_DIR=/path/to/ardy/checkpoints \
MAMBA_TRITON_OVERLAY=/path/to/mamba_op_triton \
MAMBA_REPO=/path/to/state-spaces-mamba \
python examples/ardy_mamba_generate_for_viser.py \
    --checkpoint /path/to/mamba_blocks_step1999.pt \
    --prompt "A person jumps." \
    --output /tmp/ardy_mamba_vis/jump \
    --launch-viser --port 2334
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARDY_ROOT = REPO_ROOT
TRITON_OVERLAY = os.environ.get("MAMBA_TRITON_OVERLAY", "/tmp/mamba_op_triton")
MAMBA_REPO = os.environ.get("MAMBA_REPO", "")


def _prepend_python_paths(*paths: str | Path) -> None:
    for path in reversed([str(p) for p in paths if p]):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _resolve_output_base(path: str, default_dir: str = "outputs") -> Path:
    p = Path(path)
    if p.parent == Path("."):
        return Path(default_dir) / p
    return p


def _single_file_path(path: Path, ext: str) -> Path:
    if path.suffix != ext:
        path = path.with_suffix(ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _select_sample(output: dict, index: int, n_samples: int) -> dict:
    return {
        k: (v[index] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
        for k, v in output.items()
    }


def _save_motion_npz(path: Path, motion_dict: dict, fps: float, text: str) -> None:
    arrays = {k: np.asarray(v) for k, v in motion_dict.items()}
    arrays["fps"] = np.asarray(fps)
    arrays["text"] = np.asarray(text)
    np.savez(path, **arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate trained ARDY-Mamba motion and visualize with ARDY viser.")
    parser.add_argument("--checkpoint", required=True, help="Trained mamba_blocks_step*.pt checkpoint.")
    parser.add_argument("--prompt", default="A person jumps.", help="Text prompt to generate.")
    parser.add_argument("--model", default=None, help="ARDY model nickname/full name. Defaults to ARDY DEFAULT_MODEL.")
    parser.add_argument("--duration", type=float, default=5.0, help="Motion duration in seconds.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of identical-prompt samples.")
    parser.add_argument("--diffusion-steps", type=int, default=None, help="Number of DDIM denoising steps.")
    parser.add_argument("--history-frames", type=int, default=None, help="Autoregressive visible history length.")
    parser.add_argument("--seed", type=int, default=0, help="Generation seed.")
    parser.add_argument("--cfg-weight", type=float, nargs="+", default=[2.0, 2.0])
    parser.add_argument("--checkpoints-dir", default=None, help="Released ARDY checkpoint directory.")
    parser.add_argument("--ardy-root", default=str(DEFAULT_ARDY_ROOT), help="Path to the ARDY source checkout.")
    parser.add_argument("--output", default="outputs/ardy_mamba_viser", help="Output stem or folder.")
    parser.add_argument("--launch-viser", action="store_true", help="Run ARDY scripts/visualize.py after saving NPZ.")
    parser.add_argument("--port", type=int, default=2334, help="Viser port when --launch-viser is set.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ardy_root = Path(args.ardy_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Mamba checkpoint not found: {checkpoint}")
    if not (ardy_root / "scripts" / "visualize.py").exists():
        raise FileNotFoundError(f"ARDY checkout not found or incomplete: {ardy_root}")

    _prepend_python_paths(TRITON_OVERLAY, MAMBA_REPO, ardy_root, REPO_ROOT / "src")

    from ardy.model import DEFAULT_MODEL, load_model
    from ardy.model.loading import get_env_var
    from ardy.model.registry import resolve_model_name
    from ardy.motion_rep.tools import length_to_mask
    from ardy.postprocess import post_process_motion
    from ardy.skeleton import SOMASkeleton30
    from ardy.tools import seed_everything, to_numpy

    from ardy.model.ardy_mamba_block import ArdyMambaBlock

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1.")
    if len(args.cfg_weight) == 1:
        cfg_weight = float(args.cfg_weight[0])
    elif len(args.cfg_weight) == 2:
        cfg_weight = (float(args.cfg_weight[0]), float(args.cfg_weight[1]))
    else:
        raise ValueError("--cfg-weight expects one float or two floats.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("ARDY-Mamba generation requires CUDA for the current Mamba-3 path.")
    torch.empty(1, device=device)
    seed_everything(args.seed)

    checkpoints_dir = args.checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
    model_name = args.model or DEFAULT_MODEL
    resolved_model = resolve_model_name(model_name, checkpoints_dir=checkpoints_dir)
    print(f"Loading ARDY model {resolved_model} on {device} ...")
    model = load_model(resolved_model, device=device, checkpoints_dir=checkpoints_dir)

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

    ckpt = torch.load(checkpoint, map_location=device)
    mamba_root.load_state_dict(ckpt["root_model"])
    mamba_body.load_state_dict(ckpt["body_model"])
    mamba_root.eval()
    mamba_body.eval()
    raw_denoiser.root_model = mamba_root
    raw_denoiser.body_model = mamba_body
    assert model.denoiser.model.root_model is mamba_root
    assert model.denoiser.model.body_model is mamba_body
    print(f"Swapped trained ArdyMambaBlock checkpoint into ARDY denoiser: step={ckpt.get('step', 'unknown')}")

    fps = model.motion_rep.fps
    num_frames = int(args.duration * fps)
    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = args.diffusion_steps if args.diffusion_steps is not None else num_base_steps
    if not 1 <= diffusion_steps <= num_base_steps:
        raise ValueError(f"--diffusion-steps must be between 1 and {num_base_steps}, got {diffusion_steps}.")

    history_frames = args.history_frames
    if history_frames is None:
        patch = model.num_frames_per_token
        max_window_len = (int(10 * fps) // patch) * patch
        history_frames = ((max_window_len - model.gen_horizon_len) // patch) * patch

    texts = [args.prompt.strip()] * args.num_samples
    lengths = torch.tensor([num_frames] * args.num_samples, device=device)
    pad_mask = length_to_mask(lengths)
    first_heading_angle = torch.zeros(args.num_samples, device=device)

    print(
        f"Generating with Mamba denoiser: prompt={args.prompt!r}, frames={num_frames}, "
        f"fps={fps:g}, diffusion_steps={diffusion_steps}, history_frames={history_frames}"
    )
    with torch.no_grad():
        motion = model(
            texts,
            num_frames,
            num_denoising_steps=diffusion_steps,
            pad_mask=pad_mask,
            first_heading_angle=first_heading_angle,
            motion_mask=None,
            observed_motion=None,
            cfg_weight=cfg_weight,
            crop_history_length=history_frames,
        )
        output = model.motion_rep.inverse(motion, is_normalized=True)

    if "g1" not in resolved_model.lower():
        corrected = post_process_motion(
            output["local_rot_mats"],
            output["root_positions"],
            output["foot_contacts"],
            model.skeleton,
            constraint_lst=None,
        )
        output.update(corrected)
    if isinstance(model.skeleton, SOMASkeleton30):
        output = model.skeleton.output_to_SOMASkeleton77(output)
    output = to_numpy(output)

    output_base = _resolve_output_base(args.output)
    if args.num_samples == 1:
        npz_path = _single_file_path(output_base, ".npz")
        _save_motion_npz(npz_path, _select_sample(output, 0, args.num_samples), fps, args.prompt)
        visualize_path = npz_path
        print(f"Saved Mamba NPZ: {npz_path}")
    else:
        out_dir = output_base.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = out_dir.name
        for i in range(args.num_samples):
            npz_path = out_dir / f"{base_name}_{i:02d}.npz"
            _save_motion_npz(npz_path, _select_sample(output, i, args.num_samples), fps, args.prompt)
            print(f"Saved Mamba NPZ: {npz_path}")
        visualize_path = out_dir

    if args.launch_viser:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([str(ardy_root), existing]) if existing else str(ardy_root)
        cmd = [
            sys.executable,
            str(ardy_root / "scripts" / "visualize.py"),
            str(visualize_path),
            "--port",
            str(args.port),
        ]
        print(f"Launching ARDY viser: http://localhost:{args.port}")
        return subprocess.run(cmd, env=env, check=False).returncode

    print(f"Open with: python {ardy_root / 'scripts' / 'visualize.py'} {visualize_path} --port {args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
