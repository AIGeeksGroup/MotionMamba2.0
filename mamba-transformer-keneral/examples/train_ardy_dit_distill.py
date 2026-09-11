"""Phase-A ARDY Transformer-to-DiT distillation smoke/overfit gate.

This entrypoint intentionally trains only the new root/body DiT blocks with
cross-window memory disabled. It is an engineering gate, not a substitute for
the unreleased real-motion training dataset or the later diffusion objective.
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--teacher-revision", default="unknown")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE_OR_UNCOMMITTED_COPY"


def _capture_block_io(raw, call):
    captured = {}

    def save(name):
        def hook(_module, inputs, output):
            captured[name] = {
                "inputs": tuple(
                    value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for value in inputs
                ),
                "target": output.detach().clone(),
            }

        return hook

    handles = [
        raw.root_model.register_forward_hook(save("root")),
        raw.body_model.register_forward_hook(save("body")),
    ]
    try:
        with torch.no_grad():
            prediction = raw(**call)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"root", "body"}:
        raise RuntimeError("teacher root/body hooks did not both execute exactly once")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("teacher prediction is non-finite")
    return captured, prediction.detach()


def _synthetic_teacher_batch(model, device: torch.device) -> dict:
    """One deterministic shape-correct gate batch, not claimed as real data."""
    raw = model.denoiser.model
    batch_size = 1
    history_tokens = 2
    frames_per_token = int(raw.num_frames_per_token)
    generation_tokens = int(model.gen_horizon_len // frames_per_token)
    num_tokens = history_tokens + generation_tokens
    num_frames = num_tokens * frames_per_token
    history_frames = history_tokens * frames_per_token
    token_dim = int(raw.nframe_root_dim + raw.latent_embedding_dim)
    history_token_mask = torch.zeros(
        batch_size, num_tokens, dtype=torch.bool, device=device
    )
    history_token_mask[:, :history_tokens] = True
    generation_token_mask = ~history_token_mask
    history_mask = torch.zeros(
        batch_size, num_frames, dtype=torch.bool, device=device
    )
    history_mask[:, :history_frames] = True
    generation_mask = ~history_mask
    clean = torch.randn(batch_size, num_tokens, token_dim, device=device)
    timestep = torch.full(
        (batch_size,), model.diffusion.num_base_steps - 2, dtype=torch.long, device=device
    )
    noisy = model.diffusion.q_sample(clean, timestep)
    return {
        "x": noisy,
        "history_len": torch.full(
            (batch_size,), history_frames, dtype=torch.long, device=device
        ),
        "generation_len": torch.full(
            (batch_size,), num_frames - history_frames, dtype=torch.long, device=device
        ),
        "future_len": torch.zeros(batch_size, dtype=torch.long, device=device),
        "history_mask": history_mask,
        "generation_mask": generation_mask,
        "future_mask": torch.zeros_like(history_mask),
        "history_token_mask": history_token_mask,
        "generation_token_mask": generation_token_mask,
        "future_token_mask": torch.zeros_like(history_token_mask),
        "text_feat": torch.randn(
            batch_size, raw.llm_shape[0], raw.llm_shape[-1], device=device
        ),
        "text_feat_pad_mask": torch.ones(
            batch_size, raw.llm_shape[0], dtype=torch.bool, device=device
        ),
        "timesteps": timestep,
        "first_heading_angle": torch.zeros(batch_size, device=device),
        "motion_mask": None,
        "observed_motion": None,
    }


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    ardy_root = Path(args.ardy_root).resolve()
    sys.path.insert(0, str(ardy_root))
    from ardy.model import load_model

    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    teacher = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )
    teacher.eval()
    teacher_call = _synthetic_teacher_batch(teacher, device)
    captured, teacher_prediction = _capture_block_io(
        teacher.denoiser.model, teacher_call
    )
    teacher_forward_mse = teacher_prediction.float().square().mean().item()
    del teacher, teacher_call, teacher_prediction
    if device.type == "cuda":
        torch.cuda.empty_cache()

    student = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )
    client = install_ardy_sequence_kernel(
        student,
        ArdySequenceKernelConfig(
            mode="dit",
            attention_backend="sdpa",
            window_size_chunks=0,
            num_sink_tokens=2,
        ),
    )
    raw = student.denoiser.model
    for parameter in raw.parameters():
        parameter.requires_grad_(False)
    trainable = list(raw.root_model.parameters()) + list(raw.body_model.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    losses = []
    tracked = next(raw.root_model.sequence_stack.parameters())
    tracked_before = tracked.detach().clone()

    for _step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            root_prediction = raw.root_model(*captured["root"]["inputs"])
            body_prediction = raw.body_model(*captured["body"]["inputs"])
            loss = F.mse_loss(
                root_prediction.float(), captured["root"]["target"].float()
            ) + F.mse_loss(
                body_prediction.float(), captured["body"]["target"].float()
            )
        if not torch.isfinite(loss):
            raise RuntimeError("distillation loss became non-finite")
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
            raise RuntimeError("missing or non-finite student gradients")
        optimizer.step()
        # Projected K/V must never cross an optimizer step. DiT has W=0 here,
        # but resetting makes the lifetime rule explicit and future-proof.
        raw.root_model.reset_memory()
        raw.body_model.reset_memory()
        losses.append(float(loss.detach()))

    parameter_changed = not torch.equal(tracked_before, tracked.detach())
    if not parameter_changed:
        raise RuntimeError("student parameter did not change")
    relative_reduction = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "mode": "dit",
        "root_model": raw.root_model.state_dict(),
        "body_model": raw.body_model.state_dict(),
        "training": {
            "phase": "A_distillation_smoke",
            "teacher_model": args.teacher_model,
            "teacher_revision": args.teacher_revision,
            "seed": args.seed,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "precision": args.precision,
            "git_commit": _git_commit(),
            "data": "single deterministic synthetic shape-correct batch",
        },
    }
    torch.save(checkpoint, output_path)

    # Destructive-in-memory round trip: zero one parameter, reload the saved
    # state, and require exact restoration.
    saved_state = torch.load(output_path, map_location=device, weights_only=False)
    tracked_key, tracked_saved = next(iter(saved_state["root_model"].items()))
    tracked_module_state = raw.root_model.state_dict()
    tracked_module_state[tracked_key].zero_()
    raw.root_model.load_state_dict(saved_state["root_model"], strict=True)
    raw.body_model.load_state_dict(saved_state["body_model"], strict=True)
    reload_exact = torch.equal(
        raw.root_model.state_dict()[tracked_key], tracked_saved.to(device)
    )
    if not reload_exact:
        raise RuntimeError("checkpoint reload did not restore the saved parameter")

    minimal_module_names = [type(module).__name__.lower() for module in raw.modules()]
    no_mamba_module = not any("mamba" in name for name in minimal_module_names)
    no_mamba_import = "mamba_ssm" not in sys.modules
    if not (no_mamba_module and no_mamba_import):
        raise RuntimeError("DiT-only smoke unexpectedly imported or instantiated Mamba")

    elapsed = time.perf_counter() - started
    peak_mib = (
        torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else None
    )
    report = {
        "status": "PASS",
        "phase": "A_distillation_smoke",
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "teacher_forward_mean_square": teacher_forward_mse,
        "initial_mse": losses[0],
        "final_mse": losses[-1],
        "relative_reduction": relative_reduction,
        "losses": losses,
        "finite_gradients": True,
        "parameter_changed": parameter_changed,
        "checkpoint_saved": str(output_path),
        "checkpoint_reload_exact": reload_exact,
        "no_mamba_module": no_mamba_module,
        "mamba_ssm_not_imported": no_mamba_import,
        "peak_allocated_mib": peak_mib,
        "elapsed_seconds": elapsed,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "precision": args.precision,
        "batch_size": 1,
        "steps": args.steps,
        "warning": "synthetic engineering gate; not real-motion training evidence",
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.metrics_output:
        metrics_path = Path(args.metrics_output).resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(encoded + os.linesep, encoding="utf-8")
    client.reset_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
