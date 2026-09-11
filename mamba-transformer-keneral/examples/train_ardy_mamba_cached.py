"""Train a real-motion Mamba denoiser control from Stage-05 teacher calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from train_ardy_minimal_cached import TrajectoryDataset, _move  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=["tiny", "train"], required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tiny-cases", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_blocks(raw, num_layers: int, device: torch.device):
    from ardy_adapter import ArdyMambaBlock

    common = dict(
        input_dim=-1,
        add_input_proj=False,
        skeleton=raw.motion_rep.skeleton,
        llm_shape=raw.llm_shape,
        use_text_mask=raw.root_model.use_text_mask,
        latent_dim=raw.latent_dim,
        num_layers=num_layers,
        input_first_heading_angle=hasattr(
            raw.root_model, "linear_first_heading_angle"
        ),
        positional_encoding_mode=raw.positional_encoding_mode,
        mamba_dtype=torch.bfloat16,
        device=device,
    )
    root = ArdyMambaBlock(
        output_dim=raw.root_model.output_linear.out_features, **common
    ).to(device)
    body = ArdyMambaBlock(
        output_dim=raw.body_model.output_linear.out_features, **common
    ).to(device)
    return root, body


def _case(sample: dict[str, Any], index: int) -> dict[str, Any]:
    return sample["calls"][index % len(sample["calls"])]


def _loss(root, body, call: dict[str, Any], device: torch.device):
    root_inputs = _move(call["root"]["inputs"], device)
    body_inputs = _move(call["body"]["inputs"], device)
    root_prediction = root(*root_inputs)
    body_prediction = body(*body_inputs)
    root_loss = F.mse_loss(
        root_prediction.float(), call["root"]["teacher_output"].to(device).float()
    )
    body_loss = F.mse_loss(
        body_prediction.float(), call["body"]["teacher_output"].to(device).float()
    )
    return root_loss + body_loss, root_loss, body_loss


@torch.inference_mode()
def _evaluate(root, body, dataset, device, autocast_enabled) -> dict[str, float]:
    root.eval()
    body.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    count = 0
    diffusion_steps = dataset.manifest["diffusion_steps"]
    selected = {0, diffusion_steps // 2, diffusion_steps - 1}
    for sample in dataset.samples:
        for call_index, call in enumerate(sample["calls"]):
            if call_index % diffusion_steps not in selected:
                continue
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                total, root_loss, body_loss = _loss(root, body, call, device)
            if not torch.isfinite(total):
                raise RuntimeError("non-finite Mamba development loss")
            totals += torch.tensor(
                [float(total), float(root_loss), float(body_loss)], dtype=torch.float64
            )
            count += 1
    return {
        "loss": float(totals[0] / count),
        "root_loss": float(totals[1] / count),
        "body_loss": float(totals[2] / count),
        "cases": count,
    }


@torch.inference_mode()
def _evaluate_calls(root, body, calls, device, autocast_enabled) -> float:
    root.eval()
    body.eval()
    losses = []
    for call in calls:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            total, _, _ = _loss(root, body, call, device)
        losses.append(float(total))
    return sum(losses) / len(losses)


@torch.inference_mode()
def _state_gate(block, call: dict[str, Any], device: torch.device) -> dict[str, float]:
    inputs = _move(call["root"]["inputs"], device)
    x, x_mask, text, text_mask, timesteps, heading, token_index = inputs
    # Reproduce the block input up to the first Mamba layer, then explicitly
    # verify recurrent carry with the existing wrapper.
    x = block.input_linear(x)
    text, text_mask = block._pad_x_and_mask_to_fixed_size(
        text, text_mask, block.num_text_tokens
    )
    prefix = [block.embed_text(text.to(block.embed_text.weight.dtype)),
              block.embed_timestep(timesteps)]
    masks = [text_mask if block.use_text_mask else torch.ones_like(text_mask),
             torch.ones((x.shape[0], 1), dtype=torch.bool, device=device)]
    if block.input_first_heading_angle:
        angle = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        prefix.append(block.linear_first_heading_angle(angle)[:, None])
        masks.append(torch.ones((x.shape[0], 1), dtype=torch.bool, device=device))
    prefix_tensor = torch.cat(prefix, dim=1)
    if block.positional_encoding_mode == "default":
        hidden = block.sequence_pos_encoder(torch.cat((prefix_tensor, x), dim=1))
    else:
        hidden = torch.cat(
            (block.learned_prefix_embedding(prefix_tensor),
             block.motion_token_embedding(x, token_index)), dim=1
        )
    valid = torch.cat((*masks, x_mask), dim=1)
    hidden = (hidden * valid[:, :, None]).to(block._mamba_dtype)
    wrapper = block._mamba_wrappers[0]
    out1, state1 = wrapper.forward_chunk(hidden, None)
    out2, state2 = wrapper.forward_chunk(hidden, state1)
    state = state1.ssm_state.float()
    carry_effect = (out2.float() - out1.float()).abs().mean()
    return {
        "ssm_state_variance": float(state.var()),
        "ssm_state_norm": float(state.norm(dim=(-2, -1)).mean()),
        "carry_mean_absolute_effect": float(carry_effect),
        "next_state_norm": float(state2.ssm_state.float().norm(dim=(-2, -1)).mean()),
    }


def _checkpoint(root, body, args, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "mode": "mamba",
        "root_model": root.state_dict(),
        "body_model": body.state_dict(),
        "training": {
            "phase": args.phase,
            "teacher_model": args.teacher_model,
            "train_manifest": str(Path(args.train_manifest).resolve()),
            "dev_manifest": str(Path(args.dev_manifest).resolve()),
            "train_manifest_sha256": _sha256(args.train_manifest),
            "dev_manifest_sha256": _sha256(args.dev_manifest),
            "num_layers": args.num_layers,
            "seed": args.seed,
            "git_commit": _git_commit(),
            "metrics": metrics,
        },
    }


def _reload_gate(raw, checkpoint_path: Path, args, call, device) -> float:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    root, body = _make_blocks(raw, args.num_layers, device)
    root.load_state_dict(checkpoint["root_model"], strict=True)
    body.load_state_dict(checkpoint["body_model"], strict=True)
    root.eval()
    body.eval()
    with torch.inference_mode():
        total, _, _ = _loss(root, body, call, device)
    return float(total)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "git_commit": _git_commit()}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    sys.path.insert(0, str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model

    model = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )
    raw = model.denoiser.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    root, body = _make_blocks(raw, args.num_layers, device)
    train = TrajectoryDataset(args.train_manifest, None)
    dev = TrajectoryDataset(args.dev_manifest, None)
    parameters = list(root.parameters()) + list(body.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"

    if args.phase == "tiny":
        tiny = []
        for sample in train.samples:
            for call in sample["calls"]:
                tiny.append(call)
                if len(tiny) == args.tiny_cases:
                    break
            if len(tiny) == args.tiny_cases:
                break
        if len(tiny) < args.tiny_cases:
            raise ValueError("not enough cached calls for tiny overfit")
    else:
        tiny = None

    initial_dev = _evaluate(root, body, dev, device, autocast_enabled)
    initial_tiny = (
        _evaluate_calls(root, body, tiny, device, autocast_enabled)
        if tiny is not None else None
    )
    root.train()
    body.train()
    tracked = root.mamba_layers[0].in_proj.weight
    tracked_before = tracked.detach().clone()
    initial_train_loss = None
    final_train_loss = None
    all_gradients_finite = True
    log_path = output / "train.jsonl"
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        if tiny is not None:
            call = tiny[(step - 1) % len(tiny)]
        else:
            sample = train.samples[random.randrange(len(train))]
            call = sample["calls"][random.randrange(len(sample["calls"]))]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            total, root_loss, body_loss = _loss(root, body, call, device)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite Mamba loss at step {step}")
        total.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        finite = bool(gradients) and all(torch.isfinite(g).all() for g in gradients)
        all_gradients_finite = all_gradients_finite and finite
        if not finite:
            raise RuntimeError(f"missing or non-finite Mamba gradients at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        final_train_loss = float(total)
        if initial_train_loss is None:
            initial_train_loss = final_train_loss
        if step % args.log_every == 0 or step == 1:
            record = {
                "step": step,
                "loss": final_train_loss,
                "root_loss": float(root_loss),
                "body_loss": float(body_loss),
                "grad_norm": float(grad_norm),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            dev_metrics = _evaluate(root, body, dev, device, autocast_enabled)
            print(json.dumps({"step": step, "development": dev_metrics}), flush=True)
            root.train()
            body.train()
        if args.save_every and step % args.save_every == 0:
            torch.save(
                _checkpoint(root, body, args, {"step": step}),
                output / f"checkpoint_step_{step:06d}.pt",
            )

    final_dev = _evaluate(root, body, dev, device, autocast_enabled)
    final_tiny = (
        _evaluate_calls(root, body, tiny, device, autocast_enabled)
        if tiny is not None else None
    )
    state = _state_gate(root, train.samples[0]["calls"][0], device)
    parameter_change = float((tracked.detach() - tracked_before).float().norm())
    metrics = {
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_tiny_mean_loss": initial_tiny,
        "final_tiny_mean_loss": final_tiny,
        "initial_development": initial_dev,
        "final_development": final_dev,
        "all_gradients_finite": all_gradients_finite,
        "tracked_parameter_change_norm": parameter_change,
        "state_gate": state,
        "steps": args.max_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
    }
    checkpoint_path = output / (
        "sequence_mamba_tiny.pt" if args.phase == "tiny"
        else "sequence_mamba_c1_base.pt"
    )
    torch.save(_checkpoint(root, body, args, metrics), checkpoint_path)
    reload_loss = _reload_gate(raw, checkpoint_path, args, train.samples[0]["calls"][0], device)
    metrics["reload_loss"] = reload_loss
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    if args.phase == "tiny":
        if final_tiny >= 0.5 * initial_tiny:
            raise RuntimeError("tiny-overfit loss did not decrease by at least 50%")
        if state["ssm_state_variance"] <= 0 or state["carry_mean_absolute_effect"] <= 0:
            raise RuntimeError("Mamba recurrent state tiny gate failed")
        if parameter_change <= 0:
            raise RuntimeError("Mamba parameters did not change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
