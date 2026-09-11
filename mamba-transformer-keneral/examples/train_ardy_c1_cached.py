"""Train bounded MotionMamba C1 pilots from the frozen Stage-05 Minimal model."""

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


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))

from train_ardy_minimal_cached import (  # noqa: E402
    TrajectoryDataset,
    _branch_zero,
    _call,
    _current_loss,
    _evaluate,
    _move,
    _reset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--minimal-checkpoint", required=True)
    parser.add_argument("--mamba-checkpoint", default=None)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--interaction",
        choices=[
            "naive_gated_add", "state_query", "state_kv_modulation",
            "h0_parallel", "h1_innovation", "h2_delta_rule",
        ],
        required=True,
    )
    parser.add_argument(
        "--use-state-delta", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--state-summary-dim", type=int, default=128)
    parser.add_argument("--state-scale-init", type=float, default=0.0)
    parser.add_argument("--naive-gate-logit-init", type=float, default=-8.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-train-trajectories", type=int, default=16)
    parser.add_argument("--max-eval-trajectories", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--new-parameter-lr-multiplier", type=float, default=10.0)
    parser.add_argument(
        "--train-new-only",
        action="store_true",
        help="Freeze the validated Minimal parent during the tiny C1 gate.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--window-size-chunks", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--mamba-headdim", type=int, default=64)
    parser.add_argument("--mamba-d-state", type=int, default=64)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-chunk-size", type=int, default=16)
    parser.add_argument("--fixed-case", action="store_true")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--log-every", type=int, default=5)
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


def _commit_history_differentiable(raw, sample: dict[str, Any], current_chunk: int) -> None:
    """Commit clean teacher history without detaching its recurrent graph."""
    if current_chunk < 1:
        return
    diffusion_steps = len(sample["calls"]) // sample["num_chunks"]
    clean_call = _call(sample, current_chunk, diffusion_steps - 1)
    if int(clean_call["diffusion_timestep"].max()) != 0:
        raise RuntimeError("C1 history must use accepted timestep-zero chunks")
    root_device = raw.root_model._stack_parameter().device
    body_device = raw.body_model._stack_parameter().device
    root_inputs = _move(_branch_zero(clean_call["root"]["inputs"]), root_device)
    body_inputs = _move(_branch_zero(clean_call["body"]["inputs"]), body_device)
    history_valid = clean_call["outer_inputs"]["history_token_mask"][:1].bool()
    tokens_per_chunk = sample["chunk_frames"] // raw.num_frames_per_token
    window = raw.root_model.sequence_stack.layers[0].mixer.config.attention.window_size_chunks
    first_history = max(0, current_chunk - window)
    for history_chunk in range(first_history, current_chunk):
        selection = torch.zeros_like(root_inputs[1], dtype=torch.bool)
        start = history_chunk * tokens_per_chunk
        selection[:, start : start + tokens_per_chunk] = True
        if bool((selection.cpu() & ~history_valid).any()):
            raise RuntimeError("C1 history selected invalid/non-history tokens")
        raw.root_model.prepare_accepted_commit(
            selection, chunk_id=history_chunk, detach=False
        )
        raw.root_model(*root_inputs)
        raw.body_model.prepare_accepted_commit(
            selection, chunk_id=history_chunk, detach=False
        )
        raw.body_model(*body_inputs)


def _load_minimal(raw, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("mode") != "minimal":
        raise ValueError("--minimal-checkpoint must contain mode='minimal'")
    missing: dict[str, list[str]] = {}
    for name, block in (("root", raw.root_model), ("body", raw.body_model)):
        incompatible = block.load_state_dict(checkpoint[f"{name}_model"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected Minimal keys for {name}: {incompatible}")
        allowed = (
            ".mamba_path.",
            ".fusion_gate_logit",
            ".state_projector.",
            ".token_state_interaction.",
            ".output_interaction.",
            ".delta_rule.",
        )
        rejected = [key for key in incompatible.missing_keys if not any(x in key for x in allowed)]
        if rejected:
            raise RuntimeError(f"non-C1 keys missing from Minimal initialization: {rejected}")
        missing[name] = list(incompatible.missing_keys)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "sha256": _sha256(checkpoint_path),
        "new_keys": missing,
    }


def _parameter_groups(raw) -> dict[str, list[torch.nn.Parameter]]:
    named = list(raw.root_model.named_parameters()) + list(raw.body_model.named_parameters())
    return {
        "mamba": [p for n, p in named if ".mamba_path." in n],
        "projector": [p for n, p in named if ".state_projector." in n],
        "interaction": [
            p
            for n, p in named
            if ".token_state_interaction." in n
            or ".output_interaction." in n
            or ".delta_rule." in n
            or n.endswith("fusion_gate_logit")
        ],
        "attention": [p for n, p in named if ".attention." in n],
    }


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    return sum(
        float(parameter.grad.detach().float().norm())
        for parameter in parameters
        if parameter.grad is not None
    )


def _interaction_snapshot(raw) -> dict[str, Any]:
    values = []
    for block_name, block in (("root", raw.root_model), ("body", raw.body_model)):
        for layer_index, layer in enumerate(block.sequence_stack.layers):
            mixer = layer.mixer
            if mixer.uses_token_state_interaction:
                values.append(
                    {
                        "block": block_name,
                        "layer": layer_index,
                        "scale": float(
                            mixer.token_state_interaction.effective_scale.detach().float().cpu()
                        ),
                    }
                )
            elif mixer.uses_output_interaction:
                values.append(
                    {
                        "block": block_name,
                        "layer": layer_index,
                        "scale": float(mixer.output_interaction.scale.detach().cpu()),
                    }
                )
            elif mixer.uses_delta_rule:
                values.append(
                    {
                        "block": block_name,
                        "layer": layer_index,
                        "scale": float(mixer.delta_rule.beta.detach().cpu()),
                        "output_scale": float(
                            mixer.delta_rule.output_scale.detach().cpu()
                        ),
                    }
                )
            else:
                values.append(
                    {
                        "block": block_name,
                        "layer": layer_index,
                        "scale": float(torch.sigmoid(mixer.fusion_gate_logit.detach()).cpu()),
                    }
                )
    return {"layers": values}


def _load_mamba_initialization(
    raw, checkpoint_path: str, device: torch.device
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("mode") != "mamba":
        raise ValueError("--mamba-checkpoint must contain mode='mamba'")
    loaded = []
    for block_name, block, source_name in (
        ("root", raw.root_model, "root_model"),
        ("body", raw.body_model, "body_model"),
    ):
        source = checkpoint[source_name]
        for layer_index, layer in enumerate(block.sequence_stack.layers):
            prefix = f"mamba_layers.{layer_index}."
            state = {
                key[len(prefix) :]: value
                for key, value in source.items()
                if key.startswith(prefix)
            }
            if not state:
                raise RuntimeError(f"missing {block_name} Mamba layer {layer_index}")
            layer.mixer.mamba_path.load_state_dict(state, strict=True)
            loaded.append(f"{block_name}:{layer_index}")
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "sha256": _sha256(checkpoint_path),
        "layers": loaded,
    }


def _save(raw, optimizer, args, output_dir: Path, step: int, metrics: dict[str, Any]) -> None:
    checkpoint = {
        "format_version": 1,
        "mode": "motionmamba_c1_pilot",
        "interaction": args.interaction,
        "use_state_delta": args.use_state_delta,
        "root_model": raw.root_model.state_dict(),
        "body_model": raw.body_model.state_dict(),
        "training": metrics,
    }
    torch.save(checkpoint, output_dir / "sequence_c1_pilot.pt")
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["global_step"] = step
    torch.save(checkpoint, output_dir / "checkpoint_last.pt")


def main() -> int:
    args = parse_args()
    if args.new_parameter_lr_multiplier <= 0:
        raise ValueError("--new-parameter-lr-multiplier must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    sys.path.insert(0, str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )
    install_ardy_sequence_kernel(
        model,
        ArdySequenceKernelConfig(
            mode="hybrid",
            attention_backend="sdpa",
            window_size_chunks=args.window_size_chunks,
            num_sink_tokens=args.sink_tokens,
            mamba_headdim=args.mamba_headdim,
            mamba_d_state=args.mamba_d_state,
            mamba_expand=args.mamba_expand,
            mamba_chunk_size=args.mamba_chunk_size,
            mamba_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32,
            token_state_interaction=args.interaction,
            state_summary_dim=args.state_summary_dim,
            use_state_delta=args.use_state_delta,
            state_conditioning_scale_init=args.state_scale_init,
            fusion_gate_logit_init=args.naive_gate_logit_init,
        ),
    )
    raw = model.denoiser.model
    initialization = _load_minimal(raw, args.minimal_checkpoint, device)
    mamba_initialization = (
        _load_mamba_initialization(raw, args.mamba_checkpoint, device)
        if args.mamba_checkpoint else None
    )
    for parameter in raw.parameters():
        parameter.requires_grad_(False)
    groups = _parameter_groups(raw)
    if not groups["mamba"] or not groups["interaction"] or not groups["attention"]:
        raise RuntimeError("C1 pilot is missing a required parameter group")
    if args.interaction in {"state_query", "state_kv_modulation"} and not groups["projector"]:
        raise RuntimeError("state-conditioned C1 pilot has no StateProjector")
    new_parameters = groups["mamba"] + groups["projector"] + groups["interaction"]
    new_ids = {id(parameter) for parameter in new_parameters}
    all_model_parameters = list(raw.root_model.parameters()) + list(raw.body_model.parameters())
    base_parameters = [parameter for parameter in all_model_parameters if id(parameter) not in new_ids]
    for parameter in new_parameters:
        parameter.requires_grad_(True)
    if not args.train_new_only:
        for parameter in base_parameters:
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in all_model_parameters if parameter.requires_grad]
    optimizer_groups = []
    if not args.train_new_only:
        optimizer_groups.append({"params": base_parameters, "lr": args.learning_rate})
    optimizer_groups.append(
        {
            "params": new_parameters,
            "lr": args.learning_rate * args.new_parameter_lr_multiplier,
        }
    )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)

    train_data = TrajectoryDataset(args.train_manifest, args.max_train_trajectories)
    eval_data = TrajectoryDataset(args.eval_manifest, args.max_eval_trajectories)
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    config = vars(args).copy()
    config.update(
        {
            "git_commit": _git_commit(),
            "minimal_initialization": initialization,
            "mamba_initialization": mamba_initialization,
            "train_manifest": str(Path(args.train_manifest).resolve()),
            "eval_manifest": str(Path(args.eval_manifest).resolve()),
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    initial_eval = _evaluate(raw, eval_data, device, autocast_enabled)
    observed_gradients = {name: False for name in groups}
    started = time.perf_counter()
    first_loss = None
    final_loss = None
    log_path = output_dir / "train.jsonl"
    raw.train()
    fixed_sample = train_data[0]
    fixed_chunk = min(1, fixed_sample["num_chunks"] - 1)
    fixed_diffusion = len(fixed_sample["calls"]) // fixed_sample["num_chunks"] // 2
    for step in range(1, args.max_steps + 1):
        if args.fixed_case:
            sample, chunk, diffusion_index = fixed_sample, fixed_chunk, fixed_diffusion
        else:
            sample = train_data[random.randrange(len(train_data))]
            chunk = random.randrange(1, sample["num_chunks"])
            diffusion_index = random.randrange(
                len(sample["calls"]) // sample["num_chunks"]
            )
        _reset(raw)
        optimizer.zero_grad(set_to_none=True)
        # Persistent attention K/V use the fp32 Minimal projection dtype.
        # Keep differentiable clean-history construction outside autocast;
        # SequenceKernel still casts only the Mamba branch to bf16.
        _commit_history_differentiable(raw, sample, chunk)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            total, root_loss, body_loss, _, _ = _current_loss(
                raw, sample, chunk, diffusion_index, device
            )
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite C1 loss at step {step}")
        total.backward()
        norms = {name: _grad_norm(parameters) for name, parameters in groups.items()}
        for name, norm in norms.items():
            observed_gradients[name] = observed_gradients[name] or norm > 0
        all_gradients = [p.grad for p in trainable if p.grad is not None]
        if not all_gradients or not all(torch.isfinite(g).all() for g in all_gradients):
            raise RuntimeError(f"missing/non-finite C1 gradients at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        _reset(raw)
        optimizer.step()
        value = float(total.detach())
        first_loss = value if first_loss is None else first_loss
        final_loss = value
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            record = {
                "step": step,
                "loss": value,
                "root_loss": float(root_loss.detach()),
                "body_loss": float(body_loss.detach()),
                "grad_norm": float(grad_norm),
                "group_grad_norms": norms,
                "interaction": _interaction_snapshot(raw),
                "elapsed_seconds": time.perf_counter() - started,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

    final_eval = _evaluate(raw, eval_data, device, autocast_enabled)
    required_gradient_groups = (
        ("mamba", "interaction")
        if args.train_new_only
        else ("mamba", "interaction", "attention")
    )
    if not all(observed_gradients[name] for name in required_gradient_groups):
        raise RuntimeError(f"required C1 gradients were never observed: {observed_gradients}")
    metrics = {
        "phase": "stage07_tiny_overfit" if args.fixed_case else "stage07_equal_budget_pilot",
        "first_train_loss": first_loss,
        "final_train_loss": final_loss,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "observed_gradients": observed_gradients,
        "interaction": _interaction_snapshot(raw),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
        "seed": args.seed,
        "git_commit": _git_commit(),
        "minimal_initialization": initialization,
        "mamba_initialization": mamba_initialization,
    }
    _save(raw, optimizer, args, output_dir, args.max_steps, metrics)
    (output_dir / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
