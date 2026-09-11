"""Train Minimal from a real DiT checkpoint and sequential teacher cache."""

from __future__ import annotations

import argparse
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
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--dit-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-train-trajectories", type=int, default=None)
    parser.add_argument("--max-eval-trajectories", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--window-size-chunks", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29)
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


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


class TrajectoryDataset:
    def __init__(self, manifest_path: str, limit: int | None) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        records = self.manifest["records"]
        if limit is not None:
            records = records[:limit]
        self.samples = [
            torch.load(
                self.manifest_path.parent / record["path"],
                map_location="cpu",
                weights_only=False,
            )
            for record in records
        ]
        if not self.samples:
            raise ValueError(f"empty trajectory cache: {manifest_path}")
        for sample in self.samples:
            expected = sample["num_chunks"] * self.manifest["diffusion_steps"]
            if len(sample["calls"]) != expected:
                raise ValueError("trajectory has an incomplete diffusion call sequence")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def _branch_zero(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    batch = inputs[0].shape[0]
    if batch not in (1, 2, 3):
        raise ValueError(f"unsupported cached CFG batch {batch}")
    return tuple(
        value[:1] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch else value
        for value in inputs
    )


def _call(sample: dict[str, Any], chunk: int, diffusion_index: int) -> dict[str, Any]:
    steps = len(sample["calls"]) // sample["num_chunks"]
    return sample["calls"][chunk * steps + diffusion_index]


def _commit_history(raw, sample: dict[str, Any], current_chunk: int) -> None:
    if current_chunk < 1:
        return
    clean_call = _call(sample, current_chunk, len(sample["calls"]) // sample["num_chunks"] - 1)
    timestep = clean_call["diffusion_timestep"]
    if int(timestep.max()) != 0:
        raise RuntimeError("teacher-forced history commit must come from timestep zero")
    root_inputs = _move(_branch_zero(clean_call["root"]["inputs"]), raw.root_model._stack_parameter().device)
    body_inputs = _move(_branch_zero(clean_call["body"]["inputs"]), raw.body_model._stack_parameter().device)
    history_valid = clean_call["outer_inputs"]["history_token_mask"][:1].bool()
    tokens_per_chunk = sample["chunk_frames"] // raw.num_frames_per_token
    first_history = max(0, current_chunk - raw.root_model.sequence_stack.layers[0].mixer.config.attention.window_size_chunks)
    for history_chunk in range(first_history, current_chunk):
        start = history_chunk * tokens_per_chunk
        stop = start + tokens_per_chunk
        selection = torch.zeros_like(root_inputs[1], dtype=torch.bool)
        selection[:, start:stop] = True
        if bool((selection.cpu() & ~history_valid).any()):
            raise RuntimeError("history commit selected a non-history token")
        with torch.no_grad():
            raw.root_model.prepare_accepted_commit(selection, chunk_id=history_chunk)
            raw.root_model(*root_inputs)
            raw.body_model.prepare_accepted_commit(selection, chunk_id=history_chunk)
            raw.body_model(*body_inputs)


def _current_loss(raw, sample: dict[str, Any], chunk: int, diffusion_index: int, device: torch.device):
    call = _call(sample, chunk, diffusion_index)
    root_inputs = _move(call["root"]["inputs"], device)
    body_inputs = _move(call["body"]["inputs"], device)
    root_target = call["root"]["teacher_output"].to(device)
    body_target = call["body"]["teacher_output"].to(device)
    root_prediction = raw.root_model(*root_inputs)
    body_prediction = raw.body_model(*body_inputs)
    root_loss = F.mse_loss(root_prediction.float(), root_target.float())
    body_loss = F.mse_loss(body_prediction.float(), body_target.float())
    return root_loss + body_loss, root_loss, body_loss, root_prediction, body_prediction


def _reset(raw) -> None:
    raw.root_model.reset_memory()
    raw.body_model.reset_memory()


def _cache_summary(block) -> dict[str, Any]:
    if block._sequence_state is None:
        return {"num_chunks": 0, "chunk_ids": [], "oldest": None, "newest": None}
    cache = block._sequence_state.layers[0].attention_cache
    return {
        "num_chunks": cache.num_chunks,
        "chunk_ids": list(cache.chunk_ids),
        "oldest": cache.oldest_chunk_id,
        "newest": cache.newest_chunk_id,
        "sink_count": cache.num_sink_tokens,
    }


def _evaluate(raw, dataset: TrajectoryDataset, device: torch.device, autocast_enabled: bool) -> dict[str, float]:
    was_training = raw.training
    raw.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    count = 0
    steps = dataset.manifest["diffusion_steps"]
    chosen_diffusion = sorted({0, steps // 2, steps - 1})
    with torch.inference_mode():
        for sample in dataset.samples:
            for chunk in range(1, sample["num_chunks"]):
                for diffusion_index in chosen_diffusion:
                    _reset(raw)
                    _commit_history(raw, sample, chunk)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=autocast_enabled,
                    ):
                        total, root_loss, body_loss, _, _ = _current_loss(
                            raw, sample, chunk, diffusion_index, device
                        )
                    if not torch.isfinite(total):
                        raise RuntimeError("non-finite Minimal eval loss")
                    totals += torch.tensor(
                        [float(total), float(root_loss), float(body_loss)], dtype=torch.float64
                    )
                    count += 1
    _reset(raw)
    raw.train(was_training)
    return {
        "loss": float(totals[0] / count),
        "root_loss": float(totals[1] / count),
        "body_loss": float(totals[2] / count),
        "cases": count,
    }


def _history_effect(raw, first: dict[str, Any], second: dict[str, Any], device: torch.device) -> dict[str, float]:
    chunk = 1
    diffusion_index = 5
    with torch.inference_mode():
        _reset(raw)
        _, _, _, root_none, body_none = _current_loss(raw, first, chunk, diffusion_index, device)
        _reset(raw)
        _commit_history(raw, first, chunk)
        _, _, _, root_same, body_same = _current_loss(raw, first, chunk, diffusion_index, device)
        same_summary = _cache_summary(raw.root_model)
        _reset(raw)
        _commit_history(raw, second, chunk)
        _, _, _, root_other, body_other = _current_loss(raw, first, chunk, diffusion_index, device)
    _reset(raw)
    same_effect = torch.cat(
        [(root_same - root_none).abs().flatten(), (body_same - body_none).abs().flatten()]
    ).mean()
    different_effect = torch.cat(
        [(root_other - root_same).abs().flatten(), (body_other - body_same).abs().flatten()]
    ).mean()
    return {
        "history_vs_none_mean_abs": float(same_effect),
        "different_history_mean_abs": float(different_effect),
        "cached_chunks": same_summary["num_chunks"],
    }


def _fifo_gate(raw, sample: dict[str, Any]) -> dict[str, Any]:
    _reset(raw)
    clean_call = _call(sample, 1, len(sample["calls"]) // sample["num_chunks"] - 1)
    root_inputs = _move(_branch_zero(clean_call["root"]["inputs"]), raw.root_model._stack_parameter().device)
    body_inputs = _move(_branch_zero(clean_call["body"]["inputs"]), raw.body_model._stack_parameter().device)
    tokens = sample["chunk_frames"] // raw.num_frames_per_token
    selection = torch.zeros_like(root_inputs[1], dtype=torch.bool)
    selection[:, :tokens] = True
    window = raw.root_model.sequence_stack.layers[0].mixer.config.attention.window_size_chunks
    with torch.no_grad():
        for chunk_id in range(window + 1):
            raw.root_model.prepare_accepted_commit(selection, chunk_id=chunk_id)
            raw.root_model(*root_inputs)
            raw.body_model.prepare_accepted_commit(selection, chunk_id=chunk_id)
            raw.body_model(*body_inputs)
    root_summary = _cache_summary(raw.root_model)
    body_summary = _cache_summary(raw.body_model)
    expected = list(range(1, window + 1))
    if root_summary["chunk_ids"] != expected or body_summary["chunk_ids"] != expected:
        raise RuntimeError("Minimal FIFO eviction gate failed")
    _reset(raw)
    return {"root": root_summary, "body": body_summary}


def _load_dit_weights(raw, path: str, device: torch.device) -> dict[str, list[str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("mode") != "dit":
        raise ValueError("--dit-checkpoint must contain mode='dit'")
    result = {}
    for name, block in (("root", raw.root_model), ("body", raw.body_model)):
        incompatible = block.load_state_dict(checkpoint[f"{name}_model"], strict=False)
        if incompatible.unexpected_keys or any(
            not key.endswith("sink_tokens") for key in incompatible.missing_keys
        ):
            raise RuntimeError(f"incompatible DiT-to-Minimal initialization: {incompatible}")
        result[name] = list(incompatible.missing_keys)
    return result


def _weights_checkpoint(raw, args, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 2,
        "mode": "minimal",
        "root_model": raw.root_model.state_dict(),
        "body_model": raw.body_model.state_dict(),
        "training": {
            "phase": "teacher_forced_sequential_kv",
            "dit_initialization": str(Path(args.dit_checkpoint).resolve()),
            "cache_manifest": str(Path(args.train_manifest).resolve()),
            "window_size_chunks": args.window_size_chunks,
            "sink_tokens": args.sink_tokens,
            "seed": args.seed,
            "git_commit": _git_commit(),
            "metrics": metrics,
        },
    }


def _save(output_dir: Path, raw, optimizer, args, step: int, metrics: dict[str, Any]) -> None:
    weights = _weights_checkpoint(raw, args, metrics)
    torch.save(weights, output_dir / "sequence_minimal.pt")
    resume = dict(weights)
    resume["optimizer"] = optimizer.state_dict()
    resume["global_step"] = step
    resume["rng_state"] = torch.get_rng_state()
    resume["python_rng_state"] = random.getstate()
    torch.save(resume, output_dir / "checkpoint_last.pt")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["git_commit"] = _git_commit()
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

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
            mode="minimal",
            attention_backend="sdpa",
            window_size_chunks=args.window_size_chunks,
            num_sink_tokens=args.sink_tokens,
        ),
    )
    raw = model.denoiser.model
    initialization = _load_dit_weights(raw, args.dit_checkpoint, device)
    if any("mamba" in type(module).__name__.lower() for module in raw.modules()):
        raise RuntimeError("Minimal instantiated a Mamba module")
    if "mamba_ssm" in sys.modules:
        raise RuntimeError("Minimal imported mamba_ssm")

    for parameter in raw.parameters():
        parameter.requires_grad_(False)
    trainable = list(raw.root_model.parameters()) + list(raw.body_model.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("mode") != "minimal":
            raise ValueError("resume checkpoint is not Minimal")
        raw.root_model.load_state_dict(resume["root_model"], strict=True)
        raw.body_model.load_state_dict(resume["body_model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["global_step"])
        torch.set_rng_state(resume["rng_state"].cpu())
        random.setstate(resume["python_rng_state"])

    train_data = TrajectoryDataset(args.train_manifest, args.max_train_trajectories)
    eval_data = TrajectoryDataset(args.eval_manifest, args.max_eval_trajectories)
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    fifo = _fifo_gate(raw, train_data[0])
    initial_effect = _history_effect(raw, train_data[0], train_data[1], device)
    if initial_effect["history_vs_none_mean_abs"] <= 0 or initial_effect["different_history_mean_abs"] <= 0:
        raise RuntimeError("history K/V does not measurably change Minimal output")
    initial_eval = _evaluate(raw, eval_data, device, autocast_enabled)
    sink_parameters = [
        parameter
        for name, parameter in list(raw.root_model.named_parameters()) + list(raw.body_model.named_parameters())
        if name.endswith("sink_tokens")
    ]
    if not sink_parameters:
        raise RuntimeError("Minimal has no learned attention sinks")
    tracked = sink_parameters[0]
    tracked_before = tracked.detach().clone()
    max_cached_chunks = 0
    saw_sink_gradient = False
    log_path = output_dir / "train.jsonl"
    started = time.perf_counter()
    last_train: dict[str, float] = {}
    steps_per_trajectory = train_data.manifest["diffusion_steps"]

    raw.train()
    for step in range(start_step + 1, args.max_steps + 1):
        sample = train_data[random.randrange(len(train_data))]
        chunk = random.randrange(1, sample["num_chunks"])
        diffusion_index = random.randrange(steps_per_trajectory)
        _reset(raw)
        root_commits = raw.root_model.accepted_commits
        body_commits = raw.body_model.accepted_commits
        _commit_history(raw, sample, chunk)
        summary = _cache_summary(raw.root_model)
        max_cached_chunks = max(max_cached_chunks, summary["num_chunks"])
        if raw.root_model.accepted_commits - root_commits != summary["num_chunks"]:
            raise RuntimeError("root history commit count mismatch")
        if raw.body_model.accepted_commits - body_commits != summary["num_chunks"]:
            raise RuntimeError("body history commit count mismatch")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            total, root_loss, body_loss, _, _ = _current_loss(
                raw, sample, chunk, diffusion_index, device
            )
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite Minimal loss at step {step}")
        total.backward()
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"missing or non-finite Minimal gradients at step {step}")
        sink_grad_norm = sum(
            float(parameter.grad.float().norm())
            for parameter in sink_parameters
            if parameter.grad is not None
        )
        saw_sink_gradient = saw_sink_gradient or sink_grad_norm > 0
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        # Projected K/V is now stale by definition. Discard it immediately.
        _reset(raw)
        last_train = {
            "loss": float(total.detach()),
            "root_loss": float(root_loss.detach()),
            "body_loss": float(body_loss.detach()),
            "grad_norm": float(grad_norm),
            "sink_grad_norm": sink_grad_norm,
            "chunk": chunk,
            "diffusion_index": diffusion_index,
        }
        record: dict[str, Any] | None = None
        if step == 1 or step % args.log_every == 0:
            record = {"step": step, "train": last_train}
        if step % args.eval_every == 0 or step == args.max_steps:
            evaluation = _evaluate(raw, eval_data, device, autocast_enabled)
            record = {"step": step, "train": last_train, "eval": evaluation}
        if record is not None:
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0:
            _save(output_dir, raw, optimizer, args, step, record or last_train)

    final_eval = _evaluate(raw, eval_data, device, autocast_enabled)
    final_effect = _history_effect(raw, train_data[0], train_data[1], device)
    parameter_changed = not torch.equal(tracked_before, tracked.detach())
    if not parameter_changed or not saw_sink_gradient:
        raise RuntimeError("attention sinks did not train")
    metrics = {
        "initialization_missing_sink_keys": initialization,
        "fifo_gate": fifo,
        "initial_history_effect": initial_effect,
        "final_history_effect": final_effect,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "relative_eval_reduction": (
            initial_eval["loss"] - final_eval["loss"]
        ) / max(abs(initial_eval["loss"]), 1e-12),
        "last_train": last_train,
        "sink_parameter_changed": parameter_changed,
        "saw_sink_gradient": saw_sink_gradient,
        "max_cached_chunks": max_cached_chunks,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
    }
    _save(output_dir, raw, optimizer, args, args.max_steps, metrics)
    saved = torch.load(output_dir / "sequence_minimal.pt", map_location=device, weights_only=False)
    raw.root_model.load_state_dict(saved["root_model"], strict=True)
    raw.body_model.load_state_dict(saved["body_model"], strict=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
