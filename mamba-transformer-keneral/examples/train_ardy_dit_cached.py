"""Train the real ARDY root/body DiT student from a captured teacher cache."""

from __future__ import annotations

import argparse
import functools
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-eval-batches", type=int, default=64)
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


@functools.lru_cache(maxsize=16)
def _load_sample(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


class TeacherCallDataset(Dataset):
    def __init__(self, manifest_path: str, *, preload: bool = True) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text())
        self.index: list[tuple[str, int]] = []
        for record in self.manifest["records"]:
            path = str((self.root / record["path"]).resolve())
            for call_index in range(int(record["num_calls"])):
                self.index.append((path, call_index))
        if not self.index:
            raise ValueError(f"empty teacher cache: {manifest_path}")
        self.preloaded = (
            [self._read_item(index) for index in range(len(self.index))]
            if preload
            else None
        )

    def __len__(self) -> int:
        return len(self.index)

    def _read_item(self, index: int) -> dict[str, Any]:
        path, call_index = self.index[index]
        call = _load_sample(path)["calls"][call_index]
        return {
            "root_inputs": call["root"]["inputs"],
            "root_target": call["root"]["teacher_output"],
            "body_inputs": call["body"]["inputs"],
            "body_target": call["body"]["teacher_output"],
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.preloaded is not None:
            return self.preloaded[index]
        return self._read_item(index)


def _merge_inputs(rows: list[dict[str, Any]], key: str) -> tuple[Any, ...]:
    merged: list[Any] = []
    for values in zip(*(row[key] for row in rows)):
        first = values[0]
        if isinstance(first, torch.Tensor):
            if not all(value.shape[1:] == first.shape[1:] for value in values):
                raise ValueError("cached calls in one batch have incompatible shapes")
            merged.append(torch.cat(values, dim=0))
        elif first is None and all(value is None for value in values):
            merged.append(None)
        elif all(value == first for value in values):
            merged.append(first)
        else:
            raise ValueError("unsupported non-tensor cached input")
    return tuple(merged)


def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "root_inputs": _merge_inputs(rows, "root_inputs"),
        "root_target": torch.cat([row["root_target"] for row in rows], dim=0),
        "body_inputs": _merge_inputs(rows, "body_inputs"),
        "body_target": torch.cat([row["body_target"] for row in rows], dim=0),
    }


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


def _loss(raw, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root_prediction = raw.root_model(*batch["root_inputs"])
    body_prediction = raw.body_model(*batch["body_inputs"])
    root_loss = F.mse_loss(root_prediction.float(), batch["root_target"].float())
    body_loss = F.mse_loss(body_prediction.float(), batch["body_target"].float())
    return root_loss + body_loss, root_loss, body_loss


def _evaluate(
    raw,
    loader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
    max_batches: int,
) -> dict[str, float]:
    was_training = raw.training
    raw.eval()
    sums = torch.zeros(3, dtype=torch.float64)
    count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = _move(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                total, root_loss, body_loss = _loss(raw, batch)
            if not torch.isfinite(total):
                raise RuntimeError("non-finite eval loss")
            sums += torch.tensor(
                [float(total), float(root_loss), float(body_loss)], dtype=torch.float64
            )
            count += 1
    raw.train(was_training)
    return {
        "loss": float(sums[0] / count),
        "root_loss": float(sums[1] / count),
        "body_loss": float(sums[2] / count),
        "batches": count,
    }


def _weights_checkpoint(raw, args: argparse.Namespace, metrics: dict[str, Any]) -> dict:
    train_manifest = json.loads(Path(args.train_manifest).read_text())
    return {
        "format_version": 2,
        "mode": "dit",
        "root_model": raw.root_model.state_dict(),
        "body_model": raw.body_model.state_dict(),
        "training": {
            "phase": "real_cached_teacher_distillation",
            "teacher_model": args.teacher_model,
            "teacher_revision": train_manifest["teacher_revision"],
            "cache_manifest": str(Path(args.train_manifest).resolve()),
            "seed": args.seed,
            "git_commit": _git_commit(),
            "metrics": metrics,
        },
    }


def _save(
    output_dir: Path,
    raw,
    optimizer,
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, Any],
) -> None:
    weights = _weights_checkpoint(raw, args, metrics)
    torch.save(weights, output_dir / "sequence_dit.pt")
    resume = dict(weights)
    resume["optimizer"] = optimizer.state_dict()
    resume["global_step"] = step
    resume["rng_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        resume["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(resume, output_dir / "checkpoint_last.pt")


def main() -> int:
    args = parse_args()
    if args.max_steps < 1 or args.batch_size < 1:
        raise ValueError("max steps and batch size must be positive")
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

    ardy_root = Path(args.ardy_root).resolve()
    sys.path.insert(0, str(ardy_root))
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
            mode="dit",
            attention_backend="sdpa",
            window_size_chunks=0,
            num_sink_tokens=2,
        ),
    )
    raw = model.denoiser.model
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
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("mode") != "dit":
            raise ValueError("resume checkpoint is not DiT")
        raw.root_model.load_state_dict(checkpoint["root_model"], strict=True)
        raw.body_model.load_state_dict(checkpoint["body_model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    train_dataset = TeacherCallDataset(args.train_manifest)
    eval_dataset = TeacherCallDataset(args.eval_manifest)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    initial_eval = _evaluate(
        raw, eval_loader, device, autocast_enabled, args.max_eval_batches
    )
    tracked = next(raw.root_model.sequence_stack.parameters())
    tracked_before = tracked.detach().clone()
    log_path = output_dir / "train.jsonl"
    started = time.perf_counter()
    train_iterator = iter(train_loader)
    last_train: dict[str, float] = {}

    raw.train()
    for step in range(start_step + 1, args.max_steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            total, root_loss, body_loss = _loss(raw, batch)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite training loss at step {step}")
        total.backward()
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"missing or non-finite gradients at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        # No projected K/V may cross an optimizer update.  DiT has no persistent
        # cache, but this invariant is kept explicit for checkpoint portability.
        raw.root_model.reset_memory()
        raw.body_model.reset_memory()
        last_train = {
            "loss": float(total.detach()),
            "root_loss": float(root_loss.detach()),
            "body_loss": float(body_loss.detach()),
            "grad_norm": float(grad_norm),
        }
        record: dict[str, Any] | None = None
        if step == 1 or step % args.log_every == 0:
            record = {"step": step, "train": last_train}
        if step % args.eval_every == 0 or step == args.max_steps:
            evaluation = _evaluate(
                raw, eval_loader, device, autocast_enabled, args.max_eval_batches
            )
            record = {"step": step, "train": last_train, "eval": evaluation}
        if record is not None:
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0:
            _save(output_dir, raw, optimizer, args, step, record or last_train)

    final_eval = _evaluate(
        raw, eval_loader, device, autocast_enabled, args.max_eval_batches
    )
    parameter_changed = not torch.equal(tracked_before, tracked.detach())
    if not parameter_changed:
        raise RuntimeError("student parameters did not change")
    metrics = {
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "relative_eval_reduction": (
            (initial_eval["loss"] - final_eval["loss"])
            / max(abs(initial_eval["loss"]), 1e-12)
        ),
        "last_train": last_train,
        "parameter_changed": parameter_changed,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
        ),
    }
    _save(output_dir, raw, optimizer, args, args.max_steps, metrics)
    # Destructive in-memory save/load gate.
    saved = torch.load(output_dir / "sequence_dit.pt", map_location=device, weights_only=False)
    raw.root_model.load_state_dict(saved["root_model"], strict=True)
    raw.body_model.load_state_dict(saved["body_model"], strict=True)
    if "mamba_ssm" in sys.modules:
        raise RuntimeError("DiT training unexpectedly imported Mamba")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
