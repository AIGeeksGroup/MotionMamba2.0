#!/usr/bin/env python3
"""Learned delayed-cue gate using the production SequenceKernel and Mamba3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
for path in (
    os.environ.get("MAMBA_REPO", "/data3/zts/packages/state-spaces-mamba"),
    os.environ.get("MAMBA_TRITON_OVERLAY", "/data3/zts/packages/mamba_op_triton"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from minimal_attention_kernel import MinimalAttentionKernelConfig  # noqa: E402
from sequence_kernel import SequenceKernel, SequenceKernelConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--state-summary-dim", type=int, default=16)
    parser.add_argument("--memory-slots", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--contextual", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DelayedCueModel(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        state_summary_dim: int,
        memory_slots: int,
        device: torch.device,
        contextual: bool = False,
    ) -> None:
        super().__init__()
        from mamba_ssm.modules.mamba3 import Mamba3

        mamba = Mamba3(
            d_model=d_model,
            d_state=16,
            expand=2,
            headdim=16,
            ngroups=1,
            is_mimo=False,
            chunk_size=16,
            layer_idx=0,
            device=device,
            dtype=torch.bfloat16,
        )
        config = SequenceKernelConfig(
            mode="hybrid",
            attention=MinimalAttentionKernelConfig(
                d_model=d_model,
                num_heads=4,
                window_size_chunks=4,
                use_attention_sinks=True,
                num_sink_tokens=1,
                attention_backend="sdpa",
            ),
            interaction="state_query",
            state_summary_dim=state_summary_dim,
            use_state_delta=True,
            state_conditioning_scale_init=0.0,
            state_aware_output_mode="native_gated_add",
            fusion_gate_logit_init=0.0,
            content_memory_enabled=True,
            content_memory_slots=memory_slots,
            content_memory_read_enabled=True,
            content_memory_use_state_delta=True,
            content_memory_write_gate_context="previous_memory",
            content_memory_decoder_init_scale=0.02,
            content_memory_write_gate_init_scale=0.02,
            content_memory_write_gate_bias_init=0.0,
            content_memory_read_scale_init=0.0,
            content_memory_conditioning=("text" if contextual else "none"),
            content_memory_normalize_delta=contextual,
            content_memory_normalize_slots=contextual,
            content_memory_max_write_rate=(0.25 if contextual else 1.0),
        )
        self.kernel = SequenceKernel(config, mamba3=mamba)
        self.classifier = nn.Linear(d_model, 2)
        self.contextual = contextual
        self.to(device=device)
        # The mechanism benchmark isolates content read/write. Mamba remains
        # trainable as the state source, but neither its direct output nor the
        # C address branch can carry the cue to W8.
        self.kernel.runtime_mamba_output_scale = 0.0
        self.kernel.fusion_gate_logit.requires_grad_(False)
        for parameter in self.kernel.token_state_interaction.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        intervention: str = "correct",
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if tokens.ndim != 4 or tokens.shape[1] != 8:
            raise ValueError("delayed-cue task requires [B,W1-W8,T,D]")
        if intervention not in {
            "correct",
            "zero_delta_at_W1",
            "permute_delta_at_W1",
            "write_off_at_W1",
            "readout_off_after_W4",
            "permute_memory_after_W4",
            "direct_mamba_readout_off_after_W4",
            "state_address_off_after_W4",
            "permute_context_at_W1",
        }:
            raise ValueError(f"unknown intervention: {intervention}")
        state = self.kernel.init_state(batch_size=tokens.shape[0])
        if self.contextual and condition is None:
            raise ValueError("contextual delayed-cue task requires condition")
        attention_chunks = []
        memory_norms = []
        for window in range(7):
            delta_mode = "normal"
            if window == 0 and intervention == "zero_delta_at_W1":
                delta_mode = "zero"
            elif window == 0 and intervention == "permute_delta_at_W1":
                delta_mode = "permute"
            _, pending, _ = self.kernel.forward_chunk(
                tokens[:, window],
                state=state,
                content_memory_write_enabled=(
                    window == 0 and intervention != "write_off_at_W1"
                ),
                content_memory_delta_mode=delta_mode,
                content_memory_condition=(
                    torch.roll(condition, shifts=2, dims=0)
                    if window == 0 and intervention == "permute_context_at_W1"
                    else condition
                ),
            )
            self.kernel.commit(state, pending, chunk_id=window, detach=False)
            attention_chunks.append(state.attention_cache.num_chunks)
            memory_norms.append(
                float(state.persistent_memory.detach().float().norm(dim=-1).mean())
            )
            if intervention == "permute_memory_after_W4" and window == 3:
                permutation = torch.roll(
                    torch.arange(tokens.shape[0], device=tokens.device),
                    shifts=(2 if self.contextual else 1),
                )
                state.persistent_memory = self.kernel.content_memory.permute_rows(
                    state.persistent_memory, permutation
                )
        read_enabled = intervention != "readout_off_after_W4"
        output, _, aux = self.kernel.forward_chunk(
            tokens[:, 7],
            state=state,
            content_memory_read_enabled=read_enabled,
            content_memory_write_enabled=False,
            content_memory_condition=condition,
        )
        return self.classifier(output.float().mean(dim=1)), {
            "attention_cache_chunks": attention_chunks,
            "memory_slot_norms": memory_norms,
            "final": aux["persistent_content_memory"],
        }


class MinimalControl(nn.Module):
    def __init__(self, *, d_model: int, device: torch.device) -> None:
        super().__init__()
        self.kernel = SequenceKernel(
            SequenceKernelConfig(
                mode="minimal",
                attention=MinimalAttentionKernelConfig(
                    d_model=d_model,
                    num_heads=4,
                    window_size_chunks=4,
                    use_attention_sinks=True,
                    num_sink_tokens=1,
                    attention_backend="sdpa",
                ),
            )
        )
        self.classifier = nn.Linear(d_model, 2)
        self.to(device=device)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        state = self.kernel.init_state(batch_size=tokens.shape[0])
        chunks = []
        for window in range(7):
            _, pending, _ = self.kernel.forward_chunk(tokens[:, window], state=state)
            self.kernel.commit(state, pending, chunk_id=window, detach=False)
            chunks.append(state.attention_cache.num_chunks)
        output, _, _ = self.kernel.forward_chunk(tokens[:, 7], state=state)
        return self.classifier(output.mean(dim=1)), {"attention_cache_chunks": chunks}


def _make_batch(
    batch_size: int,
    d_model: int,
    chunk_tokens: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    contextual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    divisor = 4 if contextual else 2
    if batch_size % divisor:
        raise ValueError(f"batch size must be divisible by {divisor}")
    cue_labels = torch.arange(batch_size, device=device) % 2
    context_labels = (torch.arange(batch_size, device=device) // 2) % 2
    labels = cue_labels ^ context_labels if contextual else cue_labels
    tokens = torch.zeros(batch_size, 8, chunk_tokens, d_model, device=device)
    cue = cue_labels.float().mul(2.0).sub(1.0)
    tokens[:, 0, :, 0] = cue[:, None]
    tokens[:, 0, :, 1] = 1.0
    distractors = torch.randn(
        batch_size,
        6,
        chunk_tokens,
        d_model - 4,
        generator=generator,
        device=device,
    )
    tokens[:, 1:7, :, 2:-2] = 0.25 * distractors
    tokens[:, 1:7, :, -2] = torch.arange(
        1, 7, device=device
    )[None, :, None].float()
    tokens[:, 7, :, -1] = 1.0
    condition = None
    if contextual:
        condition = torch.zeros(batch_size, d_model, device=device)
        condition[:, 0] = context_labels.float().mul(2.0).sub(1.0)
        condition[:, 1] = 1.0
    return tokens, condition, labels


def _grad_norm(parameters) -> float:
    total = sum(
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    )
    return float(total.sqrt()) if not isinstance(total, int) else 0.0


def _train(
    model: nn.Module,
    *,
    steps: int,
    batch_size: int,
    d_model: int,
    chunk_tokens: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    log_every: int,
    log_path: Path,
    is_memory_model: bool,
    contextual: bool = False,
) -> dict:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    nonzero_gradient_updates = {
        "content_decoder": 0,
        "write_gate": 0,
        "read_scale": 0,
        "state_projector": 0,
        "mamba": 0,
    }
    named_groups = {}
    if is_memory_model:
        patterns = {
            "content_decoder": "content_memory.content_decoder",
            "write_gate": "content_memory.write_gate",
            "read_scale": "content_memory.read_gate",
            "state_projector": "state_projector",
            "mamba": "mamba_path",
        }
        named_groups = {
            key: [
                parameter
                for name, parameter in model.named_parameters()
                if marker in name and parameter.requires_grad
            ]
            for key, marker in patterns.items()
        }
    started = time.perf_counter()
    last_audit = {}
    model.train()
    for step in range(1, steps + 1):
        tokens, condition, labels = _make_batch(
            batch_size,
            d_model,
            chunk_tokens,
            device=device,
            generator=generator,
            contextual=contextual,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, audit = (
            model(tokens, condition=condition)
            if is_memory_model
            else model(tokens)
        )
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite benchmark loss at step {step}")
        loss.backward()
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError(f"missing/non-finite benchmark gradient at step {step}")
        group_norms = {
            name: _grad_norm(group) for name, group in named_groups.items()
        }
        for name, value in group_norms.items():
            nonzero_gradient_updates[name] += int(value > 0.0)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        last_audit = audit
        if step == 1 or step % log_every == 0 or step == steps:
            record = {
                "step": step,
                "loss": losses[-1],
                "accuracy": float((logits.argmax(dim=-1) == labels).float().mean()),
                "grad_norm": float(total_grad_norm),
                "group_grad_norms": group_norms,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    return {
        "first_20_mean_loss": sum(losses[:20]) / min(20, len(losses)),
        "last_20_mean_loss": sum(losses[-20:]) / min(20, len(losses)),
        "all_finite": True,
        "nonzero_gradient_updates": nonzero_gradient_updates,
        "last_forward_audit": last_audit,
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    intervention: str | None = None,
    condition: torch.Tensor | None = None,
) -> dict:
    model.eval()
    if intervention is None:
        if condition is None:
            logits, audit = model(tokens)
        else:
            logits, audit = model(tokens, condition=condition)
    else:
        logits, audit = model(
            tokens, intervention=intervention, condition=condition
        )
    probabilities = logits.float().softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "mean_confidence": float(probabilities.max(dim=-1).values.mean()),
        "predictions": predictions.cpu().tolist(),
        "probability_cue_b": probabilities[:, 1].cpu().tolist(),
        "forward_audit": audit,
    }


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_samples <= 0:
        raise ValueError("steps, batch size, and eval samples must be positive")
    divisor = 4 if args.contextual else 2
    if args.batch_size % divisor or args.eval_samples % divisor:
        raise ValueError(f"batch size and eval samples must be divisible by {divisor}")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "git_sha": _git_sha(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "task": (
            "W1 cue XOR semantic context; W2-W7 distractors; W8 classification"
            if args.contextual
            else "W1 binary cue; W2-W7 matched distractors; W8 cue classification"
        ),
        "isolation": {
            "write_windows": [1],
            "direct_mamba_readout_scale": 0.0,
            "state_address_scale": 0.0,
            "attention_window_chunks": 4,
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    memory_model = DelayedCueModel(
        d_model=args.d_model,
        state_summary_dim=args.state_summary_dim,
        memory_slots=args.memory_slots,
        device=device,
        contextual=args.contextual,
    )
    memory_training = _train(
        memory_model,
        steps=args.steps,
        batch_size=args.batch_size,
        d_model=args.d_model,
        chunk_tokens=args.chunk_tokens,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
        device=device,
        log_every=args.log_every,
        log_path=output / "memory_train.jsonl",
        is_memory_model=True,
        contextual=args.contextual,
    )
    torch.save(memory_model.state_dict(), output / "learned_memory_model.pt")

    minimal = MinimalControl(d_model=args.d_model, device=device)
    minimal_training = _train(
        minimal,
        steps=args.steps,
        batch_size=args.batch_size,
        d_model=args.d_model,
        chunk_tokens=args.chunk_tokens,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
        device=device,
        log_every=args.log_every,
        log_path=output / "minimal_train.jsonl",
        is_memory_model=False,
        contextual=args.contextual,
    )
    torch.save(minimal.state_dict(), output / "minimal_w4_model.pt")

    eval_generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    tokens, condition, labels = _make_batch(
        args.eval_samples,
        args.d_model,
        args.chunk_tokens,
        device=device,
        generator=eval_generator,
        contextual=args.contextual,
    )
    interventions = (
        "correct",
        "zero_delta_at_W1",
        "permute_delta_at_W1",
        "write_off_at_W1",
        "readout_off_after_W4",
        "permute_memory_after_W4",
        "direct_mamba_readout_off_after_W4",
        "state_address_off_after_W4",
    ) + (("permute_context_at_W1",) if args.contextual else ())
    evaluations = {
        name: _evaluate(
            memory_model,
            tokens,
            labels,
            intervention=name,
            condition=condition,
        )
        for name in interventions
    }
    evaluations["minimal_W4_control"] = _evaluate(minimal, tokens, labels)
    core_ablations = (
        "write_off_at_W1",
        "readout_off_after_W4",
        "permute_memory_after_W4",
    )
    correct_accuracy = evaluations["correct"]["accuracy"]
    best_causal_ablation = max(
        evaluations[name]["accuracy"] for name in core_ablations
    )
    gate = {
        "correct_accuracy_at_least_0_90": correct_accuracy >= 0.90,
        "correct_minus_best_core_causal_ablation_at_least_0_25": (
            correct_accuracy - best_causal_ablation >= 0.25
        ),
        "at_least_one_read_write_or_permute_drop": any(
            correct_accuracy - evaluations[name]["accuracy"] >= 0.25
            for name in core_ablations
        ),
        "W1_evicted_before_W8": (
            evaluations["correct"]["forward_audit"]["attention_cache_chunks"][-1]
            == 4
        ),
        "fixed_memory_capacity": args.memory_slots == 2,
    }
    if args.contextual:
        gate["context_permutation_drop_at_least_0_25"] = (
            correct_accuracy
            - evaluations["permute_context_at_W1"]["accuracy"]
            >= 0.25
        )
    gate["status"] = "PASS" if all(gate.values()) else "FAIL"
    result = {
        "schema_version": "d2_learned_causal_memory_v1",
        "config": config,
        "training": {
            "memory": memory_training,
            "minimal_W4": minimal_training,
        },
        "labels": labels.cpu().tolist(),
        "evaluations": evaluations,
        "gate": gate,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "artifacts": {
            "memory_checkpoint_sha256": _sha256(output / "learned_memory_model.pt"),
            "minimal_checkpoint_sha256": _sha256(output / "minimal_w4_model.pt"),
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"gate": gate, "accuracies": {
        key: value["accuracy"] for key, value in evaluations.items()
    }}, indent=2))
    if gate["status"] != "PASS":
        raise RuntimeError("learned delayed-cue mechanism gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
