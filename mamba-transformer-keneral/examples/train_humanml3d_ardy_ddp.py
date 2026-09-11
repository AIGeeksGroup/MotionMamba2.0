"""Resumable multi-GPU HumanML3D/control trainer for ARDY/Minimal/Hybrid."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))
sys.path.insert(0, str(REPO / "examples"))
for path in ("/data3/zts/packages/state-spaces-mamba", "/data3/zts/packages/mamba_op_triton"):
    sys.path.insert(0, path)

from ardy_data import HumanML3DArdyDataset  # noqa: E402
from ardy_data.humanml3d import long_horizon_v1_window_range  # noqa: E402
from train_humanml3d_ardy_tiny import (  # noqa: E402
    _build_batch,
    _differentiable_rollout_surrogate,
    _evaluate_many,
    _loss,
    _load_denoiser_weights,
    _prepare_generated_history,
    _replace_stats,
    _unwrap_block,
    _unwrap_raw,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--text-root", default=None)
    parser.add_argument("--text-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["original", "minimal", "hybrid"], required=True)
    parser.add_argument("--model", default="core")
    parser.add_argument(
        "--train-scope",
        choices=["mixer", "denoiser", "recurrent", "state_memory"],
        default="denoiser",
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-rank batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--override-resume-learning-rate",
        action="store_true",
        help="Apply --learning-rate after loading optimizer state from --resume.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=0,
        help="Keep only the newest N resume checkpoints in this output directory; 0 keeps all.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-final-resume-checkpoint", action="store_true")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--history-probability", type=float, default=0.0)
    parser.add_argument("--constraint-dropout", type=float, default=0.1)
    parser.add_argument("--text-dropout", type=float, default=0.1)
    parser.add_argument(
        "--sequential-cache-probability",
        type=float,
        default=0.0,
        help="Probability that an entire global batch trains a teacher-forced second window with committed history.",
    )
    parser.add_argument(
        "--sequential-rollout-min-windows",
        type=int,
        default=2,
        help="Minimum total windows in a sequential cache training batch.",
    )
    parser.add_argument(
        "--sequential-rollout-windows",
        type=int,
        default=2,
        help="Maximum total windows; all but the last are teacher-history commits.",
    )
    parser.add_argument(
        "--sequential-long-rollout-probability",
        type=float,
        default=1.0,
        help="Within sequential batches, use min/max rollout depth with this probability; otherwise use 2 windows.",
    )
    parser.add_argument(
        "--sequential-rollout-curriculum",
        choices=[
            "uniform_range",
            "long_horizon_v1",
            "generated_gate_v1",
            "generated_deep_v3",
            "generated_multidepth_v4",
            "d2_three_window_v1",
            "d2_post_eviction_v1",
        ],
        default="uniform_range",
        help=(
            "uniform_range preserves the legacy 2-vs-[min,max] sampler; "
            "long_horizon_v1 uses 50%% x2, 20%% x4, 15%% x[6,10], and 15%% x[12,19]; "
            "generated_gate_v1 uses 50%% x2, 20%% x3, 20%% x4, and 10%% x5; "
            "generated_deep_v3 uses the same teacher branch and generated targets "
            "at x3/x5/x10/x19; generated_multidepth_v4 biases generated targets "
            "toward x10/x19 while retaining the same teacher branch; "
            "d2_three_window_v1 fixes seed/generated/target for D2 write training; "
            "d2_post_eviction_v1 fixes W1-W5 accepted history with a W6 target."
        ),
    )
    parser.add_argument(
        "--generated-history-probability-1k", type=float, default=0.0,
        help="Scheduled-sampling probability reached at step 1000.",
    )
    parser.add_argument(
        "--generated-history-probability-5k", type=float, default=0.0,
        help="Scheduled-sampling probability reached at step 5000.",
    )
    parser.add_argument(
        "--generated-history-probability-start", type=float, default=0.0,
        help="Scheduled-sampling probability at step zero.",
    )
    parser.add_argument(
        "--history-sampling-strategy",
        choices=["legacy_conditional", "balanced_v2", "deep_v3", "multidepth_v4"],
        default="legacy_conditional",
    )
    parser.add_argument(
        "--short-window-probability", type=float, default=0.2,
        help="For balanced_v2/deep_v3, fixed probability of a standard short-window update.",
    )
    parser.add_argument("--generated-history-diffusion-steps", type=int, default=10)
    parser.add_argument("--generated-history-cfg-weight-text", type=float, default=2.0)
    parser.add_argument("--generated-history-cfg-weight-constraint", type=float, default=2.0)
    parser.add_argument(
        "--generated-history-max-recovery-meters",
        type=float,
        default=0.0,
        help="Cap accumulated world-frame recovery requested within one supervised window; 0 disables.",
    )
    parser.add_argument(
        "--generated-history-depth-ramp-steps",
        type=int,
        default=0,
        help="For deep_v3, progressively unlock x10/x19 over this many optimizer steps.",
    )
    parser.add_argument(
        "--generated-history-periodic-reset-windows", type=int, default=0,
        help="Reset only persistent sequence state every K generated windows; 0 disables.",
    )
    parser.add_argument(
        "--generated-history-max-supervision-points",
        type=int,
        default=1,
        help=(
            "Maximum next-window targets per on-policy rollout. Values above one "
            "enable multi-depth supervision at early, window-five, and final prefixes."
        ),
    )
    parser.add_argument(
        "--generated-history-bptt-windows",
        type=int,
        default=0,
        help=(
            "Retain gradients through this many newest accepted history writes "
            "per supervised depth; 0 uses detached state snapshots."
        ),
    )
    parser.add_argument(
        "--generated-history-attach-seed-write",
        action="store_true",
        help="Retain gradients through the seed accepted write when BPTT starts at depth zero.",
    )
    parser.add_argument(
        "--generated-history-state-consistency-weight",
        type=float,
        default=0.0,
        help="Hybrid-only normalized Mamba-state consistency loss weight.",
    )
    parser.add_argument("--mamba-output-decay-start-windows", type=int, default=0)
    parser.add_argument("--mamba-output-decay-half-life-windows", type=float, default=0.0)
    parser.add_argument("--mamba-output-decay-min-scale", type=float, default=0.0)
    parser.add_argument(
        "--generated-history-root-endpoint-loss-weight", type=float, default=0.0
    )
    parser.add_argument("--differentiable-rollout-probability", type=float, default=0.0)
    parser.add_argument("--differentiable-rollout-windows", type=int, default=0)
    parser.add_argument("--differentiable-rollout-diffusion-steps", type=int, default=1)
    parser.add_argument("--differentiable-rollout-warm-prefix-windows", type=int, default=0)
    parser.add_argument("--long-slope-loss-weight", type=float, default=0.0)
    parser.add_argument("--long-auc-loss-weight", type=float, default=0.0)
    parser.add_argument("--long-final-loss-weight", type=float, default=0.0)
    parser.add_argument("--long-slope-margin-cm-per-window", type=float, default=0.0)
    parser.add_argument("--rollout-gradient-checkpointing", action="store_true")
    parser.add_argument("--attention-window-chunks", type=int, default=4)
    parser.add_argument("--token-state-interaction", choices=["naive_gated_add", "state_query", "state_kv_modulation"], default="naive_gated_add")
    parser.add_argument("--state-summary-dim", type=int, default=128)
    parser.add_argument("--use-state-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--state-conditioning-scale-init", type=float, default=0.0)
    parser.add_argument(
        "--state-aware-output-mode",
        choices=["attention_only", "native_gated_add"],
        default="attention_only",
    )
    parser.add_argument("--content-memory-enabled", action="store_true")
    parser.add_argument("--content-memory-slots", type=int, default=1)
    parser.add_argument("--content-memory-read-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--content-memory-use-state-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--content-memory-write-gate-context",
        choices=["previous_memory", "proposal"],
        default="previous_memory",
    )
    parser.add_argument("--content-memory-decoder-init-scale", type=float, default=0.02)
    parser.add_argument("--content-memory-write-gate-init-scale", type=float, default=0.02)
    parser.add_argument("--content-memory-write-gate-bias-init", type=float, default=0.0)
    parser.add_argument("--content-memory-read-scale-init", type=float, default=0.0)
    parser.add_argument(
        "--content-memory-conditioning", choices=["none", "text"], default="none"
    )
    parser.add_argument(
        "--content-memory-normalize-delta",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--content-memory-normalize-slots",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--content-memory-max-write-rate", type=float, default=1.0)
    parser.add_argument("--content-memory-intervention", choices=["normal", "zero_delta", "permute_delta", "write_off", "readout_off", "permute_memory"], default="normal")
    parser.add_argument(
        "--minimal-ff-size", type=int, default=None,
        help="Override Minimal FFN width; 5248 is the registered Hybrid parameter match.",
    )
    parser.add_argument(
        "--training-state-intervention",
        choices=["carry", "fresh_each_window"],
        default="carry",
    )
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument(
        "--eval-protocol",
        choices=["explicit_history", "cache_only", "both"],
        default="both",
        help="Evaluation protocol is independent of sequential training probability.",
    )
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--root-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--constraint-loss-weight", type=float, default=0.0)
    parser.add_argument("--fk-constraint-loss-weight", type=float, default=0.0)
    parser.add_argument("--foot-skate-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-final-eval-loss", type=float, default=None)
    parser.add_argument("--min-root-output-weight-norm", type=float, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--original-initialization",
        choices=["fresh_mixer", "released_mixer"],
        default="fresh_mixer",
        help=(
            "Original-only initialization. fresh_mixer rebuilds the native "
            "Transformer mixer while retaining the same compatible frontend "
            "initialization used by Minimal/Hybrid."
        ),
    )
    return parser.parse_args()


def _fresh_original_block(raw, source):
    """Rebuild native ARDY with a fresh mixer and matched shared frontend."""
    from ardy.model.backbone import TransformerEncoderBlock
    from ardy_adapter.integration import _copy_compatible_frontend

    target = TransformerEncoderBlock(
        input_dim=source.input_dim,
        output_dim=source.output_dim,
        skeleton=raw.motion_rep.skeleton,
        llm_shape=source.llm_shape,
        use_text_mask=source.use_text_mask,
        latent_dim=source.latent_dim,
        ff_size=source.ff_size,
        num_layers=source.num_layers,
        num_heads=source.num_heads,
        activation=source.activation,
        dropout=source.dropout,
        pe_dropout=source.pe_dropout,
        norm_first=source.norm_first,
        input_first_heading_angle=source.input_first_heading_angle,
        add_input_proj=source.add_input_proj,
        positional_encoding_mode=source.positional_encoding_mode,
    )
    _copy_compatible_frontend(source, target)
    return target.to(device=next(raw.parameters()).device)


def _mean_across_ranks(value: torch.Tensor) -> float:
    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return float(reduced)


D2_GRADIENT_GROUP_PATTERNS = {
    "state_projector_grad_norm": (".state_projector.",),
    "state_address_grad_norm": (".token_state_interaction.",),
    "content_decoder_grad_norm": (".content_memory.content_decoder.",),
    "write_gate_grad_norm": (".content_memory.write_gate.",),
    "read_scale_grad_norm": (".content_memory.read_gate",),
}


def _d2_gradient_groups(raw) -> dict[str, list[torch.nn.Parameter]]:
    groups = {}
    for group, patterns in D2_GRADIENT_GROUP_PATTERNS.items():
        groups[group] = [
            parameter
            for name, parameter in raw.named_parameters()
            if parameter.requires_grad and any(pattern in name for pattern in patterns)
        ]
    return groups


def _gradient_group_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    tensors = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not tensors:
        device = parameters[0].device if parameters else torch.device("cpu")
        return torch.zeros((), device=device)
    return torch.stack([gradient.square().sum() for gradient in tensors]).sum().sqrt()


D2_MEMORY_DIAGNOSTIC_KEYS = (
    "memory_slot_norm",
    "memory_update_norm",
    "write_gate_mean",
    "write_gate_min",
    "write_gate_max",
    "read_residual_scale",
    "state_summary_norm",
    "state_delta_norm",
    "content_proposal_norm",
    "augmented_minus_base_attention_norm",
)


def _d2_memory_diagnostics(raw, device: torch.device) -> dict[str, float | None]:
    model = _unwrap_raw(raw)
    collected = {key: [] for key in D2_MEMORY_DIAGNOSTIC_KEYS}
    for block in (_unwrap_block(model.root_model), _unwrap_block(model.body_model)):
        for layer in block.sequence_stack.layers:
            diagnostic = layer.mixer.last_content_memory_diagnostics
            for key in D2_MEMORY_DIAGNOSTIC_KEYS:
                value = diagnostic.get(key)
                if value is not None:
                    collected[key].append(float(value))
    result = {}
    for key, values in collected.items():
        if not values:
            result[key] = None
            continue
        local = torch.tensor(sum(values) / len(values), device=device)
        result[key] = _mean_across_ranks(local)
    return result


def _ddp_no_sync_context(raw):
    """Disable gradient synchronization for every DDP wrapper in one raw pass."""
    stack = ExitStack()
    if isinstance(raw, DDP):
        stack.enter_context(raw.no_sync())
        return stack
    model = _unwrap_raw(raw)
    for block in (model.root_model, model.body_model):
        if isinstance(block, DDP):
            stack.enter_context(block.no_sync())
    return stack


def summarize_generated_history_supervision(audit: dict) -> dict:
    """Validate counter conservation and expose per-depth mean diagnostics."""
    counts = {str(key): int(value) for key, value in audit["depth_counts"].items()}
    target_count = int(audit["target_count"])
    if sum(counts.values()) != target_count:
        raise RuntimeError("generated-history supervision target counter mismatch")
    result = {
        "target_count": target_count,
        "depth_counts": counts,
        "depth_mean_loss": {},
        "depth_mean_raw_recovery_m": {},
        "depth_mean_applied_recovery_m": {},
    }
    mappings = (
        ("depth_loss_sums", "depth_mean_loss"),
        ("depth_raw_recovery_m_sums", "depth_mean_raw_recovery_m"),
        ("depth_applied_recovery_m_sums", "depth_mean_applied_recovery_m"),
    )
    for source_key, target_key in mappings:
        source = {str(key): float(value) for key, value in audit[source_key].items()}
        if set(source) != set(counts):
            raise RuntimeError(f"generated-history audit keys disagree for {source_key}")
        result[target_key] = {
            depth: source[depth] / counts[depth] for depth in sorted(counts)
        }
    return result


def _global_draws(
    count: int,
    dataset_size: int,
    generator: torch.Generator,
    device: torch.device,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rank == 0:
        indices = torch.randint(dataset_size, (count,), generator=generator, dtype=torch.long).to(device)
        seeds = torch.randint(0, 2**63 - 1, (count,), generator=generator, dtype=torch.long).to(device)
    else:
        indices = torch.empty(count, dtype=torch.long, device=device)
        seeds = torch.empty(count, dtype=torch.long, device=device)
    dist.broadcast(indices, src=0)
    dist.broadcast(seeds, src=0)
    return indices, seeds


RESUME_CONTRACT_KEYS = (
    "mode", "model", "world_size", "batch_size", "learning_rate", "seed",
    "frames", "history_probability", "constraint_dropout", "text_dropout",
    "differentiable_rollout_probability", "differentiable_rollout_windows",
    "differentiable_rollout_diffusion_steps",
    "differentiable_rollout_warm_prefix_windows", "long_slope_loss_weight",
    "long_auc_loss_weight", "long_final_loss_weight",
    "long_slope_margin_cm_per_window", "attention_window_chunks",
    "token_state_interaction", "state_summary_dim", "use_state_delta",
    "state_conditioning_scale_init", "state_aware_output_mode",
    "content_memory_enabled", "content_memory_slots",
    "content_memory_read_enabled", "content_memory_use_state_delta",
    "content_memory_write_gate_context",
    "content_memory_decoder_init_scale",
    "content_memory_write_gate_init_scale",
    "content_memory_write_gate_bias_init",
    "content_memory_read_scale_init",
    "content_memory_conditioning", "content_memory_normalize_delta",
    "content_memory_normalize_slots", "content_memory_max_write_rate",
    "generated_history_attach_seed_write",
    "minimal_ff_size", "training_state_intervention",
    "mamba_output_decay_start_windows",
    "mamba_output_decay_half_life_windows",
    "mamba_output_decay_min_scale",
)


def _assert_resume_contract(current: dict, saved: dict) -> None:
    mismatches = {
        key: {"saved": saved.get(key), "current": current.get(key)}
        for key in RESUME_CONTRACT_KEYS
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(f"resume contract mismatch: {mismatches}")


def _global_probability_draw(
    probability: float,
    generator: torch.Generator,
    device: torch.device,
    rank: int,
) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("--sequential-cache-probability must be in [0, 1]")
    if probability in (0.0, 1.0):
        return bool(probability)
    if rank == 0:
        draw = (torch.rand((), generator=generator) < probability).to(device)
    else:
        draw = torch.empty((), dtype=torch.bool, device=device)
    dist.broadcast(draw, src=0)
    return bool(draw.item())


def _global_integer_draw(
    low: int,
    high: int,
    generator: torch.Generator,
    device: torch.device,
    rank: int,
) -> int:
    if low > high:
        raise ValueError("integer draw lower bound exceeds upper bound")
    if rank == 0:
        value = torch.randint(low, high + 1, (1,), generator=generator, dtype=torch.long).to(device)
    else:
        value = torch.empty(1, dtype=torch.long, device=device)
    dist.broadcast(value, src=0)
    return int(value.item())


def _global_uniform_draw(
    generator: torch.Generator,
    device: torch.device,
    rank: int,
) -> float:
    if rank == 0:
        value = torch.rand((), generator=generator).to(device)
    else:
        value = torch.empty((), device=device)
    dist.broadcast(value, src=0)
    return float(value.item())


def generated_gate_v1_window_range(draw: float) -> tuple[int, int]:
    if not 0.0 <= draw < 1.0:
        raise ValueError("curriculum draw must be in [0, 1)")
    if draw < 0.50:
        return 2, 2
    if draw < 0.70:
        return 3, 3
    if draw < 0.90:
        return 4, 4
    return 5, 5


def generated_gate_v2_window_range(draw: float) -> tuple[int, int]:
    """Generated branch only: 40% x3, 40% x4, and 20% x5 windows."""
    if not 0.0 <= draw < 1.0:
        raise ValueError("curriculum draw must be in [0, 1)")
    if draw < 0.40:
        return 3, 3
    if draw < 0.80:
        return 4, 4
    return 5, 5


def generated_deep_v3_window_range(
    draw: float, *, step: int = 1000, ramp_steps: int = 0
) -> tuple[int, int]:
    """Generated branch: supervise targets at about 6, 10, 21, or 40 seconds."""
    if not 0.0 <= draw < 1.0:
        raise ValueError("curriculum draw must be in [0, 1)")
    if step < 0 or ramp_steps < 0:
        raise ValueError("step and ramp_steps must be non-negative")
    if ramp_steps and step <= ramp_steps // 2:
        return (3, 3) if draw < 0.60 else (5, 5)
    if ramp_steps and step <= ramp_steps:
        if draw < 0.40:
            return 3, 3
        return (5, 5) if draw < 0.75 else (10, 10)
    if draw < 0.40:
        return 3, 3
    if draw < 0.70:
        return 5, 5
    if draw < 0.90:
        return 10, 10
    return 19, 19


def generated_multidepth_v4_window_range(draw: float) -> tuple[int, int]:
    """Generated branch: 10% x3, 15% x5, 35% x10, and 40% x19."""
    if not 0.0 <= draw < 1.0:
        raise ValueError("curriculum draw must be in [0, 1)")
    if draw < 0.10:
        return 3, 3
    if draw < 0.25:
        return 5, 5
    if draw < 0.60:
        return 10, 10
    return 19, 19


def generated_history_probability(
    step: int,
    probability_start: float,
    probability_1k: float,
    probability_5k: float,
) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    if not all(
        0.0 <= value <= 1.0
        for value in (probability_start, probability_1k, probability_5k)
    ):
        raise ValueError("generated-history probabilities must be in [0, 1]")
    if probability_5k < probability_1k or probability_1k < probability_start:
        raise ValueError("generated-history probability schedule must be non-decreasing")
    if step <= 1000:
        return probability_start + (
            probability_1k - probability_start
        ) * step / 1000.0
    if step <= 5000:
        fraction = (step - 1000) / 4000.0
        return probability_1k + fraction * (probability_5k - probability_1k)
    return probability_5k


def _eligible_indices(dataset: HumanML3DArdyDataset, required_frames: int) -> list[int]:
    return [
        index
        for index, record in enumerate(dataset.records)
        if int(record.get("frames", 0)) >= required_frames
    ]


def _save_checkpoint(
    path: Path,
    *,
    args,
    raw,
    optimizer,
    step: int,
    losses: list[float],
    initial_eval: float,
    all_finite: bool,
    sequential_cache_steps: int,
    sequential_rollout_steps: int,
    generated_history_steps: int,
    generated_history_windows: int,
    generated_history_supervision_audit: dict,
    teacher_history_steps: int,
    short_window_steps: int,
    rollout_window_counts: dict[int, int],
    differentiable_rollout_steps: int,
    d2_gradient_update_counts: dict[str, int],
    data_generator: torch.Generator,
    elapsed_seconds: float,
    device: torch.device,
    rank: int,
) -> None:
    local_rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }
    rng_by_rank = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rng_by_rank, local_rng)
    if rank == 0:
        model = _unwrap_raw(raw)
        root = _unwrap_block(model.root_model)
        body = _unwrap_block(model.body_model)
        auxiliary = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith(("root_model.", "body_model."))
        }
        checkpoint = {
            "format_version": 2,
            "mode": args.mode,
            "model": args.model,
            "phase": "ddp",
            "world_size": dist.get_world_size(),
            "step": step,
            "root_model": root.state_dict(),
            "body_model": body.state_dict(),
            "denoiser_aux": auxiliary,
            "optimizer": optimizer.state_dict(),
            "losses": losses,
            "initial_eval_loss": initial_eval,
            "all_gradients_finite": all_finite,
            "sequential_cache_steps": sequential_cache_steps,
            "sequential_rollout_steps": sequential_rollout_steps,
            "generated_history_steps": generated_history_steps,
            "generated_history_windows": generated_history_windows,
            "generated_history_supervision_audit": generated_history_supervision_audit,
            "teacher_history_steps": teacher_history_steps,
            "short_window_steps": short_window_steps,
            "rollout_window_counts": rollout_window_counts,
            "differentiable_rollout_steps": differentiable_rollout_steps,
            "d2_gradient_update_counts": d2_gradient_update_counts,
            "data_generator_state": data_generator.get_state(),
            "rng_by_rank": rng_by_rank,
            "elapsed_seconds": elapsed_seconds,
            "config": vars(args),
            "commit_protocol_version": (
                "d2_generated_history_bptt_v1"
                if args.train_scope == "state_memory"
                else "generated_history_endpoint_v9"
                if args.generated_history_root_endpoint_loss_weight > 0
                else "generated_history_state_consistency_v6"
                if args.generated_history_state_consistency_weight > 0
                else "generated_history_decay_aligned_v7"
                if args.mamba_output_decay_half_life_windows > 0
                else "generated_history_bptt_v5"
                if args.generated_history_bptt_windows > 0
                else "generated_history_multidepth_v4"
                if args.generated_history_max_supervision_points > 1
                and (
                    args.generated_history_probability_start > 0
                    or args.generated_history_probability_1k > 0
                    or args.generated_history_probability_5k > 0
                )
                else "generated_history_deep_v3"
                if args.history_sampling_strategy == "deep_v3"
                else
                "generated_history_balanced_v2"
                if args.history_sampling_strategy == "balanced_v2"
                else "generated_history_scheduled_v1"
                if args.generated_history_probability_start > 0
                or args.generated_history_probability_1k > 0
                or args.generated_history_probability_5k > 0
                else "canonical_history_v2"
            ),
            "production_contract": {
                "motion_mask_mode": getattr(model, "motion_mask_mode", None),
                "positional_encoding_mode": getattr(model, "positional_encoding_mode", None),
                "root_input_first_heading_angle": getattr(model.root_model, "input_first_heading_angle", None),
                "body_input_first_heading_angle": getattr(model.body_model, "input_first_heading_angle", None),
                "num_frames_per_token": int(model.num_frames_per_token),
                "gen_horizon_len": int(getattr(model, "gen_horizon_len", 0)),
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    dist.barrier()


def _prune_checkpoints(output: Path, keep: int, rank: int) -> None:
    if keep < 0:
        raise ValueError("--keep-last-checkpoints must be non-negative")
    if keep and rank == 0:
        checkpoints = sorted(output.glob("checkpoint_step_*.pt"))
        for checkpoint in checkpoints[:-keep]:
            checkpoint.unlink()
    dist.barrier()


def _load_model_weights(raw, checkpoint: dict) -> None:
    _load_denoiser_weights(raw, checkpoint)


def main() -> int:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)

    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model(args.model, device=str(device), text_encoder=False)
    generation_horizon = int(model.gen_horizon_len)
    max_rollout_windows = {
        "long_horizon_v1": 19,
        "generated_gate_v1": 5,
        "generated_deep_v3": 19,
        "generated_multidepth_v4": 19,
        "d2_three_window_v1": 3,
        "d2_post_eviction_v1": 6,
    }.get(args.sequential_rollout_curriculum, args.sequential_rollout_windows)
    if args.sequential_rollout_min_windows < 2:
        raise ValueError("--sequential-rollout-min-windows must be at least 2")
    if args.sequential_rollout_min_windows > args.sequential_rollout_windows:
        raise ValueError("sequential rollout minimum exceeds maximum")
    generated_sampling_enabled = (
        args.generated_history_probability_start > 0
        or
        args.generated_history_probability_1k > 0
        or args.generated_history_probability_5k > 0
    )
    expected_curriculum = (
        (
            "d2_post_eviction_v1"
            if args.generated_history_attach_seed_write
            else "d2_three_window_v1"
        )
        if args.train_scope == "state_memory"
        else "generated_multidepth_v4"
        if args.history_sampling_strategy == "multidepth_v4"
        else "generated_deep_v3"
        if args.history_sampling_strategy == "deep_v3"
        else "generated_gate_v1"
    )
    if generated_sampling_enabled and args.sequential_rollout_curriculum != expected_curriculum:
        raise ValueError(
            f"{args.history_sampling_strategy} generated-history sampling requires "
            f"{expected_curriculum} curriculum"
        )
    generated_history_probability(
        0,
        args.generated_history_probability_start,
        args.generated_history_probability_1k,
        args.generated_history_probability_5k,
    )
    if not 0.0 <= args.short_window_probability <= 1.0:
        raise ValueError("short-window-probability must be in [0, 1]")
    if args.generated_history_max_recovery_meters < 0:
        raise ValueError("generated-history-max-recovery-meters must be non-negative")
    if args.generated_history_depth_ramp_steps < 0:
        raise ValueError("generated-history-depth-ramp-steps must be non-negative")
    if args.generated_history_max_supervision_points <= 0:
        raise ValueError("generated-history-max-supervision-points must be positive")
    if args.generated_history_bptt_windows < 0:
        raise ValueError("generated-history-bptt-windows must be non-negative")
    if args.generated_history_bptt_windows and args.train_scope == "mixer":
        raise ValueError("generated-history BPTT requires outer denoiser DDP")
    if args.train_scope == "state_memory":
        if args.mode != "hybrid":
            raise ValueError("state_memory train scope requires Hybrid mode")
        if (
            not args.content_memory_enabled
            and args.token_state_interaction
            not in {"state_query", "state_kv_modulation"}
        ):
            raise ValueError(
                "state_memory train scope requires state_query/state_kv_modulation "
                "or persistent content memory"
            )
        if args.generated_history_bptt_windows <= 0:
            raise ValueError("state_memory train scope requires generated-history BPTT")
        if args.generated_history_attach_seed_write and args.generated_history_bptt_windows < 1:
            raise ValueError("attached seed writes require generated-history BPTT")
        if (
            args.generated_history_attach_seed_write
            and args.generated_history_bptt_windows
            < args.sequential_rollout_windows - 2
        ):
            raise ValueError(
                "attached seed writes require BPTT to start at depth zero"
            )
    if not 0.0 < args.content_memory_max_write_rate <= 1.0:
        raise ValueError("content-memory-max-write-rate must be in (0, 1]")
    if args.generated_history_state_consistency_weight < 0:
        raise ValueError("generated-history-state-consistency-weight must be non-negative")
    if (
        args.generated_history_state_consistency_weight
        and not args.generated_history_bptt_windows
    ):
        raise ValueError("state consistency requires generated-history BPTT")
    if (
        args.mamba_output_decay_start_windows < 0
        or args.mamba_output_decay_half_life_windows < 0
    ):
        raise ValueError("Mamba output decay settings must be non-negative")
    if not 0.0 <= args.mamba_output_decay_min_scale <= 1.0:
        raise ValueError("Mamba output decay minimum scale must be in [0, 1]")
    if args.generated_history_root_endpoint_loss_weight < 0:
        raise ValueError("generated-history-root-endpoint-loss-weight must be non-negative")
    if not 0.0 <= args.differentiable_rollout_probability <= 1.0:
        raise ValueError("differentiable-rollout-probability must be in [0, 1]")
    if args.differentiable_rollout_windows and not 5 <= args.differentiable_rollout_windows <= 8:
        raise ValueError("differentiable-rollout-windows must be in [5, 8]")
    if args.differentiable_rollout_diffusion_steps <= 0:
        raise ValueError("differentiable-rollout-diffusion-steps must be positive")
    if args.differentiable_rollout_warm_prefix_windows < 0:
        raise ValueError("differentiable-rollout-warm-prefix-windows must be non-negative")
    if any(value < 0 for value in (args.long_slope_loss_weight, args.long_auc_loss_weight, args.long_final_loss_weight)):
        raise ValueError("long rollout loss weights must be non-negative")
    if args.attention_window_chunks <= 0:
        raise ValueError("attention-window-chunks must be positive")
    if args.minimal_ff_size is not None and args.mode != "minimal":
        raise ValueError("minimal-ff-size is valid only for Minimal")
    if args.training_state_intervention != "carry" and args.mode != "hybrid":
        raise ValueError("training state intervention is Hybrid-only")
    if args.differentiable_rollout_probability > 0:
        if args.differentiable_rollout_windows == 0:
            raise ValueError("differentiable rollout probability requires a rollout depth")
        if args.frames < args.differentiable_rollout_windows * generation_horizon:
            raise ValueError(
                "--frames is shorter than the differentiable rollout depth"
            )
        if args.differentiable_rollout_windows == 5 and any(
            value > 0
            for value in (
                args.long_slope_loss_weight,
                args.long_auc_loss_weight,
                args.long_final_loss_weight,
            )
        ):
            raise ValueError("post-window-5 losses require at least 6 rollout windows")
    if (
        args.sequential_cache_probability > 0 or args.eval_protocol in {"cache_only", "both"}
    ) and args.frames < max_rollout_windows * generation_horizon:
        raise ValueError(
            f"--frames must be at least {max_rollout_windows * generation_horizon} "
            "for the configured sequential rollout"
        )
    _replace_stats(model, args.stats_dir)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.mode == "original":
        if args.original_initialization == "fresh_mixer":
            raw = model.denoiser.model
            raw.root_model = _fresh_original_block(raw, raw.root_model)
            raw.body_model = _fresh_original_block(raw, raw.body_model)
        # Keep ARDY's native root/body Transformer architecture. It has no
        # adapter-owned persistent cache.
        client = model
    else:
        client = install_ardy_sequence_kernel(
            model,
            ArdySequenceKernelConfig(
                mode=args.mode,
                attention_backend="sdpa",
                window_size_chunks=args.attention_window_chunks,
                num_sink_tokens=4,
                ff_size=args.minimal_ff_size,
                token_state_interaction=args.token_state_interaction,
                state_summary_dim=args.state_summary_dim,
                use_state_delta=args.use_state_delta,
                state_conditioning_scale_init=args.state_conditioning_scale_init,
                state_aware_output_mode=args.state_aware_output_mode,
                content_memory_enabled=args.content_memory_enabled,
                content_memory_slots=args.content_memory_slots,
                content_memory_read_enabled=args.content_memory_read_enabled,
                content_memory_use_state_delta=args.content_memory_use_state_delta,
                content_memory_write_gate_context=args.content_memory_write_gate_context,
                content_memory_decoder_init_scale=args.content_memory_decoder_init_scale,
                content_memory_write_gate_init_scale=args.content_memory_write_gate_init_scale,
                content_memory_write_gate_bias_init=args.content_memory_write_gate_bias_init,
                content_memory_read_scale_init=args.content_memory_read_scale_init,
                content_memory_conditioning=args.content_memory_conditioning,
                content_memory_normalize_delta=args.content_memory_normalize_delta,
                content_memory_normalize_slots=args.content_memory_normalize_slots,
                content_memory_max_write_rate=args.content_memory_max_write_rate,
            ),
        )
        client.controller.configure_rollout_diagnostics(
            enabled=(
                args.mode == "hybrid" and generated_sampling_enabled and rank == 0
            ),
            periodic_reset_windows=args.generated_history_periodic_reset_windows,
        )
        client.controller.configure_mamba_output_decay(
            start_windows=args.mamba_output_decay_start_windows,
            half_life_windows=(
                args.mamba_output_decay_half_life_windows
                if args.mode == "hybrid" else 0.0
            ),
            min_scale=(
                args.mamba_output_decay_min_scale
                if args.mode == "hybrid" else 0.0
            ),
        )
        client.controller.configure_state_intervention(
            args.training_state_intervention
        )
        client.controller.configure_content_memory_intervention(
            args.content_memory_intervention
        )
    raw = model.denoiser.model
    if args.train_scope in {"denoiser", "recurrent", "state_memory"}:
        if args.train_scope == "denoiser":
            for parameter in raw.parameters():
                parameter.requires_grad_(True)
        elif args.train_scope == "recurrent":
            if args.mode != "hybrid":
                raise ValueError("recurrent train scope requires hybrid mode")
            for name, parameter in raw.named_parameters():
                trainable = ".mamba_path." in name or name.endswith("fusion_gate_logit")
                parameter.requires_grad_(trainable)
        else:
            for name, parameter in raw.named_parameters():
                trainable = any(
                    marker in name
                    for marker in (
                        ".state_projector.",
                        ".token_state_interaction.",
                        ".content_memory.",
                    )
                )
                parameter.requires_grad_(trainable)
        raw = DDP(raw, device_ids=[local_rank], broadcast_buffers=False)
        raw.train()
        parameters = [
            parameter for parameter in raw.parameters() if parameter.requires_grad
        ]
    else:
        for parameter in list(raw.root_model.parameters()) + list(raw.body_model.parameters()):
            parameter.requires_grad_(True)
        raw.root_model = DDP(raw.root_model, device_ids=[local_rank], broadcast_buffers=False)
        raw.body_model = DDP(raw.body_model, device_ids=[local_rank], broadcast_buffers=False)
        raw.root_model.train()
        raw.body_model.train()
        parameters = list(raw.root_model.parameters()) + list(raw.body_model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    d2_gradient_groups = (
        _d2_gradient_groups(raw) if args.train_scope == "state_memory" else {}
    )
    required_d2_groups = {"state_projector_grad_norm"}
    if args.token_state_interaction in {"state_query", "state_kv_modulation"}:
        required_d2_groups.add("state_address_grad_norm")
    if args.content_memory_enabled:
        required_d2_groups.update(
            {"content_decoder_grad_norm", "write_gate_grad_norm", "read_scale_grad_norm"}
        )
    if args.train_scope == "state_memory" and any(
        not d2_gradient_groups[name] for name in required_d2_groups
    ):
        missing = [
            name for name in sorted(required_d2_groups) if not d2_gradient_groups[name]
        ]
        raise RuntimeError(f"state_memory train scope is missing groups: {missing}")
    trainable_parameter_count = sum(parameter.numel() for parameter in parameters)
    trainable_parameter_names = (
        [name for name, parameter in raw.named_parameters() if parameter.requires_grad]
        if args.train_scope == "state_memory"
        else []
    )
    model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps, device=device))

    train = HumanML3DArdyDataset(
        args.manifest, args.stats_dir, text_root=args.text_root, frames=args.frames, horizon=generation_horizon,
        condition_dropout=args.constraint_dropout, seed=args.seed, text_cache_dir=args.text_cache,
        history_probability=args.history_probability,
        text_dropout=args.text_dropout,
    )
    dev = HumanML3DArdyDataset(
        args.dev_manifest, args.stats_dir, text_root=args.text_root, frames=args.frames, horizon=generation_horizon,
        condition_dropout=0.0, seed=args.seed + 10000, text_cache_dir=args.text_cache,
        history_probability=args.history_probability,
        text_dropout=0.0,
    )
    train_indices_by_windows = {
        windows: _eligible_indices(train, windows * generation_horizon)
        for windows in range(2, max_rollout_windows + 1)
    }
    dev_indices_by_windows = {
        windows: _eligible_indices(dev, windows * generation_horizon)
        for windows in range(2, max_rollout_windows + 1)
    }
    eval_protocols = ["explicit_history", "cache_only"] if args.eval_protocol == "both" else [args.eval_protocol]
    if args.mode == "original":
        eval_protocols = ["explicit_history"]
    dev_fixed_by_protocol = {}
    for protocol in eval_protocols:
        cache_only_eval = protocol == "cache_only"
        rollout_eval = cache_only_eval or (
            args.mode == "original" and args.sequential_cache_probability > 0
        )
        eval_rollout_windows = max_rollout_windows if rollout_eval else 1
        if rollout_eval and not dev_indices_by_windows[eval_rollout_windows]:
            raise ValueError("dev split has no samples eligible for long-rollout evaluation")
        fixed_batches = []
        for batch_index in range(args.eval_batches):
            dev_start = (batch_index * world + rank) * args.batch_size
            dev_pool = (
                dev_indices_by_windows[eval_rollout_windows]
                if rollout_eval else range(len(dev))
            )
            dev_indices = [dev_pool[(dev_start + offset) % len(dev_pool)] for offset in range(args.batch_size)]
            dev_batch = default_collate([
                dev.sample(
                    index,
                    args.seed + 10000 + index,
                    force_history=False if not rollout_eval else None,
                    force_history_frames=(
                        (eval_rollout_windows - 1) * generation_horizon
                        if rollout_eval else None
                    ),
                    out_of_horizon=False if rollout_eval else None,
                )
                for index in dev_indices
            ])
            fixed_batches.append(_build_batch(
                model, dev_batch, device, fixed_noise=True,
                boundary_loss_weight=args.boundary_loss_weight,
                root_velocity_loss_weight=args.root_velocity_loss_weight,
                constraint_loss_weight=args.constraint_loss_weight,
                fk_constraint_loss_weight=args.fk_constraint_loss_weight,
                foot_skate_loss_weight=args.foot_skate_loss_weight,
                sequential_cache=cache_only_eval,
                sequential_rollout_windows=eval_rollout_windows if rollout_eval else 2,
                rollout_total_frames=(
                    eval_rollout_windows * generation_horizon if rollout_eval else None
                ),
            ))
        dev_fixed_by_protocol[protocol] = fixed_batches
    data_generator = torch.Generator().manual_seed(args.seed + 555)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    log_path = output / "train.jsonl"

    start_step = 0
    losses: list[float] = []
    all_finite = True
    sequential_cache_steps = 0
    sequential_rollout_steps = 0
    generated_history_steps = 0
    generated_history_windows = 0
    generated_history_supervision_audit = {
        "target_count": 0,
        "depth_counts": {},
        "depth_loss_sums": {},
        "depth_raw_recovery_m_sums": {},
        "depth_applied_recovery_m_sums": {},
    }
    teacher_history_steps = 0
    short_window_steps = 0
    rollout_window_counts: dict[int, int] = {}
    differentiable_rollout_steps = 0
    d2_gradient_update_counts = {
        "updates_with_nonzero_content_decoder_grad": 0,
        "updates_with_nonzero_write_gate_grad": 0,
        "updates_with_nonzero_read_scale_grad": 0,
    }
    d2_last_memory_diagnostics = {}
    elapsed_before = 0.0
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("mode") != args.mode:
            raise ValueError("initial checkpoint mode does not match requested run")
        _load_model_weights(raw, initial)
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("mode") != args.mode or resume.get("world_size") != world:
            raise ValueError("resume checkpoint mode/world size does not match")
        current_contract = {**vars(args), "world_size": world}
        saved_contract = {
            **resume.get("config", {}),
            "world_size": resume.get("world_size"),
        }
        _assert_resume_contract(current_contract, saved_contract)
        _load_model_weights(raw, resume)
        optimizer.load_state_dict(resume["optimizer"])
        if args.override_resume_learning_rate:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
        start_step = int(resume["step"])
        losses = list(resume["losses"])
        all_finite = bool(resume["all_gradients_finite"])
        sequential_cache_steps = int(resume.get("sequential_cache_steps", 0))
        sequential_rollout_steps = int(
            resume.get("sequential_rollout_steps", sequential_cache_steps)
        )
        generated_history_steps = int(resume.get("generated_history_steps", 0))
        generated_history_windows = int(resume.get("generated_history_windows", 0))
        generated_history_supervision_audit = resume.get(
            "generated_history_supervision_audit",
            generated_history_supervision_audit,
        )
        teacher_history_steps = int(resume.get("teacher_history_steps", 0))
        short_window_steps = int(resume.get("short_window_steps", 0))
        rollout_window_counts = {
            int(key): int(value)
            for key, value in resume.get("rollout_window_counts", {}).items()
        }
        differentiable_rollout_steps = int(
            resume.get("differentiable_rollout_steps", 0)
        )
        d2_gradient_update_counts = dict(
            resume.get("d2_gradient_update_counts", d2_gradient_update_counts)
        )
        data_generator.set_state(resume["data_generator_state"].cpu())
        local_rng = resume["rng_by_rank"][rank]
        torch.set_rng_state(local_rng["torch"].cpu())
        torch.cuda.set_rng_state(local_rng["cuda"].cpu(), device)
        initial_eval = float(resume["initial_eval_loss"])
        elapsed_before = float(resume.get("elapsed_seconds", 0.0))
        if args.steps < start_step:
            raise ValueError(f"--steps must be at least resumed step {start_step}")
    else:
        initial_eval_by_protocol = {
            protocol: _mean_across_ranks(torch.tensor(_evaluate_many(raw, client, batches), device=device))
            for protocol, batches in dev_fixed_by_protocol.items()
        }
        initial_eval = float(np.mean(list(initial_eval_by_protocol.values())))
        if rank == 0 and log_path.exists():
            raise FileExistsError(f"refusing to overwrite existing log: {log_path}")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    global_batch = args.batch_size * world
    for step in range(start_step + 1, args.steps + 1):
        scheduled_probability = generated_history_probability(
            step,
            args.generated_history_probability_start,
            args.generated_history_probability_1k,
            args.generated_history_probability_5k,
        )
        if args.history_sampling_strategy in {
            "balanced_v2", "deep_v3", "multidepth_v4"
        }:
            use_short_window = _global_probability_draw(
                args.short_window_probability, data_generator, device, rank
            )
            use_sequential_cache = not use_short_window
            use_generated_history = bool(
                use_sequential_cache
                and _global_probability_draw(
                    scheduled_probability, data_generator, device, rank
                )
            )
            if use_sequential_cache:
                if use_generated_history:
                    range_fn = (
                        generated_multidepth_v4_window_range
                        if args.history_sampling_strategy == "multidepth_v4"
                        else generated_deep_v3_window_range
                        if args.history_sampling_strategy == "deep_v3"
                        else generated_gate_v2_window_range
                    )
                else:
                    range_fn = generated_gate_v1_window_range
                curriculum_draw = _global_uniform_draw(data_generator, device, rank)
                if args.history_sampling_strategy == "deep_v3" and use_generated_history:
                    low, high = range_fn(
                        curriculum_draw,
                        step=step,
                        ramp_steps=args.generated_history_depth_ramp_steps,
                    )
                else:
                    low, high = range_fn(curriculum_draw)
                rollout_windows = _global_integer_draw(
                    low, high, data_generator, device, rank
                )
            else:
                rollout_windows = 1
            use_long_rollout = rollout_windows > 2
        else:
            use_sequential_cache = _global_probability_draw(
                args.sequential_cache_probability, data_generator, device, rank
            )
            if use_sequential_cache and args.sequential_rollout_curriculum in {
                "long_horizon_v1", "generated_gate_v1", "d2_three_window_v1",
                "d2_post_eviction_v1",
            }:
                curriculum_draw = _global_uniform_draw(data_generator, device, rank)
                if args.sequential_rollout_curriculum == "long_horizon_v1":
                    range_fn = long_horizon_v1_window_range
                elif args.sequential_rollout_curriculum == "generated_gate_v1":
                    range_fn = generated_gate_v1_window_range
                elif args.sequential_rollout_curriculum == "d2_three_window_v1":
                    range_fn = lambda _draw: (3, 3)
                else:
                    range_fn = lambda _draw: (6, 6)
                low, high = range_fn(curriculum_draw)
                rollout_windows = _global_integer_draw(
                    low, high, data_generator, device, rank
                )
                use_long_rollout = rollout_windows > 2
            else:
                use_long_rollout = use_sequential_cache and _global_probability_draw(
                    args.sequential_long_rollout_probability, data_generator, device, rank
                )
                rollout_windows = (
                    _global_integer_draw(
                        args.sequential_rollout_min_windows,
                        args.sequential_rollout_windows,
                        data_generator,
                        device,
                        rank,
                    )
                    if use_long_rollout else (2 if use_sequential_cache else 1)
                )
            use_generated_history = bool(
                use_sequential_cache
                and rollout_windows >= 3
                and _global_probability_draw(
                    scheduled_probability, data_generator, device, rank
                )
            )
        use_differentiable_rollout = _global_probability_draw(
            args.differentiable_rollout_probability,
            data_generator,
            device,
            rank,
        )
        train_pool = (
            train_indices_by_windows[rollout_windows]
            if use_sequential_cache else range(len(train))
        )
        if use_sequential_cache and not train_pool:
            raise ValueError("train split has no samples eligible for sequential-cache training")
        indices, augmentation_seeds = _global_draws(
            global_batch, len(train_pool), data_generator, device, rank
        )
        indices = torch.tensor(
            [train_pool[int(index)] for index in indices.cpu().tolist()],
            dtype=torch.long,
            device=device,
        )
        local_indices = indices[rank * args.batch_size:(rank + 1) * args.batch_size].cpu().tolist()
        local_seeds = augmentation_seeds[
            rank * args.batch_size:(rank + 1) * args.batch_size
        ].cpu().tolist()
        batch = default_collate(
            [
                train.sample(
                    index,
                    seed,
                    force_history=None if use_sequential_cache else None,
                    force_history_frames=(
                        (rollout_windows - 1) * generation_horizon
                        if use_sequential_cache else None
                    ),
                    out_of_horizon=False if use_sequential_cache else None,
                )
                for index, seed in zip(local_indices, local_seeds)
            ]
        )
        # Re-seed stochastic training tensors every update so separate model
        # runs retain matched noise/dropout despite architecture-specific
        # generation consuming different numbers of random values.
        step_seed = args.seed * 10_000_000 + step * world + rank
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed(step_seed)
        fixed = _build_batch(
            model, batch, device, fixed_noise=False,
            boundary_loss_weight=args.boundary_loss_weight,
            root_velocity_loss_weight=args.root_velocity_loss_weight,
            constraint_loss_weight=args.constraint_loss_weight,
            fk_constraint_loss_weight=args.fk_constraint_loss_weight,
            foot_skate_loss_weight=args.foot_skate_loss_weight,
            sequential_cache=use_sequential_cache and args.mode != "original",
            sequential_rollout_windows=rollout_windows if use_sequential_cache else 2,
            rollout_total_frames=(
                rollout_windows * generation_horizon if use_sequential_cache else None
            ),
        )
        fixed["sequential_rollout_windows"] = rollout_windows if use_sequential_cache else 1
        fixed["generated_history"] = use_generated_history
        fixed["training_state_intervention"] = args.training_state_intervention
        if use_generated_history:
            _prepare_generated_history(
                model,
                raw,
                client,
                fixed,
                mode=args.mode,
                diffusion_steps=args.generated_history_diffusion_steps,
                cfg_weight_text=args.generated_history_cfg_weight_text,
                cfg_weight_constraint=args.generated_history_cfg_weight_constraint,
                max_recovery_meters=args.generated_history_max_recovery_meters,
                max_supervision_points=args.generated_history_max_supervision_points,
                bptt_windows=args.generated_history_bptt_windows,
                attach_seed_write=args.generated_history_attach_seed_write,
                state_consistency_weight=(
                    args.generated_history_state_consistency_weight
                    if args.mode == "hybrid" else 0.0
                ),
                root_endpoint_loss_weight=(
                    args.generated_history_root_endpoint_loss_weight
                ),
            )
            if rank == 0 and args.mode == "hybrid":
                diagnostics = client.controller.state_diagnostics
                if diagnostics:
                    with (output / "state_diagnostics.jsonl").open("a") as handle:
                        for diagnostic in diagnostics:
                            handle.write(json.dumps({"step": step, **diagnostic}) + "\n")
                    diagnostics.clear()
        supervision_batches = fixed.get(
            "generated_history_supervision_batches", [fixed]
        )
        supervision_count = len(supervision_batches)
        if supervision_count <= 0:
            raise RuntimeError("training update has no supervision target")
        optimizer.zero_grad(set_to_none=True)
        component_sums = [torch.zeros((), device=device) for _ in range(10)]
        depth_components = []
        differentiable_metrics = None
        differentiable_loss_value = torch.zeros((), device=device)
        for supervision_index, supervision in enumerate(supervision_batches):
            synchronize = supervision_index == supervision_count - 1
            sync_context = nullcontext() if synchronize else _ddp_no_sync_context(raw)
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    components = _loss(
                        raw, client, supervision, return_components=True
                    )
                    scaled_loss = components[0] / supervision_count
                    if synchronize and use_differentiable_rollout:
                        surrogate, differentiable_metrics = (
                            _differentiable_rollout_surrogate(
                                model,
                                raw,
                                client,
                                fixed,
                                mode=args.mode,
                                windows=args.differentiable_rollout_windows,
                                diffusion_steps=(
                                    args.differentiable_rollout_diffusion_steps
                                ),
                                warm_prefix_windows=(
                                    args.differentiable_rollout_warm_prefix_windows
                                ),
                                slope_weight=args.long_slope_loss_weight,
                                auc_weight=args.long_auc_loss_weight,
                                final_weight=args.long_final_loss_weight,
                                margin_cm_per_window=(
                                    args.long_slope_margin_cm_per_window
                                ),
                                gradient_checkpointing=(
                                    args.rollout_gradient_checkpointing
                                ),
                            )
                        )
                        scaled_loss = scaled_loss + surrogate
                        differentiable_loss_value = surrogate.detach()
                if not torch.isfinite(scaled_loss):
                    depth = supervision.get("generated_history_supervision_depth")
                    raise RuntimeError(
                        f"rank {rank}: non-finite loss at step {step}, depth {depth}"
                    )
                scaled_loss.backward()
            for index, component in enumerate(components):
                component_sums[index] += component.detach() / supervision_count
            if use_generated_history:
                depth_components.append((supervision, tuple(c.detach() for c in components)))
        (
            loss, root_loss, body_loss, boundary_loss, velocity_loss,
            constraint_loss, fk_constraint_loss, foot_skate_loss,
            state_consistency_loss, root_endpoint_loss,
        ) = component_sums
        loss = loss + differentiable_loss_value
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        finite = bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)
        finite_tensor = torch.tensor(int(finite), device=device)
        dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        all_finite = all_finite and bool(finite_tensor.item())
        if not all_finite:
            raise RuntimeError(f"non-finite DDP gradient at step {step}")
        d2_gradient_norms = {}
        if args.train_scope == "state_memory":
            d2_gradient_norms = {
                name: _mean_across_ranks(_gradient_group_norm(group).to(device))
                for name, group in d2_gradient_groups.items()
            }
            if rank == 0:
                counter_map = {
                    "content_decoder_grad_norm": "updates_with_nonzero_content_decoder_grad",
                    "write_gate_grad_norm": "updates_with_nonzero_write_gate_grad",
                    "read_scale_grad_norm": "updates_with_nonzero_read_scale_grad",
                }
                for metric, counter in counter_map.items():
                    d2_gradient_update_counts[counter] += int(
                        d2_gradient_norms[metric] > 0.0
                    )
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        d2_memory_diagnostics = (
            _d2_memory_diagnostics(raw, device)
            if args.train_scope == "state_memory"
            else {}
        )
        d2_last_memory_diagnostics = d2_memory_diagnostics
        mean_loss = _mean_across_ranks(loss)
        depth_metrics = {}
        for supervision, components in depth_components:
            depth = int(supervision["generated_history_supervision_depth"])
            depth_metrics[str(depth)] = {
                "loss": _mean_across_ranks(components[0]),
                "raw_recovery_m": _mean_across_ranks(
                    supervision["generated_history_raw_recovery_norm"].mean()
                ),
                "applied_recovery_m": _mean_across_ranks(
                    supervision["generated_history_applied_recovery_norm"].mean()
                ),
            }
        if rank == 0:
            losses.append(mean_loss)
            sequential_rollout_steps += int(use_sequential_cache)
            differentiable_rollout_steps += int(use_differentiable_rollout)
            generated_history_steps += int(use_generated_history)
            generated_history_windows += (
                rollout_windows - 2 if use_generated_history else 0
            )
            for depth, depth_metric in depth_metrics.items():
                audit = generated_history_supervision_audit
                audit["target_count"] += 1
                audit["depth_counts"][depth] = audit["depth_counts"].get(depth, 0) + 1
                audit["depth_loss_sums"][depth] = (
                    audit["depth_loss_sums"].get(depth, 0.0) + depth_metric["loss"]
                )
                audit["depth_raw_recovery_m_sums"][depth] = (
                    audit["depth_raw_recovery_m_sums"].get(depth, 0.0)
                    + depth_metric["raw_recovery_m"]
                )
                audit["depth_applied_recovery_m_sums"][depth] = (
                    audit["depth_applied_recovery_m_sums"].get(depth, 0.0)
                    + depth_metric["applied_recovery_m"]
                )
            teacher_history_steps += int(
                use_sequential_cache and not use_generated_history
            )
            short_window_steps += int(not use_sequential_cache)
            sequential_cache_steps += int(
                use_sequential_cache and args.mode != "original"
            )
            if use_sequential_cache:
                rollout_window_counts[rollout_windows] = (
                    rollout_window_counts.get(rollout_windows, 0) + 1
                )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            mean_grad = _mean_across_ranks(grad_norm)
            mean_root_loss = _mean_across_ranks(root_loss)
            mean_body_loss = _mean_across_ranks(body_loss)
            mean_boundary_loss = _mean_across_ranks(boundary_loss)
            mean_velocity_loss = _mean_across_ranks(velocity_loss)
            mean_constraint_loss = _mean_across_ranks(constraint_loss)
            mean_fk_constraint_loss = _mean_across_ranks(fk_constraint_loss)
            mean_foot_skate_loss = _mean_across_ranks(foot_skate_loss)
            mean_state_consistency_loss = _mean_across_ranks(
                state_consistency_loss
            )
            mean_root_endpoint_loss = _mean_across_ranks(root_endpoint_loss)
            mean_differentiable_rollout_loss = _mean_across_ranks(
                differentiable_loss_value
            )
            mean_raw_recovery = (
                float(np.mean([value["raw_recovery_m"] for value in depth_metrics.values()]))
                if depth_metrics else None
            )
            mean_applied_recovery = (
                float(np.mean([value["applied_recovery_m"] for value in depth_metrics.values()]))
                if depth_metrics else None
            )
            if rank == 0:
                record = {
                    "step": step,
                    "loss": mean_loss,
                    "root_loss": mean_root_loss,
                    "body_loss": mean_body_loss,
                    "boundary_loss": mean_boundary_loss,
                    "root_velocity_loss": mean_velocity_loss,
                    "constraint_loss": mean_constraint_loss,
                    "fk_constraint_loss": mean_fk_constraint_loss,
                    "foot_skate_loss": mean_foot_skate_loss,
                    "generated_history_state_consistency_loss": (
                        mean_state_consistency_loss
                    ),
                    "generated_history_root_endpoint_loss": mean_root_endpoint_loss,
                    "grad_norm": mean_grad,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "sequential_rollout": use_sequential_cache,
                    "sequential_cache": use_sequential_cache and args.mode != "original",
                    "history_mechanism": (
                        "native_explicit_history"
                        if use_sequential_cache and args.mode == "original"
                        else "persistent_cache" if use_sequential_cache else "none"
                    ),
                    "sequential_rollout_windows": rollout_windows,
                    "long_rollout": use_long_rollout,
                    "generated_history": use_generated_history,
                    "generated_history_supervision_depths": (
                        fixed.get("generated_history_supervision_depths", [])
                    ),
                    "generated_history_supervision_count": supervision_count,
                    "generated_history_depth_metrics": depth_metrics,
                    "generated_history_probability": scheduled_probability,
                    "differentiable_rollout": use_differentiable_rollout,
                    "differentiable_rollout_loss": mean_differentiable_rollout_loss,
                    "differentiable_rollout_windows": (
                        args.differentiable_rollout_windows
                        if use_differentiable_rollout else 0
                    ),
                    "differentiable_state_graph_depth": (
                        int(differentiable_metrics["state_graph_depth"].item())
                        if differentiable_metrics is not None else 0
                    ),
                    "differentiable_root_errors_cm": (
                        differentiable_metrics["root_errors_cm"].detach().float().mean(0).cpu().tolist()
                        if differentiable_metrics is not None else []
                    ),
                    "differentiable_constraint_errors_cm": (
                        differentiable_metrics["constraint_errors_cm"].detach().float().mean(0).cpu().tolist()
                        if differentiable_metrics is not None else []
                    ),
                    "differentiable_endpoint_grad_nonzero": (
                        sum(
                            int(endpoint.grad is not None and torch.count_nonzero(endpoint.grad) > 0)
                            for endpoint in differentiable_metrics["gradient_endpoints"]
                        )
                        if differentiable_metrics is not None else 0
                    ),
                    "history_sampling_strategy": args.history_sampling_strategy,
                    "short_window": not use_sequential_cache,
                    "generated_history_raw_recovery_m_mean": mean_raw_recovery,
                    "generated_history_applied_recovery_m_mean": mean_applied_recovery,
                    **d2_gradient_norms,
                    **d2_memory_diagnostics,
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            eval_losses = {
                protocol: _mean_across_ranks(torch.tensor(_evaluate_many(raw, client, batches), device=device))
                for protocol, batches in dev_fixed_by_protocol.items()
            }
            if rank == 0:
                record = {"step": step, **{f"eval_loss_{k}": v for k, v in eval_losses.items()}}
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        if args.save_every and step % args.save_every == 0:
            _save_checkpoint(
                output / f"checkpoint_step_{step:06d}.pt",
                args=args, raw=raw, optimizer=optimizer, step=step,
                losses=losses, initial_eval=initial_eval, all_finite=all_finite,
                sequential_cache_steps=sequential_cache_steps,
                sequential_rollout_steps=sequential_rollout_steps,
                generated_history_steps=generated_history_steps,
                generated_history_windows=generated_history_windows,
                generated_history_supervision_audit=generated_history_supervision_audit,
                teacher_history_steps=teacher_history_steps,
                short_window_steps=short_window_steps,
                rollout_window_counts=rollout_window_counts,
                differentiable_rollout_steps=differentiable_rollout_steps,
                d2_gradient_update_counts=d2_gradient_update_counts,
                data_generator=data_generator,
                elapsed_seconds=elapsed_before + time.perf_counter() - started,
                device=device, rank=rank,
            )
            _prune_checkpoints(output, args.keep_last_checkpoints, rank)

    final_eval_by_protocol = {
        protocol: _mean_across_ranks(torch.tensor(_evaluate_many(raw, client, batches), device=device))
        for protocol, batches in dev_fixed_by_protocol.items()
    }
    final_eval = float(np.mean(list(final_eval_by_protocol.values())))
    torch.cuda.synchronize(device)
    elapsed = elapsed_before + time.perf_counter() - started
    peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**20, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        supervision_summary = summarize_generated_history_supervision(
            generated_history_supervision_audit
        )
        gate_window = min(50, max(1, len(losses) // 2))
        first = float(np.mean(losses[:gate_window]))
        last = float(np.mean(losses[-gate_window:]))
        model = _unwrap_raw(raw)
        # A run initialized from a trained checkpoint is a fine-tune/resume
        # gate: its fixed validation loss may legitimately move upward while
        # the sampled training objective continues to improve.  Fresh full
        # training keeps the stricter train+validation decrease requirement.
        initialized = bool(args.init_checkpoint or args.resume)
        eval_gate = final_eval < initial_eval if not initialized else True
        root = _unwrap_block(model.root_model)
        root_output_weight_norm = float(root.output_linear.weight.detach().float().norm())
        if args.max_final_eval_loss is not None:
            eval_gate = eval_gate and final_eval <= args.max_final_eval_loss
        root_output_gate = (
            args.min_root_output_weight_norm is None
            or root_output_weight_norm >= args.min_root_output_weight_norm
        )
        metrics = {
            "status": (
                "PASS"
                if all_finite and last < first and eval_gate and root_output_gate
                else "FAIL"
            ),
            "mode": args.mode,
            "model": args.model,
            "world_size": world,
            "per_rank_batch_size": args.batch_size,
            "global_batch_size": global_batch,
            "steps": args.steps,
            "first_50_mean_loss": first,
            "last_50_mean_loss": last,
            "initial_eval_loss": initial_eval,
            "final_eval_loss": final_eval,
            "initial_eval_loss_by_protocol": initial_eval_by_protocol if not args.resume else None,
            "final_eval_loss_by_protocol": final_eval_by_protocol,
            "initialized_from_checkpoint": initialized,
            "evaluation_gate": "informational_finetune" if initialized else "strict_decrease",
            "max_final_eval_loss": args.max_final_eval_loss,
            "root_output_weight_norm": root_output_weight_norm,
            "min_root_output_weight_norm": args.min_root_output_weight_norm,
            "root_output_gate_passed": root_output_gate,
            "final_learning_rate": optimizer.param_groups[0]["lr"],
            "all_gradients_finite": all_finite,
            "sequential_cache_probability": args.sequential_cache_probability,
            "eval_protocol": args.eval_protocol,
            "production_contract": {
                "motion_mask_mode": getattr(model, "motion_mask_mode", None),
                "positional_encoding_mode": getattr(model, "positional_encoding_mode", None),
                "root_input_first_heading_angle": getattr(model.root_model, "input_first_heading_angle", None),
                "body_input_first_heading_angle": getattr(model.body_model, "input_first_heading_angle", None),
                "num_frames_per_token": int(model.num_frames_per_token),
                "gen_horizon_len": generation_horizon,
                "commit_protocol_version": (
                    "differentiable_long_rollout_v1"
                    if args.differentiable_rollout_probability > 0
                    else "d2_generated_history_bptt_v1"
                    if args.train_scope == "state_memory"
                    else
                    "generated_history_endpoint_v9"
                    if generated_sampling_enabled
                    and args.generated_history_root_endpoint_loss_weight > 0
                    else "generated_history_state_consistency_v6"
                    if generated_sampling_enabled
                    and args.generated_history_state_consistency_weight > 0
                    else "generated_history_decay_aligned_v7"
                    if generated_sampling_enabled
                    and args.mamba_output_decay_half_life_windows > 0
                    else "generated_history_bptt_v5"
                    if generated_sampling_enabled
                    and args.generated_history_bptt_windows > 0
                    else "generated_history_multidepth_v4"
                    if generated_sampling_enabled
                    and args.generated_history_max_supervision_points > 1
                    else "generated_history_deep_v3"
                    if args.history_sampling_strategy == "deep_v3"
                    else
                    "generated_history_balanced_v2"
                    if args.history_sampling_strategy == "balanced_v2"
                    else "generated_history_scheduled_v1"
                    if generated_sampling_enabled
                    else "canonical_history_v2"
                ),
            },
            "sequential_cache_steps": sequential_cache_steps,
            "sequential_rollout_steps": sequential_rollout_steps,
            "generated_history_steps": generated_history_steps,
            "generated_history_windows": generated_history_windows,
            "generated_history_supervision_audit": supervision_summary,
            "teacher_history_steps": teacher_history_steps,
            "short_window_steps": short_window_steps,
            "history_sampling_strategy": args.history_sampling_strategy,
            "short_window_probability": args.short_window_probability,
            "generated_history_probability_start": args.generated_history_probability_start,
            "generated_history_probability_1k": args.generated_history_probability_1k,
            "generated_history_probability_5k": args.generated_history_probability_5k,
            "generated_history_diffusion_steps": args.generated_history_diffusion_steps,
            "generated_history_max_recovery_meters": args.generated_history_max_recovery_meters,
            "generated_history_depth_ramp_steps": args.generated_history_depth_ramp_steps,
            "generated_history_periodic_reset_windows": args.generated_history_periodic_reset_windows,
            "generated_history_max_supervision_points": args.generated_history_max_supervision_points,
            "generated_history_bptt_windows": args.generated_history_bptt_windows,
            "generated_history_attach_seed_write": args.generated_history_attach_seed_write,
            "generated_history_state_consistency_weight_requested": (
                args.generated_history_state_consistency_weight
            ),
            "generated_history_state_consistency_weight_applied": (
                args.generated_history_state_consistency_weight
                if args.mode == "hybrid" else 0.0
            ),
            "mamba_output_decay_start_windows": args.mamba_output_decay_start_windows,
            "mamba_output_decay_half_life_windows_requested": (
                args.mamba_output_decay_half_life_windows
            ),
            "mamba_output_decay_half_life_windows_applied": (
                args.mamba_output_decay_half_life_windows
                if args.mode == "hybrid" else 0.0
            ),
            "mamba_output_decay_min_scale_requested": args.mamba_output_decay_min_scale,
            "mamba_output_decay_min_scale_applied": (
                args.mamba_output_decay_min_scale if args.mode == "hybrid" else 0.0
            ),
            "generated_history_root_endpoint_loss_weight": (
                args.generated_history_root_endpoint_loss_weight
            ),
            "differentiable_rollout_probability": args.differentiable_rollout_probability,
            "differentiable_rollout_windows": args.differentiable_rollout_windows,
            "differentiable_rollout_diffusion_steps": args.differentiable_rollout_diffusion_steps,
            "differentiable_rollout_warm_prefix_windows": args.differentiable_rollout_warm_prefix_windows,
            "long_slope_loss_weight": args.long_slope_loss_weight,
            "long_auc_loss_weight": args.long_auc_loss_weight,
            "long_final_loss_weight": args.long_final_loss_weight,
            "long_slope_margin_cm_per_window": args.long_slope_margin_cm_per_window,
            "rollout_gradient_checkpointing": args.rollout_gradient_checkpointing,
            "sequential_rollout_min_windows": args.sequential_rollout_min_windows,
            "sequential_rollout_windows": args.sequential_rollout_windows,
            "sequential_long_rollout_probability": args.sequential_long_rollout_probability,
            "sequential_rollout_curriculum": args.sequential_rollout_curriculum,
            "max_rollout_windows": max_rollout_windows,
            "rollout_window_counts": rollout_window_counts,
            "differentiable_rollout_steps": differentiable_rollout_steps,
            "train_scope": args.train_scope,
            "d2_gradient_update_counts": d2_gradient_update_counts,
            "d2_gradient_update_fraction": {
                key: value / max(args.steps, 1)
                for key, value in d2_gradient_update_counts.items()
            },
            "d2_last_memory_diagnostics": d2_last_memory_diagnostics,
            "trainable_parameters": trainable_parameter_count,
            "trainable_parameter_names": trainable_parameter_names,
            "elapsed_seconds": elapsed,
            "mean_step_seconds": elapsed / max(args.steps, 1),
            "samples_per_second": global_batch * args.steps / elapsed,
            "frames_per_second": global_batch * args.frames * args.steps / elapsed,
            "peak_memory_mib": float(peak),
            "text_backend": "llm2vec_cache" if args.text_cache else "deterministic_hash_smoke_only",
            "attention_backend": "sdpa",
            "attention_window_chunks": args.attention_window_chunks,
            "minimal_ff_size": args.minimal_ff_size,
            "training_state_intervention": args.training_state_intervention,
            "token_state_interaction": args.token_state_interaction,
            "state_summary_dim": args.state_summary_dim,
            "use_state_delta": args.use_state_delta,
            "state_aware_output_mode": args.state_aware_output_mode,
            "content_memory_enabled": args.content_memory_enabled,
            "content_memory_slots": args.content_memory_slots,
            "content_memory_read_enabled": args.content_memory_read_enabled,
            "content_memory_use_state_delta": args.content_memory_use_state_delta,
            "content_memory_intervention": args.content_memory_intervention,
            "content_memory_conditioning": args.content_memory_conditioning,
            "content_memory_normalize_delta": args.content_memory_normalize_delta,
            "content_memory_normalize_slots": args.content_memory_normalize_slots,
            "content_memory_max_write_rate": args.content_memory_max_write_rate,
            "content_memory_write_gate_context": args.content_memory_write_gate_context,
            "content_memory_decoder_init_scale": args.content_memory_decoder_init_scale,
            "content_memory_write_gate_init_scale": args.content_memory_write_gate_init_scale,
            "content_memory_write_gate_bias_init": args.content_memory_write_gate_bias_init,
            "content_memory_read_scale_init": args.content_memory_read_scale_init,
            "keep_last_checkpoints": args.keep_last_checkpoints,
        }
        body = _unwrap_block(model.body_model)
        auxiliary = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith(("root_model.", "body_model."))
        }
        checkpoint = {
            "mode": args.mode,
            "model": args.model,
            "root_model": root.state_dict(),
            "body_model": body.state_dict(),
            "denoiser_aux": auxiliary,
            "metrics": metrics,
        }
        torch.save(checkpoint, output / f"{args.mode}_humanml3d_ddp.pt")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    if (
        not args.no_final_resume_checkpoint
        and (not args.save_every or args.steps % args.save_every != 0)
    ):
        _save_checkpoint(
            output / f"checkpoint_step_{args.steps:06d}.pt",
            args=args, raw=raw, optimizer=optimizer, step=args.steps,
            losses=losses, initial_eval=initial_eval, all_finite=all_finite,
            sequential_cache_steps=sequential_cache_steps,
            sequential_rollout_steps=sequential_rollout_steps,
            generated_history_steps=generated_history_steps,
            generated_history_windows=generated_history_windows,
            generated_history_supervision_audit=generated_history_supervision_audit,
            teacher_history_steps=teacher_history_steps,
            short_window_steps=short_window_steps,
            rollout_window_counts=rollout_window_counts,
            differentiable_rollout_steps=differentiable_rollout_steps,
            d2_gradient_update_counts=d2_gradient_update_counts,
            data_generator=data_generator, elapsed_seconds=elapsed,
            device=device, rank=rank,
        )
        _prune_checkpoints(output, args.keep_last_checkpoints, rank)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
