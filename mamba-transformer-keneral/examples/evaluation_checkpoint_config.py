"""Recover sequence-kernel architecture from an auditable training checkpoint."""

from __future__ import annotations

import argparse
from typing import Any


def add_sequence_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional architecture overrides while defaulting to checkpoint metadata."""
    parser.add_argument(
        "--token-state-interaction",
        choices=["naive_gated_add", "state_query", "state_kv_modulation"],
        default=None,
    )
    parser.add_argument(
        "--state-aware-output-mode",
        choices=["attention_only", "native_gated_add"],
        default=None,
    )
    parser.add_argument("--state-summary-dim", type=int, default=None)
    parser.add_argument(
        "--use-state-delta", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--state-conditioning-scale-init", type=float, default=None)
    parser.add_argument(
        "--content-memory-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--content-memory-slots", type=int, default=None)
    parser.add_argument(
        "--content-memory-read-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--content-memory-use-state-delta",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--content-memory-write-gate-context",
        choices=["previous_memory", "proposal"],
        default=None,
    )
    parser.add_argument("--content-memory-decoder-init-scale", type=float, default=None)
    parser.add_argument("--content-memory-write-gate-init-scale", type=float, default=None)
    parser.add_argument("--content-memory-write-gate-bias-init", type=float, default=None)
    parser.add_argument("--content-memory-read-scale-init", type=float, default=None)
    parser.add_argument(
        "--content-memory-conditioning", choices=["none", "text"], default=None
    )
    parser.add_argument(
        "--content-memory-normalize-delta",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--content-memory-normalize-slots",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--content-memory-max-write-rate", type=float, default=None)


def _metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    # Final inference checkpoints retain the training contract in metrics;
    # resumable checkpoints retain it in config.
    result: dict[str, Any] = {}
    for key in ("config", "metrics"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            result.update(value)
    return result


def sequence_config_kwargs(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return constructor kwargs and the resolved architecture audit record."""
    metadata = _metadata(checkpoint)
    root_keys = checkpoint.get("root_model", {})
    has_content_parameters = any(".content_memory." in key for key in root_keys)

    defaults = {
        "token_state_interaction": "naive_gated_add",
        "state_aware_output_mode": "attention_only",
        "state_summary_dim": 128,
        "use_state_delta": True,
        "state_conditioning_scale_init": 0.0,
        "content_memory_enabled": has_content_parameters,
        "content_memory_slots": 1,
        "content_memory_read_enabled": True,
        "content_memory_use_state_delta": True,
        "content_memory_write_gate_context": "previous_memory",
        "content_memory_decoder_init_scale": 0.02,
        "content_memory_write_gate_init_scale": 0.02,
        "content_memory_write_gate_bias_init": 0.0,
        "content_memory_read_scale_init": 0.0,
        "content_memory_conditioning": "none",
        "content_memory_normalize_delta": False,
        "content_memory_normalize_slots": False,
        "content_memory_max_write_rate": 1.0,
    }
    resolved = {}
    sources = {}
    for key, default in defaults.items():
        override = getattr(args, key, None)
        if override is not None:
            value, source = override, "cli"
        elif key in metadata:
            value, source = metadata[key], "checkpoint"
        else:
            value, source = default, "inferred" if key == "content_memory_enabled" else "default"
        resolved[key] = value
        sources[key] = source

    if bool(resolved["content_memory_enabled"]) != has_content_parameters:
        raise ValueError(
            "resolved content-memory architecture disagrees with checkpoint parameters"
        )
    audit = {"resolved": resolved, "source": sources}
    return resolved, audit


__all__ = ["add_sequence_checkpoint_arguments", "sequence_config_kwargs"]
