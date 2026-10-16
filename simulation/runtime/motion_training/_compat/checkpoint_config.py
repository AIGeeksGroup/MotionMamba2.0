import argparse
from typing import Any

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
        "content_memory_require_write": False,
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

