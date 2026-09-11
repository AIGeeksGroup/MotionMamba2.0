from argparse import Namespace
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from evaluation_checkpoint_config import sequence_config_kwargs


_FIELDS = {
    "token_state_interaction": None,
    "state_aware_output_mode": None,
    "state_summary_dim": None,
    "use_state_delta": None,
    "state_conditioning_scale_init": None,
    "content_memory_enabled": None,
    "content_memory_slots": None,
    "content_memory_read_enabled": None,
    "content_memory_use_state_delta": None,
    "content_memory_write_gate_context": None,
    "content_memory_decoder_init_scale": None,
    "content_memory_write_gate_init_scale": None,
    "content_memory_write_gate_bias_init": None,
    "content_memory_read_scale_init": None,
}


def test_d2_architecture_is_recovered_from_final_checkpoint_metrics():
    checkpoint = {
        "root_model": {"layers.0.content_memory.read_gate": torch.zeros(())},
        "metrics": {
            "token_state_interaction": "state_query",
            "state_aware_output_mode": "native_gated_add",
            "state_summary_dim": 128,
            "content_memory_enabled": True,
            "content_memory_slots": 2,
            "content_memory_use_state_delta": False,
        },
    }
    resolved, audit = sequence_config_kwargs(Namespace(**_FIELDS), checkpoint)
    assert resolved["token_state_interaction"] == "state_query"
    assert resolved["state_aware_output_mode"] == "native_gated_add"
    assert resolved["content_memory_slots"] == 2
    assert resolved["content_memory_use_state_delta"] is False
    assert audit["source"]["content_memory_slots"] == "checkpoint"


def test_checkpoint_content_parameters_cannot_be_silently_disabled():
    fields = dict(_FIELDS)
    fields["content_memory_enabled"] = False
    checkpoint = {
        "root_model": {"layers.0.content_memory.read_gate": torch.zeros(())},
        "metrics": {"token_state_interaction": "state_query"},
    }
    with pytest.raises(ValueError, match="disagrees"):
        sequence_config_kwargs(Namespace(**fields), checkpoint)


def test_memory_only_checkpoint_is_recovered_without_query_interaction():
    checkpoint = {
        "root_model": {"layers.0.content_memory.read_gate": torch.zeros(())},
        "metrics": {
            "token_state_interaction": "naive_gated_add",
            "state_aware_output_mode": "attention_only",
            "content_memory_enabled": True,
            "content_memory_slots": 2,
        },
    }
    resolved, audit = sequence_config_kwargs(Namespace(**_FIELDS), checkpoint)
    assert resolved["token_state_interaction"] == "naive_gated_add"
    assert resolved["state_aware_output_mode"] == "attention_only"
    assert resolved["content_memory_enabled"] is True
    assert audit["source"]["token_state_interaction"] == "checkpoint"
