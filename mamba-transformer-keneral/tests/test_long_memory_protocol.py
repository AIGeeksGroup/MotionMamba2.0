from types import SimpleNamespace

import torch

from ardy_adapter.diffusion_memory import ArdyDiffusionMemoryController
from minimal_attention_kernel import ChunkSlidingWindowKVCache
from long_memory_protocol import (
    assert_actor_disjoint,
    canonical_sha256,
    first_threshold_window,
    holm_adjust,
    load_panel,
    paired_cluster_bootstrap,
    window_summary,
    verify_locked_final_guard,
)
import pytest


def test_recursive_state_transform_preserves_structure_and_gradient_shape():
    state = SimpleNamespace(
        ssm_state=torch.ones(3, 4, requires_grad=True),
        k_state=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        untouched="metadata",
    )
    zero = ArdyDiffusionMemoryController._transform_state(state, torch.zeros_like)
    assert zero is not state
    assert zero.untouched == state.untouched
    assert torch.count_nonzero(zero.ssm_state) == 0
    assert torch.count_nonzero(zero.k_state) == 0

    permutation = torch.tensor([2, 0, 1])
    permuted = ArdyDiffusionMemoryController._transform_state(
        state, lambda value: value.index_select(0, permutation)
    )
    torch.testing.assert_close(permuted.ssm_state, state.ssm_state.index_select(0, permutation))
    torch.testing.assert_close(permuted.k_state, state.k_state.index_select(0, permutation))


def test_state_checksum_is_finite_and_changes_after_intervention():
    state = torch.randn(2, 5)
    original = ArdyDiffusionMemoryController._state_checksum(state)
    changed = ArdyDiffusionMemoryController._state_checksum(state.roll(1, dims=0))
    assert len(original) == 64
    assert original != changed


def test_mamba_only_intervention_preserves_attention_checksum():
    cache = ChunkSlidingWindowKVCache(
        batch_size=2, num_heads=1, head_dim=3, window_size_chunks=4,
        num_sink_tokens=0, device="cpu", dtype=torch.float32,
    )
    cache.append(
        torch.randn(2, 1, 2, 3), torch.randn(2, 1, 2, 3), chunk_id=0
    )
    layer = SimpleNamespace(
        attention_cache=cache,
        mamba_state=SimpleNamespace(ssm_state=torch.randn(2, 4)),
        state_summary=torch.randn(2, 4),
        state_delta=torch.randn(2, 4),
    )
    sequence_state = SimpleNamespace(layers=[layer])
    block = SimpleNamespace(_sequence_state=sequence_state)
    controller = ArdyDiffusionMemoryController.__new__(ArdyDiffusionMemoryController)
    controller.root_block = block
    controller.body_block = SimpleNamespace(_sequence_state=None)
    controller.state_intervention = "zero_after_w4"
    controller._completed_windows = 4
    controller._intervention_applied_window = None
    controller._frozen_mamba_states = {}
    controller._last_intervention_checksum = None
    controller.intervention_audit = []
    before = controller._attention_checksum()
    controller._apply_mamba_state_intervention()
    assert controller._attention_checksum() == before
    assert torch.count_nonzero(layer.mamba_state.ssm_state) == 0
    assert controller.intervention_audit[0]["attention_checksum_before"] == before


def _content_intervention_controller(batch_size=2):
    cache = ChunkSlidingWindowKVCache(
        batch_size=batch_size, num_heads=1, head_dim=2, window_size_chunks=4,
        num_sink_tokens=0, device="cpu", dtype=torch.float32,
    )
    cache.append(
        torch.randn(batch_size, 1, 2, 2),
        torch.randn(batch_size, 1, 2, 2),
        chunk_id=0,
    )
    memory = torch.arange(batch_size * 6, dtype=torch.float32).reshape(
        batch_size, 2, 3
    )
    layer = SimpleNamespace(
        attention_cache=cache,
        mamba_state=None,
        state_summary=None,
        state_delta=None,
        persistent_memory=memory,
    )
    block = SimpleNamespace(
        _sequence_state=SimpleNamespace(layers=[layer]),
        content_memory_read_enabled_runtime=None,
        content_memory_write_enabled_runtime=True,
        content_memory_delta_mode_runtime="normal",
    )
    controller = ArdyDiffusionMemoryController.__new__(ArdyDiffusionMemoryController)
    controller.root_block = block
    controller.body_block = SimpleNamespace(
        _sequence_state=None,
        content_memory_read_enabled_runtime=None,
        content_memory_write_enabled_runtime=True,
        content_memory_delta_mode_runtime="normal",
    )
    controller._completed_windows = 4
    controller._content_memory_permutation_applied = False
    controller.content_memory_audit = []
    return controller, layer


def test_content_memory_interventions_have_all_write_semantics():
    controller, _ = _content_intervention_controller()
    for strategy, expected_write, expected_delta in (
        ("write_off", False, "normal"),
        ("zero_delta", True, "zero"),
        ("permute_delta", True, "permute"),
    ):
        controller.content_memory_intervention = strategy
        controller._set_content_memory_runtime()
        assert controller.root_block.content_memory_write_enabled_runtime is expected_write
        assert controller.root_block.content_memory_delta_mode_runtime == expected_delta


def test_content_memory_permutation_runs_once_and_preserves_attention():
    controller, layer = _content_intervention_controller()
    controller.content_memory_intervention = "permute_memory"
    before_attention = controller._attention_checksum()
    original = layer.persistent_memory.clone()
    controller._maybe_permute_content_memory()
    torch.testing.assert_close(layer.persistent_memory, original.roll(1, dims=0))
    assert controller._attention_checksum() == before_attention
    assert len(controller.content_memory_audit) == 1
    once = layer.persistent_memory.clone()
    controller._completed_windows = 5
    controller._maybe_permute_content_memory()
    torch.testing.assert_close(layer.persistent_memory, once)
    assert len(controller.content_memory_audit) == 1


def test_content_memory_permutation_requires_non_singleton_batch():
    controller, _ = _content_intervention_controller(batch_size=1)
    controller.content_memory_intervention = "permute_memory"
    with pytest.raises(ValueError, match="batch size > 1"):
        controller._maybe_permute_content_memory()


def test_persistent_read_permutation_is_read_only_and_fixed_derangement():
    controller, layer = _content_intervention_controller()
    controller.content_memory_intervention = "persistent_read_permutation"
    original = layer.persistent_memory.clone()
    attention_before = controller._attention_checksum()
    controller._set_content_memory_runtime()
    first_view = controller.root_block.content_memory_read_view_runtime[0]
    assert controller._content_memory_permutation.tolist() == [1, 0]
    torch.testing.assert_close(first_view, original.roll(1, dims=0))
    torch.testing.assert_close(layer.persistent_memory, original)
    assert controller._attention_checksum() == attention_before
    controller._completed_windows = 5
    controller._set_content_memory_runtime()
    assert controller._content_memory_permutation.tolist() == [1, 0]
    torch.testing.assert_close(layer.persistent_memory, original)
    assert len([x for x in controller.content_memory_audit if x.get("event") == "persistent_read_window"]) == 2


def test_persistent_read_permutation_rejects_singleton_batch():
    controller, _ = _content_intervention_controller(batch_size=1)
    controller.content_memory_intervention = "persistent_read_permutation"
    with pytest.raises(ValueError, match="batch size > 1"):
        controller._set_content_memory_runtime()


def test_freeze_write_controls_after_w4():
    controller, _ = _content_intervention_controller()
    controller.content_memory_intervention = "freeze_write_after_w4"
    controller._set_content_memory_runtime()
    assert controller.root_block.content_memory_write_enabled_runtime is False
    controller._completed_windows = 3
    controller._set_content_memory_runtime()
    assert controller.root_block.content_memory_write_enabled_runtime is True


def _panel_row(panel_id="p0"):
    return {
        "panel_id": panel_id,
        "id": "motion",
        "actor_uid": "actor",
        "crop_start": 0,
        "crop_end": 100,
        "caption": "walk",
        "goal_frames": [51],
        "goal_joints": [0],
        "augmentation_seed": 1,
        "noise_seed": 2,
        "source_manifest_sha256": "a" * 64,
    }


def test_panel_hash_load_roundtrip_and_duplicate_guard(tmp_path):
    rows = [_panel_row("p0"), _panel_row("p1")]
    path = tmp_path / "panel.jsonl"
    path.write_text(
        "".join(__import__("json").dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    assert load_panel(path, expected_sha256=canonical_sha256(rows)) == rows
    rows[1]["panel_id"] = "p0"
    path.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="unique"):
        load_panel(path)


def test_actor_disjoint_and_window_statistics_match_manual_values():
    assert_actor_disjoint(("train", {"a"}), ("selection", {"b"}), ("final", {"c"}))
    with pytest.raises(ValueError, match="leakage"):
        assert_actor_disjoint(("train", {"a"}), ("selection", {"a"}))
    summary = window_summary([1, 2, 3, 4, 5, 6, 8], start_index=5)
    assert summary == {
        "mean": pytest.approx(29 / 7),
        "post_w5_auc": pytest.approx(7),
        "post_w5_slope": pytest.approx(2),
        "final": pytest.approx(8),
    }
    assert first_threshold_window([1, 3, 7], 5) == 3
    assert first_threshold_window([1, 3], 5) is None


def test_actor_motion_cluster_bootstrap_and_holm_are_deterministic():
    rows = [
        {"actor_uid": "a", "id": "m1", "delta": -2.0},
        {"actor_uid": "a", "id": "m1", "delta": -4.0},
        {"actor_uid": "b", "id": "m2", "delta": -1.0},
    ]
    first = paired_cluster_bootstrap(rows, value_key="delta", resamples=200, seed=3)
    second = paired_cluster_bootstrap(rows, value_key="delta", resamples=200, seed=3)
    assert first == second
    assert first["mean_difference"] == pytest.approx(-2.0)
    assert first["win_fraction"] == 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_locked_final_guard_checks_hashes_and_once_marker(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    panel = tmp_path / "panel.jsonl"
    lock = tmp_path / "lock.json"
    marker = tmp_path / "done.json"
    checkpoint.write_bytes(b"checkpoint")
    panel.write_text("panel\n")
    lock.write_text("{}\n")
    from long_memory_protocol import file_sha256
    frozen = tmp_path / "FROZEN_CONFIG.json"
    frozen.write_text(__import__("json").dumps({
        "status": "FROZEN",
        "checkpoint_sha256": {"model": file_sha256(checkpoint)},
        "panel_file_sha256": {"panel": file_sha256(panel)},
        "protocol_lock_sha256": file_sha256(lock),
    }))
    assert verify_locked_final_guard(
        frozen, checkpoint=checkpoint, panel=panel, protocol_lock=lock,
        completion_marker=marker,
    )["status"] == "FROZEN"
    marker.write_text("done")
    with pytest.raises(FileExistsError):
        verify_locked_final_guard(
            frozen, checkpoint=checkpoint, panel=panel, protocol_lock=lock,
            completion_marker=marker,
        )


def test_resume_contract_rejects_rollout_or_window_changes():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from train_humanml3d_ardy_ddp import RESUME_CONTRACT_KEYS, _assert_resume_contract
    saved = {key: 1 for key in RESUME_CONTRACT_KEYS}
    current = dict(saved)
    _assert_resume_contract(current, saved)
    current["differentiable_rollout_windows"] = 8
    with pytest.raises(ValueError, match="resume contract mismatch"):
        _assert_resume_contract(current, saved)


def test_resume_contract_rejects_decay_changes():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from train_humanml3d_ardy_ddp import RESUME_CONTRACT_KEYS, _assert_resume_contract
    saved = {key: 1 for key in RESUME_CONTRACT_KEYS}
    current = dict(saved)
    current["mamba_output_decay_half_life_windows"] = 4.0
    with pytest.raises(ValueError, match="resume contract mismatch"):
        _assert_resume_contract(current, saved)
