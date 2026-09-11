"""Phase-0 validation for the `state_memory_operator` package.

Run with:

/home/plote/miniconda3/envs/env_isaaclab/bin/python -m pytest \
    mamba_transformer_keneral/tests/test_smoke.py -v

Covers, per `docs/mamba_integration_contract.md`'s "Minimal Integration
Checklist": full-vs-chunk consistency, state reset, determinism, basic
forward/backward, plus this package's own additions (per-row async reset,
memory compaction, FIFO vs. Participative eviction behavior).
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

from conftest import requires_cuda


def test_event_memory_import_does_not_require_mamba():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(os.path.join(os.path.dirname(__file__), "..", "src")), env.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from state_memory_operator import EventMemory; "
            "import sys; assert 'mamba_ssm' not in sys.modules; assert EventMemory.__name__ == 'EventMemory'",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _make_mamba3(seed: int = 0, d_model: int = 64):
    from mamba_ssm.modules.mamba3 import Mamba3

    torch.manual_seed(seed)
    return Mamba3(
        d_model=d_model,
        d_state=32,
        expand=2,
        headdim=32,
        ngroups=1,
        is_mimo=False,
        chunk_size=16,
        layer_idx=0,
        device="cuda",
        dtype=torch.bfloat16,
    )


@requires_cuda
def test_chunk_consistency_matches_full_sequence():
    from state_memory_operator import Mamba3ChunkWrapper

    model = _make_mamba3().eval()
    wrapper = Mamba3ChunkWrapper(model)

    torch.manual_seed(1)
    batch, seqlen, chunk_len = 2, 128, 32
    x = torch.randn(batch, seqlen, model.d_model, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        y_full = model(x)
        state = None
        outs = []
        for start in range(0, seqlen, chunk_len):
            y_chunk, state = wrapper.forward_chunk(x[:, start : start + chunk_len], state)
            outs.append(y_chunk)
        y_chunked = torch.cat(outs, dim=1)

    max_abs_diff = (y_full.float() - y_chunked.float()).abs().max().item()
    assert torch.isfinite(y_chunked).all()
    assert max_abs_diff < 0.5, f"chunk vs. full-sequence diverged: max_abs_diff={max_abs_diff}"


@requires_cuda
def test_reset_mask_is_per_row_and_isolates_rows():
    from state_memory_operator import Mamba3ChunkWrapper
    from state_memory_operator.mamba3_chunk_wrapper import apply_reset_mask

    model = _make_mamba3().eval()
    wrapper = Mamba3ChunkWrapper(model)
    torch.manual_seed(2)
    batch, chunk_len = 3, 16

    with torch.no_grad():
        x1 = torch.randn(batch, chunk_len, model.d_model, device="cuda", dtype=torch.bfloat16)
        _, state = wrapper.forward_chunk(x1, None)

        reset_mask = torch.tensor([True, False, True], device="cuda")
        reset_state = apply_reset_mask(state, reset_mask)

        # Reset rows must be exactly zero; untouched row must be unchanged.
        assert torch.count_nonzero(reset_state.ssm_state[0]) == 0
        assert torch.count_nonzero(reset_state.ssm_state[2]) == 0
        assert torch.equal(reset_state.ssm_state[1], state.ssm_state[1])

        # Running the next chunk with a reset row must match a fresh state in
        # the same batch shape. The bf16 Triton Mamba path is not required to
        # be bitwise invariant when B changes from 3 to 1, so a singleton
        # reference would conflate reset semantics with batch-shape numerics.
        x2 = torch.randn(batch, chunk_len, model.d_model, device="cuda", dtype=torch.bfloat16)
        out_with_reset, _ = wrapper.forward_chunk(x2, reset_state)
        out_fresh, _ = wrapper.forward_chunk(x2, None)
        for row in (0, 2):
            max_diff = (
                out_with_reset[row].float() - out_fresh[row].float()
            ).abs().max().item()
            assert max_diff < 1e-3, (
                f"reset row {row} diverged from a fresh-state run: {max_diff}"
            )


@requires_cuda
def test_determinism():
    from state_memory_operator import Mamba3ChunkWrapper

    torch.manual_seed(5)
    x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16)

    def run():
        model = _make_mamba3(seed=0).eval()
        wrapper = Mamba3ChunkWrapper(model)
        with torch.no_grad():
            y, _ = wrapper.forward_chunk(x, None)
        return y

    y1, y2 = run(), run()
    assert torch.equal(y1, y2)


@requires_cuda
def test_forward_backward_smoke():
    from state_memory_operator import Mamba3ChunkWrapper

    model = _make_mamba3()
    wrapper = Mamba3ChunkWrapper(model)
    x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y, state = wrapper.forward_chunk(x, None)
    loss = y.float().sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_event_memory_fifo_evicts_oldest():
    from state_memory_operator import EventMemory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mem = EventMemory(batch_size=1, capacity=3, d_key=4, d_value=2, device=device, backend="fifo")
    for i in range(5):
        k = torch.zeros(1, 4, device=device)
        k[0, 0] = i  # distinct-ish key per write, doesn't need to be normalized for this test
        v = torch.full((1, 2), float(i), device=device)
        mem.write(k, v)
    # After 5 writes into capacity=3, only writes 2,3,4 should remain valid.
    assert int(mem._valid.sum().item()) == 3
    remaining_values = {mem._values[i][0, 0].item() for i in range(mem.num_entries) if mem._valid[0, i]}
    assert remaining_values == {2.0, 3.0, 4.0}


def test_event_memory_participative_protects_reused_entry():
    from state_memory_operator import EventMemory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mem = EventMemory(batch_size=1, capacity=2, d_key=4, d_value=2, device=device, backend="participative")

    k0 = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device), dim=-1)
    v0 = torch.tensor([[1.0, 0.0]], device=device)
    mem.write(k0, v0)

    # Retrieve entry 0 repeatedly so it accrues participation credit.
    for _ in range(5):
        mem.retrieve(k0, topk=1, temperature=0.1)

    # Now fill with two more, DISTINCT, never-reused entries -- FIFO would
    # evict entry 0 first (oldest); Participative should protect it instead.
    k1 = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device), dim=-1)
    mem.write(k1, torch.tensor([[2.0, 0.0]], device=device))
    k2 = torch.nn.functional.normalize(torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device), dim=-1)
    mem.write(k2, torch.tensor([[3.0, 0.0]], device=device))

    r, topk_idx, _, _ = mem.retrieve(k0, topk=1, temperature=0.1)
    assert r[0, 0].item() == 1.0, "participative eviction dropped the reused entry it should have protected"


def test_event_memory_reset_rows_is_per_row():
    from state_memory_operator import EventMemory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mem = EventMemory(batch_size=2, capacity=4, d_key=4, d_value=2, device=device, backend="fifo")
    k = torch.zeros(2, 4, device=device)
    v = torch.zeros(2, 2, device=device)
    mem.write(k, v)

    mem.reset_rows(torch.tensor([True, False], device=device))
    assert not bool(mem._valid[0].any())
    assert bool(mem._valid[1].any())


def test_event_memory_compaction_preserves_content():
    from state_memory_operator import EventMemory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mem = EventMemory(batch_size=1, capacity=4, d_key=4, d_value=2, device=device, backend="fifo", compact_every=8)
    for i in range(20):
        k = torch.nn.functional.normalize(torch.randn(1, 4, device=device), dim=-1)
        v = torch.full((1, 2), float(i), device=device)
        mem.write(k, v)
    assert mem.num_entries < 20, "compaction never ran despite exceeding compact_every"
    # Only the last 4 writes (16,17,18,19) should be valid/retrievable.
    remaining = {mem._values[i][0, 0].item() for i in range(mem.num_entries) if mem._valid[0, i]}
    assert remaining == {16.0, 17.0, 18.0, 19.0}


@requires_cuda
def test_full_operator_end_to_end_with_memory():
    from state_memory_operator import StateMemoryOperatorConfig, StateTransitionMemoryOperator

    model = _make_mamba3()
    config = StateMemoryOperatorConfig(d_model=64, d_summary=32, d_qkv=16, d_value=16, memory_capacity=8, topk=2)
    op = StateTransitionMemoryOperator(model, config).cuda()

    batch, chunk_len = 2, 8
    device = torch.device("cuda")
    op_state = op.init_operator_state(batch, device=device)
    memory = op.init_memory(batch, device=device)

    for step in range(6):
        x = torch.randn(batch, chunk_len, 64, device=device, dtype=torch.bfloat16)
        content = x[:, 0, :].float()
        out, op_state, memory, aux = op.forward_chunk_with_memory(
            x, op_state, memory, read_content=content, write_content=content
        )
        assert out.shape == x.shape
        assert torch.isfinite(out.float()).all()
        assert aux["state_after_summary"].shape == (batch, config.d_summary)

    # A reset on row 0 only must not disturb row 1's memory content.
    row1_entries_before = int(memory._valid[1].sum().item())
    memory.reset_rows(torch.tensor([True, False], device=device))
    assert int(memory._valid[0].sum().item()) == 0
    assert int(memory._valid[1].sum().item()) == row1_entries_before

    # Backward pass sanity: loss through the operator should reach the
    # value encoder and memory_proj.
    x = torch.randn(batch, chunk_len, 64, device=device, dtype=torch.bfloat16)
    content = x[:, 0, :].float()
    out, op_state, memory, aux = op.forward_chunk_with_memory(
        x, op_state, memory, read_content=content, write_content=content
    )
    out.float().sum().backward()
    assert op.memory_proj.weight.grad is not None
    assert op.value_encoder.proj.weight.grad is not None
