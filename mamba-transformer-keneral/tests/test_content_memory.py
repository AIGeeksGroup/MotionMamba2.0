import torch
import pytest
from torch import nn

from minimal_attention_kernel import MinimalAttentionKernelConfig
from sequence_kernel import SequenceKernel, SequenceKernelConfig
from state_memory_operator.content_memory import PersistentContentMemory


def test_content_memory_fixed_capacity_reset_carry_and_permutation():
    module = PersistentContentMemory(state_dim=4, d_model=6, slots=2)
    previous = module.initial(3, device=torch.device("cpu"), dtype=torch.float32)
    summary = torch.randn(3, 4)
    delta = torch.randn(3, 4)
    pending = module.propose(summary, delta, previous)
    assert pending.slots.shape == (3, 2, 6)
    # Neutral initialization keeps the proposal/readout numerically safe.
    assert torch.isfinite(pending.slots).all()
    carried = pending.slots.clone()
    reset = module.reset(carried, torch.tensor([True, False, False]))
    assert torch.equal(reset[0], torch.zeros_like(reset[0]))
    assert torch.equal(reset[1:], carried[1:])
    permuted = module.permute_rows(carried, torch.tensor([2, 0, 1]))
    assert torch.equal(permuted[0], carried[2])


def test_content_memory_readout_off_is_exact_zero():
    module = PersistentContentMemory(state_dim=2, d_model=3, slots=1)
    slots = torch.ones(2, 1, 3)
    assert torch.equal(module.read(slots, enabled=False), torch.zeros_like(slots))


def test_text_conditioned_delta_changes_proposal_and_bounds_writes():
    torch.manual_seed(11)
    module = PersistentContentMemory(
        state_dim=4,
        d_model=8,
        slots=2,
        condition_dim=8,
        normalize_delta=True,
        normalize_slots=True,
        max_write_rate=0.25,
    )
    summary = torch.randn(2, 4)
    delta = torch.randn(2, 4) * 1e-3
    previous = torch.zeros(2, 2, 8)
    condition_a = torch.randn(2, 8)
    condition_b = -condition_a
    pending_a = module.propose(summary, delta, previous, condition_a)
    pending_b = module.propose(summary, delta, previous, condition_b)
    assert not torch.equal(pending_a.proposal, pending_b.proposal)
    assert float(pending_a.write_gate.max()) <= 0.25
    proposal_norm = float(pending_a.proposal.float().norm(dim=-1).mean())
    assert 0.0 < proposal_norm <= 8**0.5 + 1e-5


def test_text_film_changes_interpretation_of_same_delta():
    module = PersistentContentMemory(
        state_dim=2, d_model=4, slots=1, condition_dim=4,
        normalize_delta=True,
    )
    with torch.no_grad():
        module.condition_projector.weight.zero_()
        module.condition_projector.weight[0, 0] = 1.0
        module.condition_projector.weight[1, 1] = 1.0
        module.delta_conditioner.weight.copy_(
            torch.tensor([[0.3, -0.2], [-0.1, 0.4], [0.2, 0.1], [-0.3, 0.2]])
        )
        module.content_decoder.weight.fill_(0.1)
    summary = torch.tensor([[1.0, -1.0]])
    delta = torch.tensor([[0.1, -0.1]])
    previous = torch.zeros(1, 1, 4)
    left = module.propose(
        summary, delta, previous, torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    turn = module.propose(
        summary, delta, previous, torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    )
    assert not torch.equal(left.proposal, turn.proposal)


class _FakeMamba(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)


class _FakeMambaWrapper:
    def __init__(self, mamba: _FakeMamba) -> None:
        self.mamba3 = mamba

    def init_state(self, batch_size, device=None, dtype=None):
        weight = self.mamba3.in_proj.weight
        return torch.zeros(
            batch_size,
            weight.shape[0],
            device=weight.device if device is None else device,
            dtype=weight.dtype if dtype is None else dtype,
        )

    def forward_chunk(self, x, state=None):
        hidden = torch.zeros_like(x[:, 0]) if state is None else state
        output = []
        for token in x.unbind(1):
            hidden = torch.tanh(self.mamba3.in_proj(token) + 0.5 * hidden)
            output.append(self.mamba3.out_proj(hidden))
        return torch.stack(output, dim=1), hidden


def _kernel(*, use_delta: bool = True) -> SequenceKernel:
    torch.manual_seed(7)
    mamba = _FakeMamba(8)
    return SequenceKernel(
        SequenceKernelConfig(
            mode="hybrid",
            attention=MinimalAttentionKernelConfig(
                d_model=8,
                num_heads=2,
                window_size_chunks=2,
                attention_backend="sdpa",
                num_sink_tokens=1,
            ),
            interaction="state_query",
            state_summary_dim=4,
            content_memory_enabled=True,
            content_memory_slots=2,
            content_memory_use_state_delta=use_delta,
        ),
        mamba3=mamba,
        mamba_wrapper=_FakeMambaWrapper(mamba),
        state_projector=nn.Linear(8, 4),
    )


def _open_memory_gates(kernel: SequenceKernel) -> None:
    with torch.no_grad():
        kernel.content_memory.content_decoder.bias.fill_(0.25)
        kernel.content_memory.content_decoder.weight[:, 4:].fill_(0.1)
        kernel.content_memory.write_gate.bias.fill_(12.0)
        kernel.content_memory.read_gate.fill_(12.0)


def test_memory_only_ablation_projects_state_without_query_interaction():
    mamba = _FakeMamba(8)
    kernel = SequenceKernel(
        SequenceKernelConfig(
            mode="hybrid",
            attention=MinimalAttentionKernelConfig(
                d_model=8,
                num_heads=2,
                window_size_chunks=2,
                attention_backend="sdpa",
                num_sink_tokens=1,
            ),
            interaction="naive_gated_add",
            state_summary_dim=4,
            content_memory_enabled=True,
            content_memory_slots=2,
        ),
        mamba3=mamba,
        mamba_wrapper=_FakeMambaWrapper(mamba),
        state_projector=nn.Linear(8, 4),
    )
    assert not kernel.uses_token_state_interaction
    assert kernel.uses_state_summary
    state = kernel.init_state(batch_size=2)
    assert state.state_summary is not None
    _, pending, _ = kernel.forward_chunk(torch.randn(2, 3, 8), state=state)
    assert pending.state_summary is not None
    assert pending.persistent_memory is not None


def test_noisy_forward_is_read_only_and_accepted_pending_commits_once():
    kernel = _kernel()
    _open_memory_gates(kernel)
    state = kernel.init_state(batch_size=2)
    before = state.persistent_memory.clone()
    _, pending, _ = kernel.forward_chunk(torch.randn(2, 3, 8), state=state)
    assert torch.equal(state.persistent_memory, before)
    assert state.commit_version == 0
    assert not torch.equal(pending.persistent_memory, before)

    kernel.commit(state, pending, chunk_id=0)
    assert state.commit_version == 1
    assert not torch.equal(state.persistent_memory, before)
    with pytest.raises(RuntimeError, match="exactly once"):
        kernel.commit(state, pending, chunk_id=1)


def test_content_memory_write_and_read_interventions_are_explicit():
    kernel = _kernel()
    _open_memory_gates(kernel)
    state = kernel.init_state(batch_size=2)
    x = torch.randn(2, 3, 8)

    _, write_off, write_aux = kernel.forward_chunk(
        x, state=state, content_memory_write_enabled=False
    )
    assert torch.equal(write_off.persistent_memory, state.persistent_memory)
    assert write_aux["persistent_content_memory"]["write_enabled"] is False

    _, normal, _ = kernel.forward_chunk(x, state=state)
    zero_delta = torch.zeros_like(state.state_delta)
    _, zeroed, _ = kernel.forward_chunk(
        x, state=state, content_memory_delta=zero_delta
    )
    assert not torch.equal(normal.persistent_memory, zeroed.persistent_memory)

    kernel.commit(state, normal, chunk_id=0)
    read_on, _, on_aux = kernel.forward_chunk(x, state=state)
    read_off, _, off_aux = kernel.forward_chunk(
        x, state=state, content_memory_read_enabled=False
    )
    assert not torch.equal(read_on, read_off)
    assert on_aux["persistent_content_memory"]["read_enabled"] is True
    assert off_aux["persistent_content_memory"]["read_enabled"] is False


def test_persistent_read_view_changes_output_not_durable_write_proposal():
    kernel = _kernel()
    _open_memory_gates(kernel)
    state = kernel.init_state(batch_size=2)
    x = torch.randn(2, 3, 8)
    _, seed, _ = kernel.forward_chunk(x, state=state)
    kernel.commit(state, seed, chunk_id=0)
    durable = state.persistent_memory.clone()
    normal_output, normal_pending, _ = kernel.forward_chunk(x, state=state)
    swapped_output, swapped_pending, _ = kernel.forward_chunk(
        x, state=state, content_memory_read_view=durable.roll(1, dims=0)
    )
    assert not torch.equal(normal_output, swapped_output)
    torch.testing.assert_close(normal_pending.persistent_memory, swapped_pending.persistent_memory)
    torch.testing.assert_close(state.persistent_memory, durable)


def test_d_no_delta_zeroes_only_content_write_delta():
    full = _kernel(use_delta=True)
    no_delta = _kernel(use_delta=False)
    _open_memory_gates(full)
    _open_memory_gates(no_delta)
    full_state = full.init_state(batch_size=2)
    no_delta_state = no_delta.init_state(batch_size=2)
    x = torch.randn(2, 3, 8)
    _, full_pending, full_aux = full.forward_chunk(x, state=full_state)
    _, no_delta_pending, no_delta_aux = no_delta.forward_chunk(x, state=no_delta_state)
    assert not torch.equal(full_pending.persistent_memory, no_delta_pending.persistent_memory)
    assert full_aux["persistent_content_memory"]["uses_state_delta"] is True
    assert no_delta_aux["persistent_content_memory"]["uses_state_delta"] is False


def test_neutral_content_memory_is_exact_baseline_parity():
    baseline = _kernel()
    # Construct the same model without the opt-in content path from the same
    # seed. Content modules are created after all baseline modules.
    torch.manual_seed(7)
    mamba = _FakeMamba(8)
    control = SequenceKernel(
        SequenceKernelConfig(
            mode="hybrid",
            attention=MinimalAttentionKernelConfig(
                d_model=8, num_heads=2, window_size_chunks=2,
                attention_backend="sdpa", num_sink_tokens=1,
            ),
            interaction="state_query", state_summary_dim=4,
        ),
        mamba3=mamba,
        mamba_wrapper=_FakeMambaWrapper(mamba),
        state_projector=nn.Linear(8, 4),
    )
    x = torch.randn(2, 3, 8)
    candidate, _, candidate_aux = baseline.forward_chunk(
        x, state=baseline.init_state(batch_size=2)
    )
    expected, _, _ = control.forward_chunk(x, state=control.init_state(batch_size=2))
    torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
    assert candidate_aux["persistent_content_memory"]["read_enabled"] is True
    assert candidate_aux["persistent_content_memory"]["read_residual_scale"] == 0.0
