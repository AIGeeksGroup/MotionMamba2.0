import os

import pytest
import torch
from torch import nn

from minimal_attention_kernel import MinimalAttentionKernelConfig
from sequence_kernel import SequenceKernel, SequenceKernelConfig


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
        outputs = []
        for token in x.unbind(1):
            hidden = torch.tanh(self.mamba3.in_proj(token) + 0.5 * hidden)
            outputs.append(self.mamba3.out_proj(hidden))
        return torch.stack(outputs, dim=1), hidden


def _kernel(
    *,
    interaction: str = "state_query",
    state_aware_output_mode: str = "attention_only",
    content_memory_enabled: bool = True,
) -> SequenceKernel:
    torch.manual_seed(20260903)
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
            interaction=interaction,
            state_summary_dim=4,
            state_aware_output_mode=state_aware_output_mode,
            content_memory_enabled=content_memory_enabled,
            content_memory_slots=2,
        ),
        mamba3=mamba,
        mamba_wrapper=_FakeMambaWrapper(mamba),
        state_projector=(
            nn.Linear(8, 4)
            if interaction in {"state_query", "state_kv_modulation"}
            else None
        ),
    )


def _grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm())


def _three_window_loss(kernel: SequenceKernel, *, detach: bool) -> torch.Tensor:
    state = kernel.init_state(batch_size=2)
    generator = torch.Generator().manual_seed(41)
    windows = [torch.randn(2, 3, 8, generator=generator) for _ in range(3)]
    for chunk_id, window in enumerate(windows[:2]):
        _, pending, _ = kernel.forward_chunk(window, state=state)
        kernel.commit(state, pending, chunk_id=chunk_id, detach=detach)
    output, _, _ = kernel.forward_chunk(windows[2], state=state)
    target = torch.randn(output.shape, generator=generator)
    return (output - target).square().mean()


def run_two_step_gradient_probe(*, detach: bool) -> list[dict[str, float]]:
    kernel = _kernel()
    optimizer = torch.optim.SGD(kernel.parameters(), lr=0.5)
    records = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _three_window_loss(kernel, detach=detach).backward()
        records.append(_gradient_record(kernel))
        optimizer.step()
    return records


def _gradient_record(kernel: SequenceKernel) -> dict[str, float]:
    memory = kernel.content_memory
    return {
        "content_decoder.weight": _grad_norm(memory.content_decoder.weight),
        "content_decoder.bias": _grad_norm(memory.content_decoder.bias),
        "write_gate.weight": _grad_norm(memory.write_gate.weight),
        "write_gate.bias": _grad_norm(memory.write_gate.bias),
        "read_gate": _grad_norm(memory.read_gate),
        "state_projector.weight": _grad_norm(kernel.state_projector.weight),
        "token_state_interaction.weight": _grad_norm(
            kernel.token_state_interaction.state_to_address.weight
        ),
        "token_state_interaction.scale": _grad_norm(
            kernel.token_state_interaction.scale
        ),
        "mamba.in_proj.weight": _grad_norm(kernel.mamba_path.in_proj.weight),
        "attention.q_proj.weight": _grad_norm(kernel.attention.q_proj.weight),
    }


def test_default_init_content_memory_receives_cross_window_gradient():
    first, second = run_two_step_gradient_probe(detach=False)
    assert first["read_gate"] > 0.0
    for name in (
        "content_decoder.weight",
        "content_decoder.bias",
        "write_gate.weight",
        "write_gate.bias",
        "read_gate",
        "state_projector.weight",
        "mamba.in_proj.weight",
    ):
        assert torch.isfinite(torch.tensor(second[name]))
        assert second[name] > 0.0


def test_detached_history_blocks_write_parameter_gradient():
    attached = run_two_step_gradient_probe(detach=False)[-1]
    detached = run_two_step_gradient_probe(detach=True)[-1]
    for name in (
        "content_decoder.weight",
        "content_decoder.bias",
        "write_gate.weight",
        "write_gate.bias",
    ):
        assert attached[name] > 0.0
        assert detached[name] == 0.0
    assert detached["read_gate"] > 0.0


def test_write_gate_changes_when_previous_memory_changes():
    memory = _kernel().content_memory
    generator = torch.Generator().manual_seed(9)
    summary = torch.randn(2, 4, generator=generator)
    delta = torch.randn(2, 4, generator=generator)
    previous_a = torch.zeros(2, 2, 8)
    previous_b = torch.randn(2, 2, 8, generator=generator)
    gate_a = memory.propose(summary, delta, previous_a).write_gate
    gate_b = memory.propose(summary, delta, previous_b).write_gate
    assert not torch.equal(gate_a, gate_b)


def test_state_query_native_gated_add_zero_scale_matches_native_hybrid():
    native = _kernel(
        interaction="naive_gated_add", content_memory_enabled=False
    )
    d2 = _kernel(state_aware_output_mode="native_gated_add")
    d2.attention.load_state_dict(native.attention.state_dict(), strict=True)
    d2.mamba_path.load_state_dict(native.mamba_path.state_dict(), strict=True)
    d2.fusion_gate_logit.data.copy_(native.fusion_gate_logit.data)
    state_native = native.init_state(batch_size=2)
    state_d2 = d2.init_state(batch_size=2)
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(17))
    expected, expected_pending, _ = native.forward_chunk(x, state=state_native)
    actual, actual_pending, _ = d2.forward_chunk(x, state=state_d2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_pending.attention_kv.key,
        expected_pending.attention_kv.key,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_pending.attention_kv.value,
        expected_pending.attention_kv.value,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_pending.mamba_state, expected_pending.mamba_state, rtol=0, atol=0
    )


def test_content_memory_zero_read_scale_matches_native_hybrid_with_nonzero_slots():
    native = _kernel(
        interaction="naive_gated_add", content_memory_enabled=False
    )
    d2 = _kernel(state_aware_output_mode="native_gated_add")
    d2.attention.load_state_dict(native.attention.state_dict(), strict=True)
    d2.mamba_path.load_state_dict(native.mamba_path.state_dict(), strict=True)
    d2.fusion_gate_logit.data.copy_(native.fusion_gate_logit.data)
    state_native = native.init_state(batch_size=2)
    state_d2 = d2.init_state(batch_size=2)
    state_d2.persistent_memory.copy_(
        torch.randn(
            state_d2.persistent_memory.shape,
            generator=torch.Generator().manual_seed(23),
        )
    )
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(29))
    expected, _, _ = native.forward_chunk(x, state=state_native)
    actual, _, aux = d2.forward_chunk(x, state=state_d2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert aux["persistent_content_memory"]["memory_slot_norm"] > 0.0
    assert aux["persistent_content_memory"]["read_residual_scale"] == 0.0


@pytest.mark.skipif(
    os.environ.get("RUN_MAMBA_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set RUN_MAMBA_CUDA_TESTS=1 for the mandatory real-Mamba CUDA gate",
)
def test_real_mamba_cross_window_gradient_attached_positive_detached_negative():
    from mamba_ssm.modules.mamba3 import Mamba3

    def build() -> SequenceKernel:
        torch.manual_seed(20260903)
        mamba = Mamba3(
            d_model=32,
            d_state=16,
            expand=2,
            headdim=16,
            ngroups=1,
            is_mimo=False,
            chunk_size=16,
            layer_idx=0,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        return SequenceKernel(
            SequenceKernelConfig(
                mode="hybrid",
                attention=MinimalAttentionKernelConfig(
                    d_model=32,
                    num_heads=4,
                    window_size_chunks=4,
                    attention_backend="sdpa",
                    num_sink_tokens=1,
                ),
                interaction="state_query",
                state_summary_dim=16,
                content_memory_enabled=True,
                content_memory_slots=2,
            ),
            mamba3=mamba,
        ).cuda()

    def probe(detach: bool) -> dict[str, float]:
        kernel = build()
        optimizer = torch.optim.SGD(kernel.parameters(), lr=0.5)
        final = {}
        for step in range(2):
            generator = torch.Generator(device="cuda").manual_seed(100 + step)
            windows = [
                torch.randn(2, 16, 32, device="cuda", generator=generator)
                for _ in range(3)
            ]
            target = torch.randn(
                2, 16, 32, device="cuda", generator=generator
            )
            state = kernel.init_state(batch_size=2)
            optimizer.zero_grad(set_to_none=True)
            for chunk_id, window in enumerate(windows[:2]):
                _, pending, _ = kernel.forward_chunk(window, state=state)
                kernel.commit(state, pending, chunk_id=chunk_id, detach=detach)
            output, _, _ = kernel.forward_chunk(windows[2], state=state)
            (output - target).square().mean().backward()
            final = {
                "decoder": _grad_norm(kernel.content_memory.content_decoder.weight),
                "write_gate": _grad_norm(kernel.content_memory.write_gate.weight),
                "read_scale": _grad_norm(kernel.content_memory.read_gate),
                "projector": _grad_norm(kernel.state_projector.linear.weight),
                "mamba": _grad_norm(kernel.mamba_path.in_proj.weight),
            }
            optimizer.step()
        return final

    detached = probe(True)
    attached = probe(False)
    assert detached["read_scale"] > 0.0
    for name in ("decoder", "write_gate", "projector", "mamba"):
        assert detached[name] == 0.0
        assert torch.isfinite(torch.tensor(attached[name]))
        assert attached[name] > 0.0
