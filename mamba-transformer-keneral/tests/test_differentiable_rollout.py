import pytest
import torch

from ardy_adapter.differentiable_rollout import (
    deterministic_recurrent_carry,
    differentiable_rollout_loss,
    least_squares_slope,
    post_window_auc,
)


def test_slope_auc_and_final_are_differentiable():
    errors = torch.tensor(
        [[1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0],
         [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
        requires_grad=True,
    )
    slopes = least_squares_slope(errors, start_window=5)
    auc = post_window_auc(errors, start_window=5)
    torch.testing.assert_close(slopes, torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(auc, torch.tensor([10.0, 2.0]))
    loss, parts = differentiable_rollout_loss(
        errors, slope_weight=1.0, auc_weight=0.1, final_weight=0.2
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert errors.grad is not None and torch.count_nonzero(errors.grad) > 0
    assert set(parts) >= {"slope_cm_per_window", "auc_cm_windows", "final_error_cm"}


def test_short_rollout_is_rejected():
    with pytest.raises(ValueError):
        least_squares_slope(torch.ones(2, 5), start_window=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA preflight failed")
def test_cuda_windows_6_to_8_backpropagate_through_early_recurrent_carry():
    device = torch.device("cuda")
    inputs = torch.randn(2, 8, 6, device=device, requires_grad=True)
    write = torch.randn(6, 6, device=device, requires_grad=True)
    read = torch.randn(6, 2, device=device, requires_grad=True)
    endpoints, _ = deterministic_recurrent_carry(inputs, write, read)
    errors = endpoints.square().mean(dim=-1)
    loss, _ = differentiable_rollout_loss(
        errors, slope_weight=0.1, auc_weight=0.01, final_weight=0.1
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.count_nonzero(inputs.grad[:, :5]) > 0
    assert write.grad is not None and torch.count_nonzero(write.grad) > 0
    assert read.grad is not None and torch.count_nonzero(read.grad) > 0

    detached_inputs = inputs.detach().clone().requires_grad_(True)
    detached_write = write.detach().clone().requires_grad_(True)
    detached_read = read.detach().clone().requires_grad_(True)
    detached_endpoints, _ = deterministic_recurrent_carry(
        detached_inputs,
        detached_write,
        detached_read,
        detach_after_window=5,
    )
    detached_errors = detached_endpoints.square().mean(dim=-1)
    detached_loss, _ = differentiable_rollout_loss(
        detached_errors, slope_weight=0.1, auc_weight=0.01, final_weight=0.1
    )
    detached_loss.backward()
    assert torch.count_nonzero(detached_inputs.grad[:, :5]) == 0
