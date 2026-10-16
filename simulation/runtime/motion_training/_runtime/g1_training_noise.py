"""Explicit training timestep distributions; inference schedule is unchanged."""
import torch

UNIFORM = 'uniform'
LOW_NOISE = 'low_noise_mixture_v1'
POLICIES = (UNIFORM, LOW_NOISE)


def low_noise_quantiles(u, num_steps):
    """50% uniform first3 timesteps, 50% uniform all timesteps (no holes)."""
    if num_steps < 1 or not torch.isfinite(u).all() or not ((u >= 0) & (u < 1)).all():
        raise ValueError('Expected positive step count and finite uniform draws in [0,1)')
    low_count = min(3, num_steps)
    return torch.where(u < .5, (u*2*low_count).long(), ((u-.5)*2*num_steps).long()).clamp(max=num_steps-1)


def sample_training_timesteps(batch_size, num_steps, device, policy=UNIFORM):
    if policy == UNIFORM:
        # Preserve the historical draw and RNG consumption exactly.
        return torch.randint(0, num_steps, (batch_size,), device=device)
    if policy == LOW_NOISE:
        return low_noise_quantiles(torch.rand(batch_size, device=device), num_steps)
    raise ValueError(f'Unknown training timestep policy: {policy}')
