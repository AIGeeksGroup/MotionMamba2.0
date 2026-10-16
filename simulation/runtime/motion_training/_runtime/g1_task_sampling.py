"""Eligible-pool-aware task sampling, retaining a uniform general component."""
import hashlib
import json
from pathlib import Path
import re
import torch


def load_sampling_spec(path, records):
    spec = json.loads(Path(path).read_text())
    fraction = float(spec['focus_fraction'])
    if not 0 < fraction < 1 or not spec['families']:
        raise ValueError('Focus sampling must retain a nonzero general component')
    patterns = {name: re.compile(pattern) for name, pattern in spec['families'].items()}
    tags = []
    for record in records:
        matches = [name for name, pattern in patterns.items() if pattern.search(record['id'])]
        if len(matches) > 1:
            raise ValueError(f'Overlapping task families for {record["id"]}: {matches}')
        tags.append(matches[0] if matches else 'general')
    return spec, tags


def weights_for_pool(tags, indices, focus_fraction):
    """Half uniform records + half uniform available families, by default.

Families without sufficiently long clips receive no invented/padded examples.
The full uniform component ensures every eligible record remains sampleable.
"""
    selected = [tags[int(i)] for i in indices]
    if not selected or not 0 < focus_fraction < 1:
        raise ValueError('Invalid pool or focus fraction')
    families = sorted(set(selected)-{'general'})
    if not families:
        return torch.full((len(selected),), 1/len(selected), dtype=torch.float64)
    weights = torch.full((len(selected),), (1-focus_fraction)/len(selected), dtype=torch.float64)
    for family in families:
        mask = torch.tensor([tag == family for tag in selected], dtype=torch.bool)
        weights[mask] += focus_fraction / len(families) / int(mask.sum())
    assert torch.all(weights > 0) and torch.isclose(weights.sum(), torch.tensor(1., dtype=weights.dtype))
    return weights


def weighted_indices(weights, count, scheduling_rng_state):
    """Independent deterministic stream; do not alter task/augmentation draws.

Hash the saved scheduling state rather than reuse its random numbers for both
record identity and augmentation seeds. Native uniform draws still advance
the original generator exactly as before in the caller.
"""
    weights = torch.as_tensor(weights, dtype=torch.float64, device='cpu')
    if weights.ndim != 1 or not len(weights) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('Sampling weights must be finite, positive and one-dimensional')
    seed = int.from_bytes(hashlib.sha256(scheduling_rng_state.cpu().numpy().tobytes()).digest()[:8], 'little')
    generator = torch.Generator().manual_seed(seed)
    return torch.multinomial(weights, count, replacement=True, generator=generator)
