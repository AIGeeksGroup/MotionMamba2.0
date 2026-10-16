"""Training-only rigid yaw coverage for text-only GT continuation.

Cold tasks are bitwise untouched. A private hash draw preserves the trainer's
sample, task, diffusion-noise and dropout RNG streams in matched experiments.
No generation output, codec parameter, physical loss or target phase changes.
"""
import hashlib
import math

import torch


POLICY = 'teacher_yaw_half_uniform_v1'


def yaw_for_seed(seed):
    digest = hashlib.sha256(f'{POLICY}:{int(seed)}'.encode()).digest()
    if digest[0] < 128:
        return 0.0
    u = int.from_bytes(digest[1:9], 'little') / 2**64
    return (2*u-1)*math.pi


@torch.no_grad()
def augment_teacher_heading(rep, batch, *, policy, teacher, device, allowed_ids=None):
    """Rotate entire continuous GT clip before encoding, or exact no-op.

Half the teacher examples remain unchanged; the others receive uniform yaw.
World-space controls are deliberately unsupported, not silently rotated.
"""
    if policy not in ('none', POLICY):
        raise ValueError('Unknown teacher heading augmentation')
    size = len(batch['motion'])
    if policy == 'none' or not teacher:
        return batch, [0.0]*size
    if not allowed_ids or not isinstance(allowed_ids, (set, frozenset)):
        raise ValueError('Teacher yaw requires an explicit caption-reviewed source allowlist')
    if (batch['motion_mask'].any() or batch['observed_motion'].any()
            or not (batch['history_frames'] > 0).all()):
        raise ValueError('Yaw augmentation requires text-only GT history')
    seeds = batch['augmentation_seed'].tolist()
    if len(seeds) != size:
        raise ValueError('Missing per-example augmentation seeds')
    angles = [yaw_for_seed(s) if source in allowed_ids else 0.
              for s, source in zip(seeds, batch['id'])]
    if len(angles) != size:
        raise ValueError('Source IDs do not match the batch')
    if not any(angles):
        return batch, angles
    original = batch['motion']
    motion = original.to(device).clone()
    for i, angle in enumerate(angles):
        if angle == 0.:
            continue
        physical = rep.unnormalize(motion[i:i+1])
        rotated = rep.rotate(physical, torch.tensor([angle], device=device, dtype=physical.dtype))
        motion[i:i+1] = rep.normalize(rotated)
    if not torch.isfinite(motion).all():
        raise ValueError('Nonfinite rotated GT motion')
    result = dict(batch, motion=motion.to(original.device))
    return result, [math.degrees(a) for a in angles]
