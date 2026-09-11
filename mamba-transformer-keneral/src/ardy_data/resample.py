"""Rotation-aware motion resampling."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .schema import NativeMotion


def _continuous_quaternions(rotations: np.ndarray) -> np.ndarray:
    frames, joints = rotations.shape[:2]
    quaternions = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(frames, joints, 4)
    if frames == 1:
        return quaternions
    transitions = np.where(np.sum(quaternions[1:] * quaternions[:-1], axis=-1) < 0.0, -1.0, 1.0)
    signs = np.concatenate((np.ones((1, joints)), np.cumprod(transitions, axis=0)), axis=0)
    return quaternions * signs[..., None]


def _slerp_pairs(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    alpha = alpha[..., None]
    close = sin_angle < 1e-7
    weights0 = np.where(close, 1.0 - alpha, np.sin((1.0 - alpha) * angle) / np.maximum(sin_angle, 1e-12))
    weights1 = np.where(close, alpha, np.sin(alpha * angle) / np.maximum(sin_angle, 1e-12))
    result = weights0 * q0 + weights1 * q1
    return result / np.linalg.norm(result, axis=-1, keepdims=True).clip(min=1e-12)


def resample_rotations_and_root(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    source_fps: float,
    target_fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample rotations with SLERP and root translation linearly in time."""
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if len(local_rot_mats) != len(root_positions) or len(local_rot_mats) < 2:
        raise ValueError("rotations/root must have the same length of at least two frames")
    if abs(source_fps - target_fps) < 1e-9:
        return local_rot_mats.clone(), root_positions.clone()

    dtype = local_rot_mats.dtype
    device = local_rot_mats.device
    rotations_np = local_rot_mats.detach().cpu().double().numpy()
    root_np = root_positions.detach().cpu().double().numpy()
    duration = (len(rotations_np) - 1) / float(source_fps)
    target_count = int(np.floor(duration * target_fps + 1e-9)) + 1
    target_times = np.arange(target_count, dtype=np.float64) / float(target_fps)
    source_position = target_times * float(source_fps)
    lower = np.floor(source_position).astype(np.int64)
    lower = np.minimum(lower, len(rotations_np) - 1)
    upper = np.minimum(lower + 1, len(rotations_np) - 1)
    alpha = source_position - lower

    quaternions = _continuous_quaternions(rotations_np)
    sampled_quaternions = _slerp_pairs(quaternions[lower], quaternions[upper], alpha[:, None])
    sampled_rotations = Rotation.from_quat(sampled_quaternions.reshape(-1, 4)).as_matrix()
    sampled_rotations = sampled_rotations.reshape(target_count, rotations_np.shape[1], 3, 3)
    source_times = np.arange(len(root_np), dtype=np.float64) / float(source_fps)
    sampled_root = np.stack(
        [np.interp(target_times, source_times, root_np[:, axis]) for axis in range(3)], axis=-1
    )
    return (
        torch.as_tensor(sampled_rotations, dtype=dtype, device=device),
        torch.as_tensor(sampled_root, dtype=root_positions.dtype, device=root_positions.device),
    )


def resample_native_motion(motion: NativeMotion, target_fps: float) -> NativeMotion:
    rotations, root = resample_rotations_and_root(
        motion.local_rot_mats, motion.root_positions_m, motion.fps, target_fps
    )
    return motion.with_motion(rotations, root, target_fps)
