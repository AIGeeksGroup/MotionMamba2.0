# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Root smoothing.'
import math
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

class TrajectorySmoother:
    """
    min f(x) + g(z)
    s.t. I x - z = 0

    x --> argmin_x (f(x) + p/2 ||I x - z + u||^2)
    z --> argmin_z (g(z) + p/2 ||I x - z + u||^2)
    u --> u + I x - z

    f(x) = pos_weight * |x - x_orig|^2 + |A x|^2
    g(z) = inf if any(|z-t| > margin) else 0

    x minimization:

    E(x) = wp/2 * |x - x_orig|^2 + 1/2 |A x|^2 + p/2 |I x - z + u|^2
    E(x) = wp/2 * (x - x_orig)^T (x - x_orig) + x^T A^T A x + p/2 (I x - z + u)^T (I x - z + u)
    E(x) = wp/2 * (x^T x - x_orig^T x - x^T x_orig + x_orig^T x_orig) +
           1/2 x^T A^T A x +
           p/2 (x^T I^T I x - x^T I^T z + x^T I^T u - z^T I x + z^T z - z^T u + u^T I x - u^T z + u^T u)

    argmin E(x) = argmin [
            wp/2 * (x^T x - 2 x^T x_orig) +
            1/2 x^T A^T A x +
            p/2 (x^T I^T I x + 2 x^T I^T (u - z) )
    ]
    = argmin [
            x^T wp/2 * I * x - wp * x^T x_orig +
            1/2 x^T A^T A x +
            x^T p/2 I^T I x + p x^T I^T (u - z)
    ]
    = argmin [
            1/2 x^T (wp * I + A^T A + p I^T I) x - x^T (wp * x_orig + p I^T (z - u))
    ]

    x = (wp * I + A^T A + p I)^-1 (wp * x_orig + p (z - u))


    """

    def __init__(self, margins, pos_weight=0.0, loop=False, admm_iters=100, alpha_overrelax=1.0, circle_project=False):
        self.pos_weight = pos_weight
        self.admm_iters = admm_iters
        self.alpha_overrelax = alpha_overrelax
        self.circle_project = circle_project
        N = len(margins)
        self.margin_vals = margins
        a_data = []
        a_rows = []
        a_cols = []
        for i in range(1, N - 1):
            scale = 1.0
            a_data.extend([-scale, 2.0 * scale, -scale])
            a_rows.extend([i, i, i])
            a_cols.extend([i - 1, i, i + 1])
        if loop:
            scale = 1.0
            a_data.extend([-scale, 2.0 * scale, -scale])
            a_rows.extend([0, 0, 0])
            a_cols.extend([N - 1, 0, 1])
            scale = 1.0
            a_data.extend([-scale, 2.0 * scale, -scale])
            a_rows.extend([N - 1, N - 1, N - 1])
            a_cols.extend([N - 2, N - 1, 0])
        A = sparse.csr_matrix((a_data, (a_rows, a_cols)), shape=(N, N))
        identity_matrix = sparse.eye(N)
        M = pos_weight * identity_matrix + A.T @ A
        diag_max = max(abs(M.diagonal()))
        self.admm_stepsize = 0.25 * np.sqrt(diag_max)
        M = M + self.admm_stepsize * identity_matrix
        self.system_lu = splu(M.tocsc())

    def smooth(self, targets, x0):
        x_target = targets.copy()
        x = x0.copy()
        z = np.zeros_like(x)
        u = np.zeros_like(x)
        for _ in range(self.admm_iters):
            self.z_update(z, x, x_target, u)
            self.u_update(u, x, z)
            self.x_update(x, z, u, x_target)
        return x

    def x_update(self, x, z, u, x_t):
        r = self.pos_weight * x_t + self.admm_stepsize * (z - u)
        x[:] = self.system_lu.solve(r)

    def z_update(self, z, x, z_t, u):
        z[:] = x + u - z_t
        z_diff_norms = np.linalg.norm(z, axis=1)
        mask = z_diff_norms > self.margin_vals
        if np.any(mask):
            scale_factors = self.margin_vals[mask] / z_diff_norms[mask]
            z[mask] *= scale_factors[:, np.newaxis]
        z[:] += z_t
        if self.circle_project:
            z[:] = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-06)

    def u_update(self, u, x, z):
        u[:] += self.alpha_overrelax * (x - z)

def smooth_signal(x, margins, pos_weight=0, alpha_overrelax=1.8, admm_iters=500, circle_project=False):
    x_smoothed = x.copy()
    x_smoothed[:] = x.mean(axis=0, keepdims=True)
    levels = int(math.floor(math.log2(len(x))))
    levels = max(levels - 4, 1)
    stepsize = 2 ** levels
    while True:
        num_steps = len(x_smoothed[::stepsize])
        smoother = TrajectorySmoother(margins=margins[::stepsize], pos_weight=pos_weight, alpha_overrelax=alpha_overrelax, admm_iters=admm_iters, circle_project=circle_project)
        x_smoothed[::stepsize] = smoother.smooth(x[::stepsize], x_smoothed[::stepsize])
        next_stepsize = stepsize // 2
        num_interleaved = len(x_smoothed[next_stepsize::stepsize])
        if num_interleaved == num_steps:
            x_smoothed[next_stepsize::stepsize][-1] = x_smoothed[::stepsize][-1] + (x_smoothed[::stepsize][-1] - x_smoothed[::stepsize][-2]) / 2
            num_interleaved = num_interleaved - 1
        x_smoothed[next_stepsize::stepsize][:num_interleaved] = (x_smoothed[::stepsize][:-1] + x_smoothed[::stepsize][1:]) / 2
        if stepsize == 1:
            break
        stepsize //= 2
    return x_smoothed

def get_smooth_root_pos(hip_translations):
    root_translations_xz = hip_translations[..., [0, 2]]
    root_translations_y = hip_translations[..., [1]]
    (batch_size, nframes) = root_translations_xz.shape[:2]
    margins = np.full(root_translations_xz.shape[1], 0.06)
    root_translations_smoothed_xz = []
    for batch in range(batch_size):
        root_translations_smoothed_xz.append(smooth_signal(root_translations_xz[batch].detach().cpu().numpy(), margins)[None])
    root_translations_smoothed_xz = torch.tensor(np.concatenate(root_translations_smoothed_xz))
    root_translations = torch.cat([root_translations_smoothed_xz.to(root_translations_y.device), root_translations_y], dim=-1)[..., [0, 2, 1]]
    return root_translations

def smooth_trajectory_gpu(positions: torch.Tensor, margin: float=0.06, smoothness_weight: float=1.0, position_weight: float=0.01, num_iters: int=50) -> torch.Tensor:
    device = positions.device
    n_frames = positions.shape[0]
    if n_frames < 3:
        return positions.clone()
    smoothed = positions.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([smoothed], lr=0.01)
    for _ in range(num_iters):
        optimizer.zero_grad()
        vel = smoothed[1:] - smoothed[:-1]
        acc = vel[1:] - vel[:-1]
        smoothness_loss = smoothness_weight * torch.sum(acc ** 2)
        position_loss = position_weight * torch.sum((smoothed - positions) ** 2)
        loss = smoothness_loss + position_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            diff = smoothed - positions
            diff_norm = torch.norm(diff, dim=1, keepdim=True)
            mask = diff_norm > margin
            if mask.any():
                scale = margin / (diff_norm[mask] + 1e-08)
                smoothed[mask] = positions[mask] + diff[mask] * scale
    return smoothed.detach()

def smooth_trajectory_gpu_fast(positions: torch.Tensor, margin: float=0.06, smoothness_weight: float=100.0, position_weight: float=0.1) -> torch.Tensor:
    device = positions.device
    dtype = positions.dtype
    n_frames = positions.shape[0]
    n_dims = positions.shape[1]
    if n_frames < 3:
        return positions.clone()
    A = torch.zeros((n_frames - 2, n_frames), device=device, dtype=dtype)
    for i in range(n_frames - 2):
        A[i, i] = 1.0
        A[i, i + 1] = -2.0
        A[i, i + 2] = 1.0
    ATA = A.T @ A
    M = smoothness_weight * ATA + position_weight * torch.eye(n_frames, device=device, dtype=dtype)
    smoothed = torch.zeros_like(positions)
    for d in range(n_dims):
        b = position_weight * positions[:, d]
        smoothed[:, d] = torch.linalg.solve(M, b)
    diff = smoothed - positions
    diff_norm = torch.norm(diff, dim=1)
    mask = diff_norm > margin
    if mask.any():
        scale = margin / (diff_norm[mask] + 1e-08)
        smoothed[mask] = positions[mask] + diff[mask] * scale[:, None]
    return smoothed

def get_smooth_root_pos_gpu(hip_translations: torch.Tensor) -> torch.Tensor:
    device = hip_translations.device
    root_translations_xz = hip_translations[..., [0, 2]]
    root_translations_y = hip_translations[..., [1]]
    batch_size = root_translations_xz.shape[0]
    root_translations_smoothed_xz = []
    for batch_idx in range(batch_size):
        smoothed_xz = smooth_trajectory_gpu(root_translations_xz[batch_idx], margin=0.06, smoothness_weight=1.0, position_weight=0.01, num_iters=50)
        root_translations_smoothed_xz.append(smoothed_xz[None])
    root_translations_smoothed_xz = torch.cat(root_translations_smoothed_xz, dim=0)
    root_translations = torch.cat([root_translations_smoothed_xz, root_translations_y], dim=-1)
    root_translations = root_translations[..., [0, 2, 1]]
    return root_translations

def get_smooth_root_pos_gpu_fast(hip_translations: torch.Tensor) -> torch.Tensor:
    device = hip_translations.device
    root_translations_xz = hip_translations[..., [0, 2]]
    root_translations_y = hip_translations[..., [1]]
    batch_size = root_translations_xz.shape[0]
    root_translations_smoothed_xz = []
    for batch_idx in range(batch_size):
        smoothed_xz = smooth_trajectory_gpu_fast(root_translations_xz[batch_idx], margin=0.06, smoothness_weight=100.0, position_weight=0.1)
        root_translations_smoothed_xz.append(smoothed_xz[None])
    root_translations_smoothed_xz = torch.cat(root_translations_smoothed_xz, dim=0)
    root_translations = torch.cat([root_translations_smoothed_xz, root_translations_y], dim=-1)
    root_translations = root_translations[..., [0, 2, 1]]
    return root_translations
