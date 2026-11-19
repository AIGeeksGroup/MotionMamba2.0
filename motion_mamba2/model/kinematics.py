# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Kinematics.'
from typing import List
import einops
import torch
import torch.nn.functional as F
from motion_mamba2.model.tensor_utils import ensure_batched

@ensure_batched(local_joint_rots=4, root_positions=2)
def fk(local_joint_rots: torch.Tensor, root_positions: torch.Tensor, skeleton, root_positions_is_global: bool=True):
    device = local_joint_rots.device
    dtype = local_joint_rots.dtype
    neutral_joints = skeleton.neutral_joints.to(device=device, dtype=dtype)
    if root_positions_is_global is True:
        pelvis_offset = neutral_joints[skeleton.root_idx]
        neutral_joints = neutral_joints - pelvis_offset
    joints = einops.repeat(neutral_joints, 'j k -> b j k', b=len(local_joint_rots))
    (posed_joints_norootpos, global_joint_rots) = batch_rigid_transform(local_joint_rots, joints, skeleton.joint_parents, skeleton.root_idx)
    posed_joints = posed_joints_norootpos + root_positions[:, None]
    return (global_joint_rots, posed_joints, posed_joints_norootpos)

def compute_idx_levels(parents):
    idx_levs = [[]]
    lev_dicts = {0: -1}
    for i in range(1, parents.shape[0]):
        assert int(parents[i]) in lev_dicts
        lev = lev_dicts[int(parents[i])] + 1
        if lev + 1 > len(idx_levs):
            idx_levs.append([])
        idx_levs[lev].append(int(i))
        lev_dicts[int(i)] = lev
    idx_levs = [torch.tensor(x).long() for x in idx_levs]
    return idx_levs

def batch_rigid_transform(rot_mats, joints, parents, root_idx):
    idx_levs = compute_idx_levels(parents)
    return forward_kinematics(rot_mats, joints, parents, idx_levs, root_idx)

@torch.jit.script
def transform_mat(R, t):
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1.0)], dim=2)

@torch.jit.script
def forward_kinematics(rot_mats, joints, parents: torch.Tensor, idx_levs: List[torch.Tensor], root_idx: int):
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    mask_no_root = torch.ones(joints.shape[1], dtype=torch.bool)
    mask_no_root[root_idx] = False
    rel_joints[:, mask_no_root] -= joints[:, parents[mask_no_root]].clone()
    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    transforms = torch.zeros_like(transforms_mat)
    transforms[:, root_idx] = transforms_mat[:, root_idx]
    for indices in idx_levs:
        curr_res = torch.matmul(transforms[:, parents[indices]], transforms_mat[:, indices])
        transforms[:, indices] = curr_res
    posed_joints = transforms[:, :, :3, 3]
    global_rot_mat = transforms[:, :, :3, :3]
    return (posed_joints, global_rot_mat)
