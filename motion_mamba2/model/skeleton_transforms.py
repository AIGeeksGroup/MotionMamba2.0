# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Skeleton transforms.'
import einops
import torch
from motion_mamba2.model.tensor_utils import ensure_batched
from motion_mamba2.model.kinematics import batch_rigid_transform

def global_rots_to_local_rots(global_joint_rots: torch.Tensor, skeleton):
    (global_joint_mats, ps) = einops.pack([global_joint_rots], '* nbjoints dim1 dim2')
    device = global_joint_mats.device
    parent_rot_mats = global_joint_mats[:, skeleton.joint_parents.to(device)]
    parent_rot_mats[:, skeleton.root_idx] = torch.eye(3, device=device, dtype=parent_rot_mats.dtype)
    parent_rot_mats_inv = parent_rot_mats.transpose(2, 3)
    local_rot_mats = torch.einsum('T N m n, T N n o -> T N m o', parent_rot_mats_inv, global_joint_mats)
    [local_rot_mats] = einops.unpack(local_rot_mats, ps, '* nbjoints dim1 dim2')
    return local_rot_mats

@ensure_batched(local_rot_mats=4)
def change_tpose(local_rot_mats: torch.Tensor, global_rot_offsets: torch.Tensor, skeleton):
    (device, dtype) = (local_rot_mats.device, local_rot_mats.dtype)
    global_rot_offsets = global_rot_offsets.to(device=device, dtype=dtype)
    root_idx = skeleton.root_idx
    joint_parents = skeleton.joint_parents
    neutral_joints = torch.ones((len(local_rot_mats), skeleton.nbjoints, 3), device=device, dtype=dtype)
    (_, global_rot_mats) = batch_rigid_transform(local_rot_mats, neutral_joints, joint_parents, root_idx)
    new_global_rot_mats = torch.einsum('T N m n, N o n -> T N m o', global_rot_mats, global_rot_offsets)
    new_local_rot_mats = global_rots_to_local_rots(new_global_rot_mats, skeleton)
    return (new_local_rot_mats, new_global_rot_mats)

@ensure_batched(local_rot_mats=4)
def to_standard_tpose(local_rot_mats: torch.Tensor, skeleton):
    global_rot_offsets = skeleton.global_rot_offsets
    return change_tpose(local_rot_mats, global_rot_offsets, skeleton)

@ensure_batched(local_rot_mats=4)
def from_standard_tpose(local_rot_mats: torch.Tensor, skeleton):
    global_rot_offsets = skeleton.global_rot_offsets
    global_rot_offsets_T = global_rot_offsets.mT
    return change_tpose(local_rot_mats, global_rot_offsets_T, skeleton)
