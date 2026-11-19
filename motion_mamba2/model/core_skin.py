# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Core skin.'
from pathlib import Path
import numpy as np
import torch
from motion_mamba2.model.kinematics import batch_rigid_transform
SKIN_NAME = 'skin_standard.npz'

class CoreSkin:

    def __init__(self, skeleton):
        self.skeleton = skeleton
        skin_data_path = Path(skeleton.folder) / SKIN_NAME
        assert skeleton.neutral_joints is not None, 'CoreSkeleton27 must have neutral joints instantiated'
        device = skeleton.neutral_joints.device
        skin_data = np.load(skin_data_path)
        bind_rig_np = np.array(skin_data['bind_rig_transform'], dtype=np.float32)
        self.bind_rig_transform = torch.from_numpy(bind_rig_np).to(device=device, dtype=torch.float)
        bind_rig_inv_np = np.linalg.inv(bind_rig_np)
        self.bind_rig_transform_inv = torch.from_numpy(bind_rig_inv_np).to(device=device, dtype=torch.float)
        self.bind_vertices = torch.tensor(skin_data['bind_vertices'], device=device, dtype=torch.float)
        self.faces = torch.tensor(skin_data['faces'], device=device, dtype=torch.long)
        self.lbs_indices = torch.tensor(skin_data['lbs_indices'], device=device, dtype=torch.long)
        self.lbs_weights = torch.tensor(skin_data['lbs_weights'], device=device, dtype=torch.float)
        rig_joint_names = list(skin_data['rig_joint_names'])
        for (sname, rname) in zip(self.skeleton.bone_order_names, rig_joint_names):
            if sname != rname:
                raise ValueError(f"MISMATCH in skinnging rig: expected='{sname}' vs rig='{rname}'")

    def lbs(self, posed_transform):
        bind_rig_transform_inv = self.bind_rig_transform_inv
        bind_vertices = self.bind_vertices
        lbs_weights = self.lbs_weights
        for _ in range(posed_transform.dim() - 3):
            bind_rig_transform_inv = bind_rig_transform_inv.unsqueeze(0)
            bind_vertices = bind_vertices.unsqueeze(0)
            lbs_weights = lbs_weights.unsqueeze(0)
        affine_mat = (posed_transform @ bind_rig_transform_inv)[..., :3, :]
        vs = affine_mat[..., self.lbs_indices, :, :] @ torch.concat([bind_vertices, torch.ones_like(bind_vertices[..., 0:1])], dim=-1)[..., None, :, None]
        ws = lbs_weights[..., None, None]
        resv = (vs * ws).sum(dim=-3).squeeze(-1)
        return resv

    def skin(self, joint_rotmat, joint_pos, rot_is_global=False):
        (nF, nJ) = joint_pos.shape[:2]
        device = joint_rotmat.device
        fk_transform = torch.eye(4, device=device)[None, None].repeat(nF, nJ, 1, 1)
        fk_transform[..., :3, 3] = joint_pos
        if rot_is_global:
            fk_transform[..., :3, :3] = joint_rotmat
        else:
            neutral_joints_seq = self.skeleton.neutral_joints[None].repeat((nF, 1, 1)).to(device)
            (_, global_joint_rotmat) = batch_rigid_transform(joint_rotmat, neutral_joints_seq, self.skeleton.joint_parents.to(device), self.skeleton.root_idx)
            fk_transform[..., :3, :3] = global_joint_rotmat
        vertices = self.lbs(fk_transform)
        return vertices
