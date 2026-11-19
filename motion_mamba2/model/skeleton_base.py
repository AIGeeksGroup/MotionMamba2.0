# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Skeleton base.'
from pathlib import Path
from typing import Optional
import torch
from motion_mamba2.model.assets import skeleton_asset_path
from motion_mamba2.model.kinematics import fk
from motion_mamba2.model.skeleton_transforms import from_standard_tpose
from motion_mamba2.model.skeleton_transforms import global_rots_to_local_rots
from motion_mamba2.model.skeleton_transforms import to_standard_tpose

class SkeletonBase(torch.nn.Module):
    name = None
    bone_order_names_with_parents = None
    bone_order_names_no_root = None
    root_idx = None
    foot_joint_names = None
    foot_joint_idx = None
    hip_joint_names = None
    hip_joint_idx = None

    def __init__(self, folder: Optional[str]=None, name: Optional[str]=None, load: bool=True, **kwargs):
        super().__init__()
        if name is not None:
            assert self.name in name
            self.name = name
        if folder is None:
            folder = str(skeleton_asset_path(self.name))
        self.folder = folder
        self.dim = len(self.bone_order_names_with_parents)
        if load and folder is not None:
            pfolder = Path(folder)
            neutral_joints = torch.load(pfolder / 'joints.p').squeeze()
            self.register_buffer('neutral_joints', neutral_joints, persistent=False)
            if (pfolder / 'bvh_joints.p').exists():
                bvh_neutral_joints = torch.load(pfolder / 'bvh_joints.p').squeeze()
                self.register_buffer('bvh_neutral_joints', bvh_neutral_joints, persistent=False)
            global_offset_path = pfolder / 'standard_t_pose_global_offsets_rots.p'
            if global_offset_path.exists():
                global_rot_offsets = torch.load(global_offset_path).squeeze()
                self.register_buffer('global_rot_offsets', global_rot_offsets, persistent=False)
        self.bone_order_names = [x for (x, y) in self.bone_order_names_with_parents]
        self.bone_parents = dict(self.bone_order_names_with_parents)
        self.bone_index = {x: idx for (idx, x) in enumerate(self.bone_order_names)}
        self.bone_order_names_index = self.bone_index
        joint_parents = torch.tensor([-1 if (y := self.bone_parents[x]) is None else self.bone_index[y] for x in self.bone_order_names])
        self.register_buffer('joint_parents', joint_parents, persistent=False)
        self.nbjoints = len(self.bone_order_names)
        assert self.nbjoints == len(self.joint_parents)
        if 'neutral_joints' in self.__dict__:
            assert self.nbjoints == len(self.neutral_joints)
        root_indices = torch.where(joint_parents == -1)[0]
        assert len(root_indices) == 1
        self.root_idx = root_indices[0].item()
        if 'neutral_joints' in self.__dict__:
            assert (self.neutral_joints[0] == 0).all()
        self.bone_order_names_no_root = self.bone_order_names[:self.root_idx] + self.bone_order_names[self.root_idx + 1:]
        self.foot_joint_names = self.left_foot_joint_names + self.right_foot_joint_names
        self.foot_joint_names_index = {x: idx for (idx, x) in enumerate(self.foot_joint_names)}
        self.left_foot_joint_idx = [self.bone_order_names.index(foot_joint) for foot_joint in self.left_foot_joint_names]
        self.right_foot_joint_idx = [self.bone_order_names.index(foot_joint) for foot_joint in self.right_foot_joint_names]
        self.foot_joint_idx = self.left_foot_joint_idx + self.right_foot_joint_idx
        self.hip_joint_idx = [self.bone_order_names.index(hip_joint) for hip_joint in self.hip_joint_names]

    def expand_joint_names(self, joint_names):
        base_ee = ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Hips']
        pelvis_name = self.bone_order_names[self.root_idx]
        base_pos_names = [self.left_foot_joint_names, self.right_foot_joint_names, self.left_hand_joint_names, self.right_hand_joint_names, [pelvis_name]]
        base_rot_names = [self.left_foot_joint_names[:1], self.right_foot_joint_names[:1], self.left_hand_joint_names[:1], self.right_hand_joint_names[:1], [pelvis_name]]
        rot_joint_names = []
        pos_joint_names = []
        for jname in joint_names:
            idx = base_ee.index(jname)
            rot_joint_names += base_rot_names[idx]
            pos_joint_names += base_pos_names[idx]
        return (rot_joint_names, pos_joint_names)

    def expand_joint_names_batched(self, joint_names):
        base_ee = ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Hips']
        pelvis_name = self.bone_order_names[self.root_idx]
        base_pos_names = [self.left_foot_joint_names, self.right_foot_joint_names, self.left_hand_joint_names, self.right_hand_joint_names, [pelvis_name]]
        base_rot_names = [self.left_foot_joint_names[:1], self.right_foot_joint_names[:1], self.left_hand_joint_names[:1], self.right_hand_joint_names[:1], [pelvis_name]]
        rot_joint_names = []
        pos_joint_names = []
        for key_joint_names in joint_names:
            key_rot_names = []
            key_pos_names = []
            for jname in key_joint_names:
                idx = base_ee.index(jname)
                key_rot_names += base_rot_names[idx]
                key_pos_names += base_pos_names[idx]
            rot_joint_names.append(key_rot_names)
            pos_joint_names.append(key_pos_names)
        return (rot_joint_names, pos_joint_names)

    def __repr__(self):
        if self.folder is None:
            return f'{self.__class__.__name__}()'
        return f'{self.__class__.__name__}(folder="{self.folder}")'

    @property
    def device(self):
        if getattr(self, 'neutral_joints', None) is None:
            return 'cpu'
        return self.neutral_joints.device

    def fk(self, local_joint_rots: torch.Tensor, root_positions: torch.Tensor):
        (global_joint_rots, posed_joints, posed_joints_norootpos) = fk(local_joint_rots, root_positions, self)
        return (global_joint_rots, posed_joints, posed_joints_norootpos)

    def to_standard_tpose(self, local_rot_mats: torch.Tensor):
        return to_standard_tpose(local_rot_mats, self)

    def from_standard_tpose(self, local_rot_mats: torch.Tensor):
        return from_standard_tpose(local_rot_mats, self)

    def global_rots_to_local_rots(self, global_joint_rots: torch.Tensor):
        return global_rots_to_local_rots(global_joint_rots, self)

    def get_skel_slice(self, skeleton: 'SkeletonBase'):
        try:
            skel_slice = [skeleton.bone_index[x] for x in self.bone_order_names]
        except KeyError:
            raise ValueError('The current skeleton contain joints that are not in the input')
        return skel_slice
