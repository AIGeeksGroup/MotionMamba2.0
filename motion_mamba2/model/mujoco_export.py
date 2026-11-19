# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Mujoco export.'
import os
import xml.etree.ElementTree as ET
from typing import Optional
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from motion_mamba2.model.assets import skeleton_asset_path
from motion_mamba2.model.geometry import matrix_to_quaternion
from motion_mamba2.model.skeleton_base import SkeletonBase
from motion_mamba2.model.tensor_utils import ensure_batched
from motion_mamba2.model.tensor_utils import to_numpy
from motion_mamba2.model.tensor_utils import to_torch
_DEFAULT_G1_XML = str(skeleton_asset_path('g1skel34', 'xml', 'g1.xml'))

class MujocoQposConverter(torch.nn.Module):

    def __init__(self, input_skeleton: SkeletonBase, xml_path: str=_DEFAULT_G1_XML, dead_joint_rotation_scheme: str='dummy'):
        super().__init__()
        self.xml_path = xml_path
        self.skeleton = input_skeleton
        self._prepare_transforms()
        self._subtree_joints = {}
        self._dead_joint_rotation_scheme = dead_joint_rotation_scheme

    def _prepare_transforms(self):
        self.mujoco_to_ardy_matrix = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        self.ardy_to_mujoco_matrix = self.mujoco_to_ardy_matrix.T
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        xml_classes = [x for x in tree.findall('.//default') if 'class' in x.attrib]
        joint_axes = dict()
        for xml_class in xml_classes:
            j = xml_class.findall('joint')
            if j:
                joint_axes[xml_class.get('class')] = j[0].get('axis')
        mujoco_hinge_joints = root.find('worldbody').findall('.//joint')
        self._mujoco_joint_axis_values_ardy_space = torch.zeros((len(mujoco_hinge_joints), 3), dtype=torch.float32)
        self._mujoco_joint_axis_values_mujoco_space = torch.zeros((len(mujoco_hinge_joints), 3), dtype=torch.float32)
        self._mujoco_indices_to_ardy_indices = torch.zeros((len(mujoco_hinge_joints),), dtype=torch.int32)
        self._ardy_indices_to_mujoco_indices = torch.ones((self.skeleton.nbjoints,), dtype=torch.int32) * -1
        self._nb_joints_mujoco = len(mujoco_hinge_joints) + 1
        self._nb_joints_ardy = self.skeleton.nbjoints
        self._mujoco_joint_including_root_parent_list = torch.full((len(mujoco_hinge_joints) + 1,), -1, dtype=torch.int32)
        self._mujoco_joint_including_root_list = ['pelvis_skel']
        for (joint_id_in_csv, joint) in enumerate(mujoco_hinge_joints):
            joint_name_in_skeleton = joint.get('name').replace('_joint', '_skel')
            joint_parent_name_in_skeleton = self.skeleton.bone_parents[joint_name_in_skeleton]
            self._mujoco_joint_including_root_list.append(joint_name_in_skeleton)
            self._mujoco_joint_including_root_parent_list[joint_id_in_csv + 1] = self._mujoco_joint_including_root_list.index(joint_parent_name_in_skeleton)
            joint_idx_in_ardy_skeleton = self.skeleton.bone_order_names.index(joint_name_in_skeleton)
            axis_values = [float(x) for x in (joint.get('axis') or joint_axes[joint.get('class')]).split(' ')]
            mujoco_joint_axis_mapping_ardy_space = [torch.tensor([0, 0, 1]), torch.tensor([1, 0, 0]), torch.tensor([0, 1, 0])][np.argmax(axis_values)]
            self._mujoco_joint_axis_values_ardy_space[joint_id_in_csv] = mujoco_joint_axis_mapping_ardy_space
            self._mujoco_joint_axis_values_mujoco_space[joint_id_in_csv] = torch.tensor(axis_values)
            self._mujoco_indices_to_ardy_indices[joint_id_in_csv] = joint_idx_in_ardy_skeleton
            self._ardy_indices_to_mujoco_indices[joint_idx_in_ardy_skeleton] = joint_id_in_csv + 1
        self._ardy_indices_to_mujoco_indices[0] = 0
        R_zup_to_yup = Rotation.from_euler('x', -90, degrees=True)
        x_forward_to_y_forward = Rotation.from_euler('z', -90, degrees=True)
        mujoco_to_ardy = R_zup_to_yup * x_forward_to_y_forward
        self._rot_offsets_q2t = torch.zeros(len(self._ardy_indices_to_mujoco_indices), 3, 3, dtype=torch.float32)
        self._rot_offsets_q2t[...] = torch.eye(3)[None]
        self._rot_offsets_f2q = torch.zeros(len(self._ardy_indices_to_mujoco_indices), 3, 3, dtype=torch.float32)
        self._rot_offsets_f2q[...] = torch.eye(3)[None]
        parent_map = {child: parent for parent in root.iter() for child in parent}
        for (i, joint) in enumerate(mujoco_hinge_joints):
            body = parent_map[joint]
            if 'quat' in body.attrib:
                rot = Rotation.from_quat([float(x) for x in body.get('quat').strip().split(' ')], scalar_first=True)
                idx = self._mujoco_indices_to_ardy_indices[i]
                self._rot_offsets_q2t[idx] = torch.from_numpy(rot.as_matrix())
                rot = mujoco_to_ardy * rot * mujoco_to_ardy.inv()
                self._rot_offsets_f2q[idx] = torch.from_numpy(rot.as_matrix().T)

    def dict_to_qpos(self, output: dict, device: Optional[str]=None, root_quat_w_first: bool=True, numpy: bool=True):
        local_rot_mats = to_torch(output['local_rot_mats'], device)
        root_positions = to_torch(output['root_positions'], device)
        qpos = self.to_qpos(local_rot_mats, root_positions, root_quat_w_first=root_quat_w_first)
        if numpy:
            qpos = to_numpy(qpos)
        return qpos

    def save_csv(self, qpos: torch.Tensor | np.ndarray, csv_path):
        qpos = to_numpy(qpos)
        shape = qpos.shape
        if len(shape) == 2:
            np.savetxt(csv_path, qpos, delimiter=',')
        if len(shape) == 3:
            if shape[0] == 1:
                np.savetxt(csv_path, qpos[0], delimiter=',')
            else:
                (csv_path_base, ext) = os.path.splitext(csv_path)
                for i in range(shape[0]):
                    self.save_csv(qpos[i], csv_path_base + '_' + str(i).zfill(2) + ext)

    @ensure_batched(local_rot_mats=5, root_positions=3, lengths=1)
    def to_qpos(self, local_rot_mats: torch.Tensor, root_positions: torch.Tensor, root_quat_w_first: bool=True) -> torch.Tensor:
        (batch_size, num_frames, nb_joints) = local_rot_mats.shape[:3]
        (device, dtype) = (local_rot_mats.device, local_rot_mats.dtype)
        local_rot_mats = torch.matmul(self._rot_offsets_f2q.to(device), local_rot_mats)
        (batch_size, num_frames) = (root_positions.shape[0], root_positions.shape[1])
        ardy_to_mujoco_matrix = self.ardy_to_mujoco_matrix.to(device=device, dtype=dtype)
        qpos = torch.zeros((batch_size, num_frames, 36), dtype=dtype, device=device)
        root_positions_mujoco = torch.matmul(ardy_to_mujoco_matrix[None, None, ...], root_positions[..., None])
        qpos[:, :, :3] = root_positions_mujoco.view(batch_size, num_frames, 3)
        root_rot = local_rot_mats[:, :, 0, :]
        mujoco_to_ardy_matrix = ardy_to_mujoco_matrix.T
        root_rot_mujoco = torch.matmul(torch.matmul(ardy_to_mujoco_matrix[None, None, ...], root_rot), mujoco_to_ardy_matrix[None, None, ...])
        root_rot_quat = matrix_to_quaternion(root_rot_mujoco)
        if root_quat_w_first:
            qpos[:, :, 3:7] = root_rot_quat[:, :, [0, 1, 2, 3]]
        else:
            qpos[:, :, 3:7] = root_rot_quat[:, :, [1, 2, 3, 0]]
        joint_rot_mujoco = local_rot_mats[:, :, self._mujoco_indices_to_ardy_indices, :]
        x_joint_dof = torch.atan2(joint_rot_mujoco[..., 2, 1], joint_rot_mujoco[..., 2, 2])
        y_joint_dof = torch.atan2(joint_rot_mujoco[..., 0, 2], joint_rot_mujoco[..., 0, 0])
        z_joint_dof = torch.atan2(joint_rot_mujoco[..., 1, 0], joint_rot_mujoco[..., 1, 1])
        xyz_joint_dofs = torch.stack([x_joint_dof, y_joint_dof, z_joint_dof], dim=-1)
        joint_dofs = (xyz_joint_dofs * self._mujoco_joint_axis_values_ardy_space[None, None, :, :].to(device)).sum(dim=-1)
        qpos[:, :, 7:] = joint_dofs
        return qpos
