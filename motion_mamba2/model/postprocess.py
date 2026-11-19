# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Postprocess.'
from types import SimpleNamespace
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import numpy as np
import torch
from motion_mamba2.model.constraints import EndEffectorConstraintSet
from motion_mamba2.model.constraints import FullBodyConstraintSet
from motion_mamba2.model.constraints import Root2DConstraintSet
from motion_mamba2.model.geometry import matrix_to_quaternion
from motion_mamba2.model.geometry import quaternion_to_matrix
from motion_mamba2.model.skeleton import G1Skeleton34
from motion_mamba2.model.skeleton_base import SkeletonBase
from motion_mamba2.model.kinematics import fk

def extract_input_motion_from_constraints(constraint_lst: List, skeleton: SkeletonBase, num_frames: int, num_joints: int) -> Tuple[torch.Tensor, torch.Tensor]:
    hip_translations_input = torch.zeros(num_frames, 3)
    rotations_input = torch.zeros(num_frames, num_joints, 4)
    rotations_input[..., 0] = 1.0

    def _match_hip_dtype(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=hip_translations_input.device, dtype=hip_translations_input.dtype)

    def _match_rot_dtype(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=rotations_input.device, dtype=rotations_input.dtype)
    if not constraint_lst:
        return (hip_translations_input, rotations_input)
    sorted_constraints = sorted(constraint_lst, key=lambda c: isinstance(c, FullBodyConstraintSet))
    for constraint in sorted_constraints:
        frame_indices = constraint.frame_indices
        if isinstance(frame_indices, torch.Tensor):
            valid_mask = frame_indices < num_frames
            if valid_mask.sum() == 0:
                continue
            frame_indices = frame_indices[valid_mask]
        else:
            valid_positions = [i for (i, idx) in enumerate(frame_indices) if idx < num_frames]
            if not valid_positions:
                continue
            frame_indices = [frame_indices[i] for i in valid_positions]
        if isinstance(constraint, Root2DConstraintSet):
            root_2d = getattr(constraint, 'root_2d', None)
            if isinstance(frame_indices, torch.Tensor):
                root_2d = root_2d[valid_mask]
            else:
                root_2d = root_2d[valid_positions]
            root_2d = _match_hip_dtype(root_2d)
            hip_translations_input[frame_indices, 0] = root_2d[:, 0]
            hip_translations_input[frame_indices, 2] = root_2d[:, 1]
            continue
        elif isinstance(constraint, FullBodyConstraintSet) or isinstance(constraint, EndEffectorConstraintSet):
            global_rots = constraint.global_joints_rots
            global_positions = constraint.global_joints_positions
            if isinstance(frame_indices, torch.Tensor):
                global_rots = global_rots[valid_mask]
                global_positions = global_positions[valid_mask]
                root_2d = constraint.root_2d[valid_mask]
            else:
                global_rots = global_rots[valid_positions]
                global_positions = global_positions[valid_positions]
                root_2d = constraint.root_2d[valid_positions]
            root_positions = global_positions[:, skeleton.root_idx]
            if isinstance(constraint, EndEffectorConstraintSet) and 'Hips' not in getattr(constraint, 'joint_names', []):
                root_positions[:, 0] = root_2d[:, 0]
                root_positions[:, 2] = root_2d[:, 1]
            local_rot_mats = skeleton.global_rots_to_local_rots(global_rots)
            local_rot_quats = matrix_to_quaternion(local_rot_mats)
            hip_translations_input[frame_indices] = _match_hip_dtype(root_positions)
            rotations_input[frame_indices] = _match_rot_dtype(local_rot_quats)
        else:
            NotImplementedError(f'Constraint {constraint.name} is not supported')
    return (hip_translations_input, rotations_input)

def create_working_rig_from_skeleton(skeleton: SkeletonBase, above_ground_offset: float=0.007) -> List[SimpleNamespace]:
    working_rig_joints = []
    joint_names = skeleton.bone_order_names
    neutral_positions = skeleton.neutral_joints.cpu().numpy()
    parent_indices = skeleton.joint_parents.cpu().numpy()
    if isinstance(skeleton, G1Skeleton34):
        retarget_map = {skeleton.bone_order_names[skeleton.root_idx]: 'Hips', skeleton.left_hand_joint_names[0]: 'LeftHand', skeleton.right_hand_joint_names[0]: 'RightHand', skeleton.left_foot_joint_names[0]: 'LeftFoot', skeleton.right_foot_joint_names[0]: 'RightFoot'}
    else:
        retarget_map = {'Hips': 'Hips', 'Head': 'Head', 'LeftHand': 'LeftHand', 'RightHand': 'RightHand', 'LeftFoot': 'LeftFoot', 'RightFoot': 'RightFoot'}
    for (i, joint_name) in enumerate(joint_names):
        parent_name = None if parent_indices[i] == -1 else joint_names[parent_indices[i]]
        if parent_indices[i] == -1:
            toe_height = neutral_positions[:, 1].min()
            local_translation = (neutral_positions[i] + np.array([0.0, -toe_height + above_ground_offset, 0.0])).tolist()
        else:
            parent_idx = parent_indices[i]
            parent_position = neutral_positions[parent_idx]
            joint_position = neutral_positions[i]
            local_translation = (joint_position - parent_position).tolist()
        default_rotation = [0.0, 0.0, 0.0, 1.0]
        joint_info = SimpleNamespace(name=joint_name, parent=parent_name, t_pose_rotation=default_rotation, t_pose_translation=local_translation, retarget_tag=retarget_map.get(joint_name))
        working_rig_joints.append(joint_info)
    return working_rig_joints

def post_process_motion(local_rot_mats: torch.Tensor, root_positions: torch.Tensor, contacts: torch.Tensor, skeleton: SkeletonBase, constraint_lst: Optional[List]=None, contact_threshold: float=0.5, root_margin: float=0.04) -> Dict[str, torch.Tensor]:
    assert local_rot_mats.dim() == 5, 'local_rot_mats should be 5D, make sure to include the batch dimension'
    (batch_size, num_frames, num_joints) = local_rot_mats.shape[:3]

    def _build_constraint_masks_dict(constraints: List) -> Dict[str, torch.Tensor]:
        out = {key: torch.zeros(num_frames, dtype=torch.float32) for key in ['FullBody', 'LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Root']}
        for constraint in constraints:
            frame_indices = constraint.frame_indices
            if isinstance(frame_indices, torch.Tensor):
                frame_indices = frame_indices[frame_indices < num_frames]
                if frame_indices.numel() == 0:
                    continue
            else:
                frame_indices = [idx for idx in frame_indices if idx < num_frames]
                if not frame_indices:
                    continue
            if constraint.name == 'fullbody':
                out['FullBody'][frame_indices] = 1.0
            elif constraint.name == 'left-foot':
                out['LeftFoot'][frame_indices] = 1.0
            elif constraint.name == 'right-foot':
                out['RightFoot'][frame_indices] = 1.0
            elif constraint.name == 'left-hand':
                out['LeftHand'][frame_indices] = 1.0
            elif constraint.name == 'right-hand':
                out['RightHand'][frame_indices] = 1.0
            elif constraint.name == 'root2d':
                out['Root'][frame_indices] = 1.0
        return out
    batched_constraints = bool(constraint_lst) and isinstance(constraint_lst[0], list)
    if batched_constraints:
        constraint_masks_dict_lst = [_build_constraint_masks_dict(constraint_lst[b]) for b in range(batch_size)]
    else:
        constraint_masks_dict = _build_constraint_masks_dict(constraint_lst) if constraint_lst else {key: torch.zeros(num_frames, dtype=torch.float32) for key in ['FullBody', 'LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Root']}
    working_rig = create_working_rig_from_skeleton(skeleton)
    has_double_ankle_joints = isinstance(skeleton, G1Skeleton34)
    neutral_joints_pelvis_offset = skeleton.neutral_joints[0].cpu().clone()
    hip_translations_corrected = root_positions.cpu().clone()
    rotations_corrected = matrix_to_quaternion(local_rot_mats).cpu().clone()
    contacts = contacts.cpu()
    hip_translations_input = torch.zeros(batch_size, num_frames, 3)
    rotations_input = torch.zeros(batch_size, num_frames, num_joints, 4)
    rotations_input[..., 0] = 1.0
    if constraint_lst:
        for b in range(batch_size):
            constraints_lst_el = constraint_lst[b] if isinstance(constraint_lst[0], list) else constraint_lst
            (hip_translations_input[b], rotations_input[b]) = extract_input_motion_from_constraints(constraints_lst_el, skeleton, num_frames, num_joints)
    try:
        from motion_correction import motion_postprocess
    except ImportError as e:
        raise RuntimeError('Motion correction is required for this postprocessing path but the motion_correction package is not installed. Install with: pip install -e .') from e
    for b in range(batch_size):
        motion_postprocess.correct_motion(hip_translations_corrected[b:b + 1], rotations_corrected[b:b + 1], contacts[b:b + 1], hip_translations_input[b:b + 1], rotations_input[b:b + 1], constraint_masks_dict_lst[b] if batched_constraints else constraint_masks_dict, contact_threshold, root_margin, working_rig, has_double_ankle_joints)
    local_rot_mats_corrected = quaternion_to_matrix(rotations_corrected)
    device = local_rot_mats.device
    (global_rot_mats, posed_joints, _) = fk(local_rot_mats_corrected.to(device), hip_translations_corrected.to(device), skeleton)
    result = {'local_rot_mats': local_rot_mats_corrected.to(device), 'root_positions': hip_translations_corrected.to(device), 'posed_joints': posed_joints, 'global_rot_mats': global_rot_mats}
    return result
