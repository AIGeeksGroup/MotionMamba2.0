# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Skeleton.'
from pathlib import Path
import numpy as np
import torch
from motion_mamba2.model.tensor_utils import ensure_batched
from motion_mamba2.model.skeleton_base import SkeletonBase

class SOMASkeleton77(SkeletonBase):
    name = 'somaskel77'
    right_foot_joint_names = ['RightFoot', 'RightToeBase', 'RightToeEnd']
    left_foot_joint_names = ['LeftFoot', 'LeftToeBase', 'LeftToeEnd']
    right_hand_joint_names = ['RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumbEnd', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4', 'RightHandIndexEnd', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 'RightHandMiddleEnd', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightHandRingEnd', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 'RightHandPinkyEnd']
    left_hand_joint_names = ['LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumbEnd', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4', 'LeftHandIndexEnd', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4', 'LeftHandMiddleEnd', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandRingEnd', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 'LeftHandPinkyEnd']
    hip_joint_names = ['RightLeg', 'LeftLeg']
    bone_order_names_with_parents = [('Hips', None), ('Spine1', 'Hips'), ('Spine2', 'Spine1'), ('Chest', 'Spine2'), ('Neck1', 'Chest'), ('Neck2', 'Neck1'), ('Head', 'Neck2'), ('HeadEnd', 'Head'), ('Jaw', 'Head'), ('LeftEye', 'Head'), ('RightEye', 'Head'), ('LeftShoulder', 'Chest'), ('LeftArm', 'LeftShoulder'), ('LeftForeArm', 'LeftArm'), ('LeftHand', 'LeftForeArm'), ('LeftHandThumb1', 'LeftHand'), ('LeftHandThumb2', 'LeftHandThumb1'), ('LeftHandThumb3', 'LeftHandThumb2'), ('LeftHandThumbEnd', 'LeftHandThumb3'), ('LeftHandIndex1', 'LeftHand'), ('LeftHandIndex2', 'LeftHandIndex1'), ('LeftHandIndex3', 'LeftHandIndex2'), ('LeftHandIndex4', 'LeftHandIndex3'), ('LeftHandIndexEnd', 'LeftHandIndex4'), ('LeftHandMiddle1', 'LeftHand'), ('LeftHandMiddle2', 'LeftHandMiddle1'), ('LeftHandMiddle3', 'LeftHandMiddle2'), ('LeftHandMiddle4', 'LeftHandMiddle3'), ('LeftHandMiddleEnd', 'LeftHandMiddle4'), ('LeftHandRing1', 'LeftHand'), ('LeftHandRing2', 'LeftHandRing1'), ('LeftHandRing3', 'LeftHandRing2'), ('LeftHandRing4', 'LeftHandRing3'), ('LeftHandRingEnd', 'LeftHandRing4'), ('LeftHandPinky1', 'LeftHand'), ('LeftHandPinky2', 'LeftHandPinky1'), ('LeftHandPinky3', 'LeftHandPinky2'), ('LeftHandPinky4', 'LeftHandPinky3'), ('LeftHandPinkyEnd', 'LeftHandPinky4'), ('RightShoulder', 'Chest'), ('RightArm', 'RightShoulder'), ('RightForeArm', 'RightArm'), ('RightHand', 'RightForeArm'), ('RightHandThumb1', 'RightHand'), ('RightHandThumb2', 'RightHandThumb1'), ('RightHandThumb3', 'RightHandThumb2'), ('RightHandThumbEnd', 'RightHandThumb3'), ('RightHandIndex1', 'RightHand'), ('RightHandIndex2', 'RightHandIndex1'), ('RightHandIndex3', 'RightHandIndex2'), ('RightHandIndex4', 'RightHandIndex3'), ('RightHandIndexEnd', 'RightHandIndex4'), ('RightHandMiddle1', 'RightHand'), ('RightHandMiddle2', 'RightHandMiddle1'), ('RightHandMiddle3', 'RightHandMiddle2'), ('RightHandMiddle4', 'RightHandMiddle3'), ('RightHandMiddleEnd', 'RightHandMiddle4'), ('RightHandRing1', 'RightHand'), ('RightHandRing2', 'RightHandRing1'), ('RightHandRing3', 'RightHandRing2'), ('RightHandRing4', 'RightHandRing3'), ('RightHandRingEnd', 'RightHandRing4'), ('RightHandPinky1', 'RightHand'), ('RightHandPinky2', 'RightHandPinky1'), ('RightHandPinky3', 'RightHandPinky2'), ('RightHandPinky4', 'RightHandPinky3'), ('RightHandPinkyEnd', 'RightHandPinky4'), ('LeftLeg', 'Hips'), ('LeftShin', 'LeftLeg'), ('LeftFoot', 'LeftShin'), ('LeftToeBase', 'LeftFoot'), ('LeftToeEnd', 'LeftToeBase'), ('RightLeg', 'Hips'), ('RightShin', 'RightLeg'), ('RightFoot', 'RightShin'), ('RightToeBase', 'RightFoot'), ('RightToeEnd', 'RightToeBase')]

    @property
    def relaxed_hands_rest_pose(self):
        if hasattr(self, '_relaxed_hands_rest_pose'):
            return self._relaxed_hands_rest_pose
        relaxed_hands_pose_path = Path(self.folder) / 'relaxed_hands_rest_pose.npy'
        relaxed_hands_rest_pose = torch.from_numpy(np.load(relaxed_hands_pose_path)).squeeze()
        self.register_buffer('_relaxed_hands_rest_pose', relaxed_hands_rest_pose, persistent=False)
        return self._relaxed_hands_rest_pose

class SOMASkeleton30(SkeletonBase):
    name = 'somaskel30'
    right_foot_joint_names = ['RightFoot', 'RightToeBase']
    left_foot_joint_names = ['LeftFoot', 'LeftToeBase']
    right_hand_joint_names = ['RightHand', 'RightHandMiddleEnd']
    left_hand_joint_names = ['LeftHand', 'LeftHandMiddleEnd']
    hip_joint_names = ['RightLeg', 'LeftLeg']
    bone_order_names_with_parents = [('Hips', None), ('Spine1', 'Hips'), ('Spine2', 'Spine1'), ('Chest', 'Spine2'), ('Neck1', 'Chest'), ('Neck2', 'Neck1'), ('Head', 'Neck2'), ('Jaw', 'Head'), ('LeftEye', 'Head'), ('RightEye', 'Head'), ('LeftShoulder', 'Chest'), ('LeftArm', 'LeftShoulder'), ('LeftForeArm', 'LeftArm'), ('LeftHand', 'LeftForeArm'), ('LeftHandThumbEnd', 'LeftHand'), ('LeftHandMiddleEnd', 'LeftHand'), ('RightShoulder', 'Chest'), ('RightArm', 'RightShoulder'), ('RightForeArm', 'RightArm'), ('RightHand', 'RightForeArm'), ('RightHandThumbEnd', 'RightHand'), ('RightHandMiddleEnd', 'RightHand'), ('LeftLeg', 'Hips'), ('LeftShin', 'LeftLeg'), ('LeftFoot', 'LeftShin'), ('LeftToeBase', 'LeftFoot'), ('RightLeg', 'Hips'), ('RightShin', 'RightLeg'), ('RightFoot', 'RightShin'), ('RightToeBase', 'RightFoot')]

    @property
    def somaskel77(self):
        if not hasattr(self, '_somaskel77'):
            self._somaskel77 = SOMASkeleton77()
        return self._somaskel77

    @ensure_batched(local_joint_rots_subset=4)
    def to_SOMASkeleton77(self, local_joint_rots_subset: torch.Tensor):
        device = local_joint_rots_subset.device
        nF = len(local_joint_rots_subset)
        local_joint_rots_mats = self.somaskel77.relaxed_hands_rest_pose.clone().to(device).repeat(nF, 1, 1, 1)
        skel_slice = self.get_skel_slice(self.somaskel77)
        local_joint_rots_mats[:, skel_slice] = local_joint_rots_subset
        return local_joint_rots_mats

    @ensure_batched(local_joint_rots_full=4)
    def from_SOMASkeleton77(self, local_joint_rots_full: torch.Tensor) -> torch.Tensor:
        skel_slice = self.get_skel_slice(self.somaskel77)
        return local_joint_rots_full[:, skel_slice]

    def output_to_SOMASkeleton77(self, output: dict) -> dict:
        local_rot_mats_77 = self.to_SOMASkeleton77(output['local_rot_mats'])
        root_positions = output['root_positions']
        (global_rot_mats_77, posed_joints_77, _) = self.somaskel77.fk(local_rot_mats_77, root_positions)
        out_77 = dict(output)
        out_77['local_rot_mats'] = local_rot_mats_77
        out_77['global_rot_mats'] = global_rot_mats_77
        out_77['posed_joints'] = posed_joints_77
        if 'foot_contacts' in output:
            fc = output['foot_contacts']
            out_77['foot_contacts'] = torch.cat([fc[..., :2], fc[..., 1:2], fc[..., 2:4], fc[..., 3:4]], dim=-1)
        return out_77

class G1Skeleton34(SkeletonBase):
    name = 'g1skel34'
    right_foot_joint_names = ['right_ankle_roll_skel', 'right_toe_base']
    left_foot_joint_names = ['left_ankle_roll_skel', 'left_toe_base']
    right_hand_joint_names = ['right_wrist_yaw_skel', 'right_hand_roll_skel']
    left_hand_joint_names = ['left_wrist_yaw_skel', 'left_hand_roll_skel']
    hip_joint_names = ['right_hip_pitch_skel', 'left_hip_pitch_skel']
    bone_order_names_with_parents = [('pelvis_skel', None), ('left_hip_pitch_skel', 'pelvis_skel'), ('left_hip_roll_skel', 'left_hip_pitch_skel'), ('left_hip_yaw_skel', 'left_hip_roll_skel'), ('left_knee_skel', 'left_hip_yaw_skel'), ('left_ankle_pitch_skel', 'left_knee_skel'), ('left_ankle_roll_skel', 'left_ankle_pitch_skel'), ('left_toe_base', 'left_ankle_roll_skel'), ('right_hip_pitch_skel', 'pelvis_skel'), ('right_hip_roll_skel', 'right_hip_pitch_skel'), ('right_hip_yaw_skel', 'right_hip_roll_skel'), ('right_knee_skel', 'right_hip_yaw_skel'), ('right_ankle_pitch_skel', 'right_knee_skel'), ('right_ankle_roll_skel', 'right_ankle_pitch_skel'), ('right_toe_base', 'right_ankle_roll_skel'), ('waist_yaw_skel', 'pelvis_skel'), ('waist_roll_skel', 'waist_yaw_skel'), ('waist_pitch_skel', 'waist_roll_skel'), ('left_shoulder_pitch_skel', 'waist_pitch_skel'), ('left_shoulder_roll_skel', 'left_shoulder_pitch_skel'), ('left_shoulder_yaw_skel', 'left_shoulder_roll_skel'), ('left_elbow_skel', 'left_shoulder_yaw_skel'), ('left_wrist_roll_skel', 'left_elbow_skel'), ('left_wrist_pitch_skel', 'left_wrist_roll_skel'), ('left_wrist_yaw_skel', 'left_wrist_pitch_skel'), ('left_hand_roll_skel', 'left_wrist_yaw_skel'), ('right_shoulder_pitch_skel', 'waist_pitch_skel'), ('right_shoulder_roll_skel', 'right_shoulder_pitch_skel'), ('right_shoulder_yaw_skel', 'right_shoulder_roll_skel'), ('right_elbow_skel', 'right_shoulder_yaw_skel'), ('right_wrist_roll_skel', 'right_elbow_skel'), ('right_wrist_pitch_skel', 'right_wrist_roll_skel'), ('right_wrist_yaw_skel', 'right_wrist_pitch_skel'), ('right_hand_roll_skel', 'right_wrist_yaw_skel')]

class CoreSkeleton27(SkeletonBase):
    name = 'cskel27'
    right_foot_joint_names = ['RightFoot', 'RightToeBase']
    left_foot_joint_names = ['LeftFoot', 'LeftToeBase']
    right_hand_joint_names = ['RightHand', 'RightHandEnd']
    left_hand_joint_names = ['LeftHand', 'LeftHandEnd']
    hip_joint_names = ['RightUpLeg', 'LeftUpLeg']
    bone_order_names_with_parents = [('Hips', None), ('Spine', 'Hips'), ('Spine1', 'Spine'), ('Spine2', 'Spine1'), ('Spine3', 'Spine2'), ('Neck', 'Spine3'), ('Head', 'Neck'), ('RightShoulder', 'Spine3'), ('RightArm', 'RightShoulder'), ('RightForeArm', 'RightArm'), ('RightHand', 'RightForeArm'), ('RightHandEnd', 'RightHand'), ('RightHandThumb1', 'RightHand'), ('LeftShoulder', 'Spine3'), ('LeftArm', 'LeftShoulder'), ('LeftForeArm', 'LeftArm'), ('LeftHand', 'LeftForeArm'), ('LeftHandEnd', 'LeftHand'), ('LeftHandThumb1', 'LeftHand'), ('RightUpLeg', 'Hips'), ('RightLeg', 'RightUpLeg'), ('RightFoot', 'RightLeg'), ('RightToeBase', 'RightFoot'), ('LeftUpLeg', 'Hips'), ('LeftLeg', 'LeftUpLeg'), ('LeftFoot', 'LeftLeg'), ('LeftToeBase', 'LeftFoot')]

from motion_mamba2.model.kinematics import batch_rigid_transform, fk
from motion_mamba2.model.skeleton_transforms import global_rots_to_local_rots, to_standard_tpose
