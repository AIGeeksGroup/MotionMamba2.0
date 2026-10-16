"""Semantic global-rotation retargeting from HiPHI to CoreSkeleton27."""

from __future__ import annotations

import torch

from ardy_data.schema import NativeMotion


HIPHI_TO_CORE27 = {
    "Hips": "Hips",
    "Spine": "Spine",
    "Spine1": "Spine1",
    "Spine2": "Spine2",
    "Spine3": "Spine3",
    "Neck": "Neck",
    "Head": "Head",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "RightHandEnd": "RightHandMiddle1",
    "RightHandThumb1": "RightHandThumb1",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "LeftHandEnd": "LeftHandMiddle1",
    "LeftHandThumb1": "LeftHandThumb1",
    "RightUpLeg": "RightUpLeg",
    "RightLeg": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
    "LeftUpLeg": "LeftUpLeg",
    "LeftLeg": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
}


def _global_from_local(local: torch.Tensor, parents: tuple[int, ...]) -> torch.Tensor:
    result = local.clone()
    for index in range(1, len(parents)):
        result[:, index] = result[:, parents[index]] @ local[:, index]
    return result


def _local_from_global(global_rotations: torch.Tensor, parents: tuple[int, ...]) -> torch.Tensor:
    result = global_rotations.clone()
    for index in range(1, len(parents)):
        result[:, index] = global_rotations[:, parents[index]].transpose(-1, -2) @ global_rotations[:, index]
    return result


def _leg_length(joints: torch.Tensor, indices: tuple[int, ...]) -> float:
    return sum(float((joints[child] - joints[parent]).norm()) for parent, child in zip(indices, indices[1:]))


def retarget_hiphi_to_core27(motion: NativeMotion, skeleton) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Map source global rotations, then recover Core27 local rotations."""
    source_index = {name: index for index, name in enumerate(motion.joint_names)}
    target_names = tuple(skeleton.bone_order_names)
    target_parents = tuple(int(value) for value in skeleton.joint_parents.tolist())
    missing_source = sorted({source for source in HIPHI_TO_CORE27.values() if source not in source_index})
    if missing_source:
        raise ValueError(f"HiPHI skeleton is missing mapped joints: {missing_source}")
    missing_targets = sorted(set(target_names) - set(HIPHI_TO_CORE27))
    if missing_targets:
        raise ValueError(f"Core27 mapping is incomplete: {missing_targets}")

    source_global = _global_from_local(motion.local_rot_mats, motion.parents)
    basis = torch.eye(3, dtype=source_global.dtype, device=source_global.device)
    transformed = basis @ source_global @ basis.transpose(-1, -2)
    target_global = torch.empty(
        (len(transformed), len(target_names), 3, 3), dtype=transformed.dtype, device=transformed.device
    )
    for target_index, target_name in enumerate(target_names):
        target_global[:, target_index] = transformed[:, source_index[HIPHI_TO_CORE27[target_name]]]
    target_local = _local_from_global(target_global, target_parents)

    source_leg = _leg_length(
        motion.neutral_joints_m,
        tuple(source_index[name] for name in ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot")),
    )
    target_index = {name: index for index, name in enumerate(target_names)}
    target_leg = _leg_length(
        skeleton.neutral_joints.detach().to(motion.neutral_joints_m),
        tuple(target_index[name] for name in ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot")),
    )
    if source_leg <= 1e-6:
        raise ValueError(f"invalid source leg length {source_leg}")
    scale = target_leg / source_leg
    root = motion.root_positions_m * scale
    provenance = {
        "mapping_version": "hiphi_core27_global_v1",
        "source_joint_list": list(motion.joint_names),
        "target_joint_list": list(target_names),
        "semantic_mapping": dict(HIPHI_TO_CORE27),
        "unused_source_joints": sorted(set(motion.joint_names) - set(HIPHI_TO_CORE27.values())),
        "missing_target_joints": [],
        "fallback_strategy": "none; HandEnd maps to source Middle1 global orientation",
        "basis_matrix": basis.tolist(),
        "root_scale": scale,
        "source_leg_length_m": source_leg,
        "target_leg_length_m": target_leg,
    }
    return target_local.contiguous(), root.contiguous(), provenance
