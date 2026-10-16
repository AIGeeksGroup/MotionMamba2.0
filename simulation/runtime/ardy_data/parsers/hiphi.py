"""HiPHI BVH parser that derives its hierarchy, channels, and timing from each file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ardy.skeleton.bvh import Bvh

from ardy_data.schema import NativeMotion


def _neutral_joints(offsets: np.ndarray, parents: tuple[int, ...]) -> np.ndarray:
    neutral = offsets.copy()
    neutral[0] = 0.0
    for index in range(1, len(parents)):
        neutral[index] = neutral[parents[index]] + offsets[index]
    return neutral


def parse_hiphi_bvh(path: str | Path, metadata: dict | None = None) -> NativeMotion:
    path = Path(path)
    mocap = Bvh(path.read_text(), backend="np")
    joints = mocap.get_joints()
    names = tuple(joint.name for joint in joints)
    parents = tuple(mocap.joint_parent_index(name) for name in names)
    channels = tuple(tuple(mocap.joint_channels(name)) for name in names)
    offsets_m = np.asarray([mocap.joint_offset(name) for name in names], dtype=np.float64) * 0.01
    neutral_m = _neutral_joints(offsets_m, parents)
    if mocap.nframes != len(mocap.frames):
        raise ValueError(f"declared Frames {mocap.nframes} != parsed rows {len(mocap.frames)} in {path}")

    local = np.empty((mocap.nframes, len(names), 3, 3), dtype=np.float64)
    euler_orders = []
    for index, (name, joint_channels) in enumerate(zip(names, channels)):
        rotation_channels = [channel for channel in joint_channels if channel.endswith("rotation")]
        if len(rotation_channels) != 3:
            raise ValueError(f"joint {name} declares rotation channels {rotation_channels}")
        order = "".join(channel[0].upper() for channel in rotation_channels)
        if len(set(order)) != 3:
            raise ValueError(f"joint {name} has invalid Euler order {order}")
        angles = np.asarray(mocap.frames_joint_channels(name, rotation_channels), dtype=np.float64)
        local[:, index] = Rotation.from_euler(order, angles, degrees=True).as_matrix()
        euler_orders.append(order)

    root_channels = channels[0]
    root_position_channels = [channel for channel in root_channels if channel.endswith("position")]
    if set(root_position_channels) != {"Xposition", "Yposition", "Zposition"}:
        raise ValueError(f"root must declare XYZ positions, got {root_position_channels}")
    raw_root = {
        channel[0].upper(): np.asarray(mocap.frames_joint_channels(names[0], [channel]), dtype=np.float64).reshape(-1)
        for channel in root_position_channels
    }
    root_m = np.stack([raw_root[axis] for axis in "XYZ"], axis=-1) * 0.01

    non_root_translation_max_m = 0.0
    for name, joint_channels in zip(names[1:], channels[1:]):
        position_channels = [channel for channel in joint_channels if channel.endswith("position")]
        if position_channels:
            values = np.asarray(mocap.frames_joint_channels(name, position_channels), dtype=np.float64) * 0.01
            non_root_translation_max_m = max(non_root_translation_max_m, float(np.abs(values).max(initial=0.0)))
    if non_root_translation_max_m > 1e-4:
        raise ValueError(
            f"dynamic non-root translations are unsupported; max={non_root_translation_max_m:.6g}m in {path}"
        )

    details = dict(metadata or {})
    details["bvh_contract"] = {
        "joint_count": len(names),
        "frame_count": mocap.nframes,
        "frame_time": mocap.frame_time,
        "fps": 1.0 / mocap.frame_time,
        "root_channels": list(root_channels),
        "euler_orders": sorted(set(euler_orders)),
        "linear_unit": "centimeter",
        "converted_root_unit": "meter",
        "non_root_translation_max_m": non_root_translation_max_m,
        "source_skeleton_height_m": float(neutral_m[:, 1].max() - neutral_m[:, 1].min()),
    }
    return NativeMotion(
        joint_names=names,
        parents=parents,
        offsets_m=torch.from_numpy(offsets_m).float(),
        neutral_joints_m=torch.from_numpy(neutral_m).float(),
        local_rot_mats=torch.from_numpy(local).float(),
        root_positions_m=torch.from_numpy(root_m).float(),
        fps=1.0 / mocap.frame_time,
        euler_orders=tuple(euler_orders),
        channels=channels,
        source_path=path,
        metadata=details,
    ).validate()
