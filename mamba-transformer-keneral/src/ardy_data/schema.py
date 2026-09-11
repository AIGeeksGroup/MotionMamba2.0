"""Shared native-motion schema for rotation-preserving dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class NativeMotion:
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    offsets_m: torch.Tensor
    neutral_joints_m: torch.Tensor
    local_rot_mats: torch.Tensor
    root_positions_m: torch.Tensor
    fps: float
    euler_orders: tuple[str, ...]
    channels: tuple[tuple[str, ...], ...]
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "NativeMotion":
        frames, joints = self.local_rot_mats.shape[:2]
        if self.local_rot_mats.shape != (frames, joints, 3, 3):
            raise ValueError(f"invalid local rotation shape {tuple(self.local_rot_mats.shape)}")
        if joints != len(self.joint_names) or len(self.parents) != joints:
            raise ValueError("joint hierarchy and rotation dimensions disagree")
        if self.root_positions_m.shape != (frames, 3):
            raise ValueError(f"invalid root shape {tuple(self.root_positions_m.shape)}")
        if self.offsets_m.shape != (joints, 3) or self.neutral_joints_m.shape != (joints, 3):
            raise ValueError("invalid skeleton offset/neutral-joint shape")
        if self.parents[0] != -1 or any(parent >= index for index, parent in enumerate(self.parents[1:], 1)):
            raise ValueError(f"hierarchy is not parent-before-child: {self.parents}")
        if not self.fps > 0 or frames < 2:
            raise ValueError(f"invalid timing: frames={frames}, fps={self.fps}")
        if not torch.isfinite(self.local_rot_mats).all() or not torch.isfinite(self.root_positions_m).all():
            raise ValueError("motion contains non-finite values")
        return self

    def with_motion(self, rotations: torch.Tensor, root: torch.Tensor, fps: float) -> "NativeMotion":
        return replace(self, local_rot_mats=rotations, root_positions_m=root, fps=float(fps)).validate()
