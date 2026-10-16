"""Name-mapped G1 rotations/translation export and physical-time resampling."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class G1EvaluatorConversion:
    def __init__(self, joint_order):
        from ardy_data.bones_seed import BonesSeedG1Converter
        self.converter = BonesSeedG1Converter()
        names = self.converter.skeleton.bone_order_names
        self.mapping = {names[index].replace('_skel', '_joint'): (index, axis, offset)
                        for index, axis, offset in self.converter.joints}
        if len(joint_order) != 29 or set(joint_order) != set(self.mapping):
            raise ValueError('G1 joint names do not form a complete one-to-one mapping')
        self.joint_order = list(joint_order)

    def export(self, local_rotations, root_positions):
        local = np.asarray(local_rotations, dtype=np.float64)
        root = np.asarray(root_positions, dtype=np.float64)
        if local.shape[1:] != (34, 3, 3) or root.shape != (len(local), 3):
            raise ValueError('expected G1Skeleton34 local rotations and root metres')
        transform = self.converter.mujoco_to_ardy
        root_rotation = transform.inv() * Rotation.from_matrix(local[:, 0]) * transform
        qpos = np.zeros((len(local), 36), dtype=np.float64)
        qpos[:, :3] = transform.inv().apply(root)
        qpos[:, 3:7] = root_rotation.as_quat(scalar_first=True)
        residuals = []
        for column, name in enumerate(self.joint_order):
            index, axis, offset = self.mapping[name]
            rotation = transform.inv() * Rotation.from_matrix(local[:, index]) * transform
            if offset is not None:
                rotation = offset.inv() * rotation
            quat = rotation.as_quat(scalar_first=True)
            component = 'xyz'.index(axis) + 1
            angle = 2 * np.arctan2(quat[:, component], quat[:, 0])
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            qpos[:, column + 7] = angle
            projected = Rotation.from_euler(axis, angle[:, None])
            residuals.append((projected.inv() * rotation).magnitude())
        return qpos.astype(np.float32), dict(
            hinge_projection_residual_rad_mean=float(np.mean(residuals)),
            hinge_projection_residual_rad_max=float(np.max(residuals)),
            projection='closest axis twist, no joint clipping, no GT alignment')

    def import_qpos(self, qpos):
        q = np.asarray(qpos, dtype=np.float64)
        transform = self.converter.mujoco_to_ardy
        rotations = np.broadcast_to(np.eye(3), (len(q), 34, 3, 3)).copy()
        rotations[:, 0] = (transform * Rotation.from_quat(q[:, 3:7], scalar_first=True) * transform.inv()).as_matrix()
        for column, name in enumerate(self.joint_order):
            index, axis, offset = self.mapping[name]
            value = Rotation.from_euler(axis, q[:, column + 7, None])
            if offset is not None:
                value = offset * value
            rotations[:, index] = (transform * value * transform.inv()).as_matrix()
        return rotations, transform.apply(q[:, :3])


def resample_qpos(qpos, source_fps=25, target_fps=30):
    q = np.asarray(qpos, dtype=np.float64)
    if len(q) < 2 or q.shape[1] != 36 or not np.isfinite(q).all():
        raise ValueError('qpos must be finite [T,36] with at least two frames')
    times = np.arange(len(q), dtype=np.float64) / source_fps
    # Preserve timestamps; do not stretch the last source interval to a frame boundary.
    target = np.arange(int(np.floor(times[-1] * target_fps + 1e-9)) + 1) / target_fps
    out = np.empty((len(target), 36), dtype=np.float64)
    for index in range(3):
        out[:, index] = np.interp(target, times, q[:, index])
    out[:, 3:7] = Slerp(times, Rotation.from_quat(q[:, 3:7], scalar_first=True))(target).as_quat(scalar_first=True)
    angles = np.unwrap(q[:, 7:], axis=0)
    for index in range(29):
        out[:, index + 7] = np.interp(target, times, angles[:, index])
    return out.astype(np.float32), target
