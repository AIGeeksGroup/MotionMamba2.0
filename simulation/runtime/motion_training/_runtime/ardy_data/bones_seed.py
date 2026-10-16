"""BONES-SEED G1 CSV conversion for ARDY's native G1 motion representation."""

from __future__ import annotations

import csv
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


DESCRIPTION_COLUMNS = (
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
)


def actor_split(actor_uid: str, seed: int = 20260824) -> str:
    """Return a deterministic 90/5/5 actor-disjoint split."""
    digest = hashlib.sha256(f"{seed}:{actor_uid}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "little") % 1000
    return "train" if bucket < 900 else ("val" if bucket < 950 else "test")


def metadata_rows(path: str | Path):
    with Path(path).open(newline="") as handle:
        yield from csv.DictReader(handle)


def captions_from_row(row: dict) -> list[str]:
    captions = []
    for column in DESCRIPTION_COLUMNS:
        value = str(row.get(column, "")).strip()
        if value and value.lower() != "nan" and value not in captions:
            captions.append(value)
    return captions


class BonesSeedG1Converter:
    """Convert one 120-fps BONES-SEED G1 CSV to 25-fps ARDY G1 features."""

    source_fps = 120
    target_fps = 25

    def __init__(self):
        from ardy.assets import skeleton_asset_path
        from ardy.motion_rep import ArdyMotionRep
        from ardy.skeleton import G1Skeleton34

        self.skeleton = G1Skeleton34(load=True)
        self.motion_rep = ArdyMotionRep(self.skeleton, fps=self.target_fps)
        xml_path = skeleton_asset_path("g1skel34", "xml", "g1.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        classes = [item for item in tree.findall(".//default") if "class" in item.attrib]
        class_axes = {}
        for item in classes:
            joints = item.findall("joint")
            if joints:
                class_axes[item.get("class")] = joints[0].get("axis")
        parent_map = {child: parent for parent in root.iter() for child in parent}
        self.joints = []
        for joint in root.find("worldbody").findall(".//joint"):
            name = joint.get("name").replace("_joint", "_skel")
            axis_values = [float(value) for value in (joint.get("axis") or class_axes[joint.get("class")]).split()]
            axis = "xyz"[int(np.argmax(axis_values))]
            body = parent_map[joint]
            offset = None
            if "quat" in body.attrib:
                offset = Rotation.from_quat(
                    [float(value) for value in body.get("quat").split()], scalar_first=True
                )
            self.joints.append((self.skeleton.bone_order_names.index(name), axis, offset))
        if len(self.joints) != 29:
            raise RuntimeError(f"expected 29 G1 hinge joints, got {len(self.joints)}")
        z_up_to_y_up = Rotation.from_euler("x", -90, degrees=True)
        x_forward_to_z_forward = Rotation.from_euler("z", -90, degrees=True)
        self.mujoco_to_ardy = z_up_to_y_up * x_forward_to_z_forward

    @staticmethod
    def _sample_indices(frames: int) -> np.ndarray:
        count = int(np.floor((frames - 1) * BonesSeedG1Converter.target_fps / BonesSeedG1Converter.source_fps)) + 1
        return np.rint(
            np.arange(count, dtype=np.float64)
            * BonesSeedG1Converter.source_fps
            / BonesSeedG1Converter.target_fps
        ).astype(np.int64)

    def convert(self, path: str | Path) -> torch.Tensor:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        if data.ndim == 1:
            data = data[None]
        if data.shape[1] != 36:
            raise ValueError(f"expected Frame + 35 G1 values, got {data.shape} in {path}")
        values = data[self._sample_indices(len(data)), 1:]
        root_translation = self.mujoco_to_ardy.apply(values[:, :3]) * 0.01
        root_rotation = Rotation.from_euler("xyz", values[:, 3:6], degrees=True)
        root_rotation = self.mujoco_to_ardy * root_rotation * self.mujoco_to_ardy.inv()
        local = np.broadcast_to(
            np.eye(3, dtype=np.float64), (len(values), self.skeleton.nbjoints, 3, 3)
        ).copy()
        local[:, 0] = root_rotation.as_matrix()
        for column, (joint_index, axis, offset) in enumerate(self.joints):
            rotation = Rotation.from_euler(axis, np.deg2rad(values[:, column + 6])[:, None])
            if offset is not None:
                rotation = offset * rotation
            local[:, joint_index] = (
                self.mujoco_to_ardy * rotation * self.mujoco_to_ardy.inv()
            ).as_matrix()
        local_tensor = torch.from_numpy(local).float()
        root_tensor = torch.from_numpy(root_translation).float()
        features = self.motion_rep(
            local_tensor[None], root_tensor[None], to_normalize=False, to_canonicalize=False
        )
        features = self.motion_rep.rotate_to(features, torch.tensor(0.0))
        first_xz = features[:, 0, [0, 2]].clone()
        features = self.motion_rep.translate_2d(features, -first_xz)
        return features[0].contiguous()


def dual_root_stats(features: torch.Tensor, motion_rep) -> np.ndarray:
    length = torch.tensor([len(features)])
    local_root = motion_rep.global_root_to_local_root(
        features[None, :, :5], normalized=False, lengths=length
    )[0]
    return torch.cat((features[:, :5], local_root, features[:, 5:]), dim=-1).numpy().astype(np.float64)
