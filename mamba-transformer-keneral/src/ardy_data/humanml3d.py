"""Native AMASS -> CoreSkeleton27 conversion for HumanML3D.

The processed 263-D HumanML3D vectors are intentionally not used here.  This
module starts from the AMASS axis-angle motion, keeps native rotations, and
rebuilds ARDY's 330-D representation after retargeting.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


CORE_JOINTS = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd",
    "RightHandThumb1", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandEnd", "LeftHandThumb1", "RightUpLeg", "RightLeg", "RightFoot",
    "RightToeBase", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
)

# SMPL-H body joints in the AMASS 22-joint order.  The target has extra spine,
# hand-end, thumb, and toe joints; those are filled by their nearest mapped
# ancestor (the target rest pose supplies their bone lengths).
SMPL22_TO_CORE = {
    0: 0, 3: 1, 6: 3, 9: 4, 12: 5, 15: 6,
    14: 7, 17: 8, 19: 9, 21: 10,
    13: 13, 16: 14, 18: 15, 20: 16,
    2: 19, 5: 20, 8: 21, 11: 22,
    1: 23, 4: 24, 7: 25, 10: 26,
}

SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
CORE_PARENTS = (-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16, 0, 19, 20, 21, 0, 23, 24, 25)

# AMASS is z-up in world coordinates while CoreSkeleton27 is y-up.  This is a
# deliberate basis change; it is allowed to be a reflection because rotations
# are conjugated by the same matrix.  We preserve x handedness so anatomical
# left/right remain left/right in the Core skeleton.
AMASS_TO_CORE = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))


@dataclass(frozen=True)
class HumanML3DRow:
    source_path: Path
    new_name: str
    start_frame: int
    end_frame: int
    dataset: str

    @property
    def is_humanact12(self) -> bool:
        return "humanact12" in self.source_path.as_posix().lower()


def load_split_ids(dataset_root: str | Path, split: str) -> set[str]:
    root = Path(dataset_root)
    path = root / "HumanML3D" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


class HumanML3DArdyDataset(torch.utils.data.Dataset):
    """Manifest-backed dataset with fixed windows and sampled controls."""

    def __init__(
        self,
        manifest: str | Path,
        stats_dir: str | Path,
        text_root: str | Path | None = None,
        frames: int = 80,
        horizon: int = 40,
        control_probability: float = 1.0,
        condition_dropout: float = 0.1,
        seed: int = 0,
        text_cache_dir: str | Path | None = None,
        history_probability: float = 0.0,
        text_dropout: float = 0.0,
    ):
        self.records = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"empty manifest: {manifest}")
        self.frames = frames
        self.horizon = horizon
        self.control_probability = control_probability
        self.condition_dropout = condition_dropout
        self.seed = seed
        self.text_root = Path(text_root) if text_root is not None else None
        self.text_cache = None
        self.text_cache_index = None
        self.text_cache_caption = None
        self.history_probability = history_probability
        if not 0.0 <= text_dropout <= 1.0:
            raise ValueError("text_dropout must be in [0, 1]")
        self.text_dropout = text_dropout
        self.history_eligible_indices = [
            index
            for index, record in enumerate(self.records)
            if int(record.get("frames", 0)) >= max(self.frames, 2 * self.horizon)
        ]
        if text_cache_dir is not None:
            cache = Path(text_cache_dir)
            metadata = json.loads((cache / "index.json").read_text())
            self.text_cache_index = {
                item["id"]: int(item["row"]) for item in metadata["samples"]
            }
            self.text_cache_caption = {
                item["id"]: item["caption"] for item in metadata["samples"]
            }
            missing = [record["id"] for record in self.records if record["id"] not in self.text_cache_index]
            if missing:
                raise ValueError(f"text cache is missing {len(missing)} manifest IDs; first={missing[0]}")
            self.text_cache = np.load(cache / "embeddings.npy", mmap_mode="r")
            if self.text_cache.shape[1:] != (4096,):
                raise ValueError(f"expected [N,4096] text cache, got {self.text_cache.shape}")
        mean = np.load(Path(stats_dir) / "mean.npy")
        std = np.load(Path(stats_dir) / "std.npy")
        if mean.ndim != 1 or std.shape != mean.shape or len(mean) < 10:
            raise ValueError(
                f"ARDY dual-root stats must be matching 1-D arrays, got {mean.shape} and {std.shape}"
            )
        # Explicit tensors use [global root 5, body 325], skipping local-root.
        explicit_indices = np.r_[np.arange(5), np.arange(9, len(mean))]
        self.mean = torch.from_numpy(mean[explicit_indices]).float()
        self.std = torch.from_numpy(std[explicit_indices]).float().clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.records)

    def _caption(self, record: dict, rng: np.random.Generator) -> str:
        sample_id = record["id"]
        if record.get("captions"):
            captions = [str(value).strip() for value in record["captions"] if str(value).strip()]
            return captions[int(rng.integers(len(captions)))] if captions else ""
        if self.text_root is None:
            return ""
        path = self.text_root / f"{sample_id.removeprefix('M')}.txt"
        if not path.exists():
            return ""
        captions = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines() if line.strip()]
        return captions[int(rng.integers(len(captions)))] if captions else ""

    def sample(
        self,
        index: int,
        augmentation_seed: int,
        *,
        control: str | None = None,
        out_of_horizon: bool | None = None,
        force_history: bool | None = None,
        force_history_frames: int | None = None,
    ) -> dict:
        """Return one reproducible stochastic view of a manifest record."""
        if force_history is not None and force_history_frames is not None:
            raise ValueError("force_history and force_history_frames are mutually exclusive")
        record = self.records[index]
        rng = np.random.default_rng(augmentation_seed)
        features = torch.from_numpy(np.load(record["feature_path"])).float()
        if len(features) >= self.frames:
            start = int(rng.integers(len(features) - self.frames + 1))
            features = features[start:start + self.frames]
            length = self.frames
        else:
            length = len(features)
            pad = features[-1:].expand(self.frames - length, -1)
            features = torch.cat((features, pad), dim=0)
        history_available = length >= 2 * self.horizon
        if force_history_frames is not None:
            if force_history_frames <= 0 or force_history_frames % self.horizon:
                raise ValueError("force_history_frames must be a positive multiple of horizon")
            if force_history_frames + self.horizon > length:
                raise ValueError(
                    f"sample {record['id']} has {length} frames; forced history of "
                    f"{force_history_frames} requires at least "
                    f"{force_history_frames + self.horizon}"
                )
        elif force_history is True and not history_available:
            raise ValueError(
                f"sample {record['id']} has {length} frames; "
                f"forced history requires at least {2 * self.horizon}"
            )
        if force_history_frames is not None:
            history_frames = force_history_frames
        else:
            use_history = (
                force_history
                if force_history is not None
                else history_available and rng.random() < self.history_probability
            )
            history_frames = self.horizon if use_history else 0
        if history_frames:
            features = features.clone()
            center = features[history_frames - 1, [0, 2]].clone()
            features[:, 0] -= center[0]
            features[:, 2] -= center[1]
        normalized = (features - self.mean) / torch.sqrt(self.std.square() + 1e-5)
        if rng.random() < self.control_probability:
            condition = sample_control_condition(
                features,
                skeleton=None,
                rng=rng,
                horizon=self.horizon,
                generation_start=history_frames,
                condition_dropout=self.condition_dropout,
                control=control,
                out_of_horizon=out_of_horizon,
            )
            observed = (condition["observed_motion"] - self.mean) / torch.sqrt(self.std.square() + 1e-5)
            observed = observed * condition["motion_mask"]
        else:
            condition = {
                "observed_motion": torch.zeros_like(features),
                "motion_mask": torch.zeros_like(features, dtype=torch.bool),
                "control": "none", "frame": -1, "out_of_horizon": False, "condition_dropped": False,
            }
            observed = condition["observed_motion"]
        text = (
            self.text_cache_caption[record["id"]]
            if self.text_cache_caption is not None
            else self._caption(record, rng)
        )
        text_dropped = bool(rng.random() < self.text_dropout)
        output = {
            "id": record["id"],
            "motion": normalized,
            "length": torch.tensor(length, dtype=torch.long),
            "observed_motion": observed,
            "motion_mask": condition["motion_mask"].float(),
            "text": "" if text_dropped else text,
            "text_dropped": text_dropped,
            "control": condition["control"],
            "constraint_frame": condition["frame"],
            "out_of_horizon": condition["out_of_horizon"],
            "condition_dropped": condition["condition_dropped"],
            "history_frames": history_frames,
        }
        if self.text_cache is not None:
            cache_row = self.text_cache_index[record["id"]]
            text_feature = torch.from_numpy(np.array(self.text_cache[cache_row], copy=True)).float()
            output["text_feature"] = torch.zeros_like(text_feature) if text_dropped else text_feature
        return output

    def __getitem__(self, index: int) -> dict:
        # Direct indexing stays deterministic for audits and fixed validation.
        # Trainers should call ``sample`` with a generated augmentation seed.
        return self.sample(index, self.seed + index)


def select_cached_caption(sample_id: str, text_root: str | Path, seed: int) -> str:
    """Choose one caption deterministically without depending on data augmentation RNG."""
    path = Path(text_root) / f"{sample_id.removeprefix('M')}.txt"
    captions = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines() if line.strip()]
    if not captions:
        return ""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return captions[int.from_bytes(digest[:8], "little") % len(captions)]


def select_manifest_caption(record: dict, seed: int, text_root: str | Path | None = None) -> str:
    """Select a stable caption from either a manifest or HumanML3D text files."""
    captions = [str(value).strip() for value in record.get("captions", []) if str(value).strip()]
    if not captions:
        if text_root is None:
            return ""
        return select_cached_caption(record["id"], text_root, seed)
    digest = hashlib.sha256(f"{seed}:{record['id']}".encode()).digest()
    return captions[int.from_bytes(digest[:8], "little") % len(captions)]


def _rows(index_path: Path, root: Path) -> Iterable[HumanML3DRow]:
    with index_path.open(newline="") as handle:
        for item in csv.DictReader(handle):
            source = Path(item["source_path"])
            if not source.is_absolute():
                source = root / source
            dataset = source.parts[source.parts.index("pose_data") + 1] if "pose_data" in source.parts else "unknown"
            # The CSV stores ``000123.npy`` while split manifests use ``000123``.
            # Keep the adapter's public sample IDs extension-free.
            yield HumanML3DRow(source, Path(item["new_name"]).stem, int(item["start_frame"]), int(item["end_frame"]), dataset)


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(axis.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(axis.shape[:-1] + (3, 3))
    return eye + torch.sin(angle)[..., None] * k + (1 - torch.cos(angle))[..., None] * (k @ k)


def _global_from_local(local: torch.Tensor, parents: tuple[int, ...]) -> torch.Tensor:
    out = local.clone()
    for joint in range(1, len(parents)):
        out[:, joint] = out[:, parents[joint]] @ local[:, joint]
    return out


def _local_from_global(global_rots: torch.Tensor, parents: tuple[int, ...]) -> torch.Tensor:
    out = global_rots.clone()
    for joint in range(1, len(parents)):
        out[:, joint] = global_rots[:, parents[joint]].transpose(-1, -2) @ global_rots[:, joint]
    return out


def _rotation_x(angle: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=dtype, device=device)


def _core_skeleton(folder: Optional[str] = None):
    from ardy.skeleton import CoreSkeleton27

    return CoreSkeleton27(folder=folder, load=True)


class HumanML3DConverter:
    """Convert one HumanML3D metadata row from native AMASS rotations."""

    def __init__(self, dataset_root: str | Path, skeleton_folder: Optional[str] = None, fps: int = 20):
        self.root = Path(dataset_root)
        self.fps = fps
        self.skeleton = _core_skeleton(skeleton_folder)
        self.target_neutral = self.skeleton.neutral_joints.detach().float()
        self._body_models: dict[str, object] = {}

    def raw_path(self, row: HumanML3DRow) -> Path:
        source = row.source_path
        parts = list(source.parts)
        try:
            parts[parts.index("pose_data")] = "amass_data"
        except ValueError:
            raise ValueError(f"source_path is not under pose_data: {source}")
        raw = Path(*parts)
        if raw.suffix != ".npz":
            raw = raw.with_suffix(".npz")
        if not raw.exists():
            raise FileNotFoundError(f"AMASS source missing for {source}: {raw}")
        return raw

    @staticmethod
    def _dataset_trim(dataset: str, count: int, fps: int) -> int:
        seconds = {
            "Eyes_Japan_Dataset": 3.0, "MPI_HDM05": 3.0, "TotalCapture": 1.0,
            "MPI_Limits": 1.0, "Transitions_mocap": 0.5,
        }.get(dataset, 0.0)
        return min(count, int(seconds * fps))

    def _body_model(self, gender: str):
        gender = "male" if str(gender).lower().startswith("m") else "female"
        if gender not in self._body_models:
            dataset_path = str(self.root)
            if dataset_path not in sys.path:
                sys.path.insert(0, dataset_path)
            from human_body_prior.body_model.body_model import BodyModel

            model = BodyModel(
                bm_fname=str(self.root / "body_models" / "smplh" / gender / "model.npz"),
                num_betas=10,
                num_dmpls=8,
                dmpl_fname=str(self.root / "body_models" / "dmpls" / gender / "model.npz"),
            ).eval()
            self._body_models[gender] = model
        return self._body_models[gender]

    def _shape_scale(self, data: np.lib.npyio.NpzFile) -> float:
        """Scale source leg length to Core's target leg length."""
        try:
            model = self._body_model(str(data["gender"]))
            betas = torch.as_tensor(data["betas"][:10], dtype=torch.float32)[None]
            zeros = torch.zeros(1, 3)
            with torch.no_grad():
                body = model(root_orient=zeros, pose_body=torch.zeros(1, 63), pose_hand=torch.zeros(1, 90), trans=zeros, betas=betas)
            joints = body.Jtr[0].float()
            source_len = sum(float((joints[a] - joints[b]).norm()) for a, b in ((1, 0), (4, 1), (7, 4)))
            target_len = sum(float((self.target_neutral[a] - self.target_neutral[b]).norm()) for a, b in ((23, 0), (24, 23), (25, 24)))
            if source_len > 1e-6:
                return target_len / source_len
        except (FileNotFoundError, KeyError, RuntimeError, ImportError):
            pass
        return 1.0

    def _load_native(self, row: HumanML3DRow) -> tuple[torch.Tensor, torch.Tensor, float]:
        data = np.load(self.raw_path(row), allow_pickle=True)
        mocap_fps = float(np.asarray(data["mocap_framerate"]).reshape(()))
        step = int(mocap_fps / self.fps)
        if step < 1:
            raise ValueError(f"Cannot downsample {mocap_fps} FPS to {self.fps} FPS")
        poses = torch.as_tensor(np.asarray(data["poses"])[::step, :66], dtype=torch.float32).reshape(-1, 22, 3)
        trans = torch.as_tensor(np.asarray(data["trans"])[::step], dtype=torch.float32)
        trim = self._dataset_trim(row.dataset, len(poses), self.fps)
        poses, trans = poses[trim:], trans[trim:]
        end = len(poses) if row.end_frame < 0 else row.end_frame
        poses, trans = poses[row.start_frame:end], trans[row.start_frame:end]
        if len(poses) < 2:
            raise ValueError(f"row {row.new_name} has fewer than two frames after crop")
        return poses, trans, self._shape_scale(data)

    def convert_row(self, row: HumanML3DRow, mirror: bool = False, canonicalize: bool = True) -> dict:
        if row.is_humanact12:
            raise ValueError("HumanAct12 is excluded: it has no native AMASS rotations")
        poses, trans, scale = self._load_native(row)
        local = _axis_angle_to_matrix(poses)
        source_global = _global_from_local(local, SMPL_PARENTS)
        c = AMASS_TO_CORE.to(dtype=source_global.dtype)
        transformed = c @ source_global @ c.transpose(-1, -2)
        # AMASS's body-model rest pose is y-up; its world recordings are z-up.
        # Rx(+90) maps the rest pose into that world frame before basis conversion.
        ref = c @ _rotation_x(math.pi / 2, transformed.dtype, transformed.device) @ c.transpose(-1, -2)
        relative = transformed @ ref.transpose(-1, -2)

        target_global = torch.eye(3, dtype=relative.dtype).repeat(len(relative), 27, 1, 1)
        mapped_targets = set(SMPL22_TO_CORE.values())
        for source_idx, target_idx in SMPL22_TO_CORE.items():
            target_global[:, target_idx] = relative[:, source_idx]
        # Extra target joints have no SMPL-H counterpart.  Keeping the parent's
        # global rotation makes their local rotation identity in the rest pose.
        for target_idx in range(27):
            if target_idx not in mapped_targets:
                target_global[:, target_idx] = target_global[:, CORE_PARENTS[target_idx]]

        root = (trans @ c.transpose(0, 1))
        root = (root - root[:1]) * scale
        if mirror:
            reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=root.dtype))
            swaps = {7: 13, 8: 14, 9: 15, 10: 16, 11: 17, 12: 18, 19: 23, 20: 24, 21: 25, 22: 26}
            mirrored = target_global.clone()
            for left, right in swaps.items():
                mirrored[:, left] = reflection @ target_global[:, right] @ reflection
                mirrored[:, right] = reflection @ target_global[:, left] @ reflection
            for idx in (0, 1, 2, 3, 4, 5, 6):
                mirrored[:, idx] = reflection @ target_global[:, idx] @ reflection
            target_global = mirrored
            root = root @ reflection.transpose(0, 1)

        local_core = _local_from_global(target_global, CORE_PARENTS)
        _, posed, _ = self.skeleton.fk(local_core, root)
        feet = posed[:, [21, 22, 25, 26], 1]
        root[:, 1] += -float(feet.min())
        _, posed, _ = self.skeleton.fk(local_core, root)

        # Import lazily so this module can still be used for metadata inspection.
        from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep

        rep = ArdyMotionRep(self.skeleton, fps=self.fps)
        features = rep(local_core[None], root[None], to_normalize=False, to_canonicalize=canonicalize)[0]
        return {
            "features": features.contiguous(),
            "local_rot_mats": local_core.contiguous(),
            "root_positions": root.contiguous(),
            "posed_joints": posed.contiguous(),
            "scale": scale,
            "source_path": str(self.raw_path(row)),
            "new_name": ("M" + row.new_name) if mirror else row.new_name,
            "mirror": mirror,
            "fps": self.fps,
        }


def sample_control_condition(
    features: torch.Tensor,
    skeleton,
    rng: Optional[np.random.Generator] = None,
    horizon: int = 40,
    control: Optional[str] = None,
    out_of_horizon: Optional[bool] = None,
    condition_dropout: float = 0.0,
    generation_start: int = 0,
) -> dict[str, torch.Tensor | str | int | bool]:
    """Sample ARDY-style observed motion and feature masks from GT features.

    The returned sequence may contain future-mask entries beyond ``horizon``;
    ARDY's ``future_token_mask`` then exposes out-of-horizon constraints to the
    autoregressive denoiser.  This is a data contract, independent of model
    architecture, and is suitable for both Minimal and Hybrid trainers.
    """
    if features.ndim != 2 or (features.shape[-1] - 6) % 12:
        raise ValueError(f"expected an ARDY [T,12J+6] feature tensor, got {tuple(features.shape)}")
    joints = (features.shape[-1] - 6) // 12
    rng = rng or np.random.default_rng()
    t = features.shape[0]
    controls = ("root_trajectory", "root_waypoint", "full_keyframe", "joint_position", "joint_rotation")
    control = control or controls[int(rng.integers(len(controls)))]
    if control not in controls:
        raise ValueError(f"unknown control {control!r}")
    use_out = bool(rng.integers(2)) if out_of_horizon is None else bool(out_of_horizon)
    generation_end = generation_start + horizon
    use_out = use_out and t > generation_end + 1
    if use_out:
        low = generation_end + 1
        frame = int(rng.integers(low, t))
    else:
        low = min(generation_start, t - 1)
        high = min(generation_end, t)
        frame = int(rng.integers(low, max(low + 1, high)))
    observed = torch.zeros_like(features)
    mask = torch.zeros_like(features, dtype=torch.bool)
    root = slice(0, 3)
    heading = slice(3, 5)
    local_end = 5 + (joints - 1) * 3
    rotations_end = local_end + joints * 6
    local = slice(5, local_end)
    rotations = slice(local_end, rotations_end)
    if control == "root_trajectory":
        start = generation_start if not use_out else max(generation_end, frame - horizon)
        mask[start:frame + 1, 0] = True
        mask[start:frame + 1, 2] = True
    elif control == "root_waypoint":
        mask[frame, 0] = True
        mask[frame, 2] = True
    elif control == "full_keyframe":
        mask[frame, root] = True
        mask[frame, heading] = True
        mask[frame, local] = True
        mask[frame, rotations] = True
    elif control == "joint_position":
        joint = int(rng.integers(1, joints))
        # ARDY end-effector position constraints carry root xyz and heading so
        # the relative joint position has an unambiguous world-space frame.
        mask[frame, 0:3] = True
        mask[frame, heading] = True
        mask[frame, 5 + (joint - 1) * 3:5 + joint * 3] = True
    else:
        joint = int(rng.integers(0, joints))
        mask[frame, local_end + joint * 6:local_end + (joint + 1) * 6] = True
    observed[mask] = features[mask]
    dropped = bool(rng.random() < condition_dropout)
    if dropped:
        observed.zero_()
        mask.zero_()
    return {
        "observed_motion": observed,
        "motion_mask": mask,
        "control": control,
        "frame": frame,
        "out_of_horizon": use_out,
        "condition_dropped": dropped,
    }


@torch.no_grad()
def validate_ardy_features(features: torch.Tensor, skeleton, fps: int = 20) -> dict[str, float | bool]:
    """Run representation round-trip and target-skeleton geometry checks."""
    from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep

    rep = ArdyMotionRep(skeleton, fps=fps)
    inverse = rep.inverse(features[None], is_normalized=False)
    local = inverse["local_rot_mats"]
    root = inverse["root_positions"]
    posed = inverse["posed_joints"]
    reencoded = rep(local, root, to_normalize=False, to_canonicalize=False)[0]
    rotations = inverse["global_rot_mats"][0]
    identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    orthogonal = rotations.transpose(-1, -2) @ rotations
    parents = skeleton.joint_parents
    neutral = skeleton.neutral_joints.to(posed)
    expected = (neutral[1:] - neutral[parents[1:]]).norm(dim=-1)
    actual = (posed[0, :, 1:] - posed[0, :, parents[1:]]).norm(dim=-1)
    root_speed = (root[:, 1:] - root[:, :-1]).norm(dim=-1) * fps
    heading = features[:, 3:5]
    feet = posed[0, :, list(skeleton.foot_joint_idx), 1]
    return {
        "finite": bool(torch.isfinite(features).all()),
        "roundtrip_max_abs": float((reencoded[:, :-4] - features[:, :-4]).abs().max()),
        "contact_disagreement_fraction": float((reencoded[:, -4:] != features[:, -4:]).float().mean()),
        "rotation_orthogonality_max_abs": float((orthogonal - identity).abs().max()),
        "rotation_det_max_abs": float((torch.linalg.det(rotations) - 1).abs().max()),
        "bone_length_max_abs": float((actual - expected).abs().max()),
        "heading_unit_max_abs": float((heading.norm(dim=-1) - 1).abs().max()),
        "minimum_foot_height": float(feet.min()),
        "maximum_root_speed": float(root_speed.max()) if root_speed.numel() else 0.0,
    }


def write_manifest(records: list[dict], path: str | Path) -> None:
    Path(path).write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")


def long_horizon_v1_window_range(draw: float) -> tuple[int, int]:
    """Map a unit draw to the fixed protocol-aligned rollout curriculum."""
    if not 0.0 <= draw < 1.0:
        raise ValueError("curriculum draw must be in [0, 1)")
    if draw < 0.50:
        return 2, 2
    if draw < 0.70:
        return 4, 4
    if draw < 0.85:
        return 6, 10
    return 12, 19
