"""HiPHI to CoreSkeleton27/ARDY conversion, QA, and split contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ardy_data.adapters.hiphi import HiPHIAdapter, HiPHIRecord
from ardy_data.humanml3d import validate_ardy_features
from ardy_data.resample import resample_native_motion
from ardy_data.retarget.core27 import retarget_hiphi_to_core27


GEOMETRY_LIMITS = {
    "roundtrip_max_abs": 2e-4,
    "contact_disagreement_fraction": 0.01,
    "rotation_orthogonality_max_abs": 1e-4,
    "rotation_det_max_abs": 1e-4,
    "bone_length_max_abs": 1e-4,
    "heading_unit_max_abs": 1e-4,
    "minimum_foot_height": -1e-4,
}


def actor_split_map(actor_ids: list[str] | set[str], seed: int = 20260907) -> dict[str, str]:
    actors = sorted(set(actor_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)
    train_end = int(round(0.8 * len(actors)))
    val_end = train_end + int(round(0.1 * len(actors)))
    return {
        actor: "train" if index < train_end else "val" if index < val_end else "test"
        for index, actor in enumerate(actors)
    }


def split_audit(records: list[dict]) -> dict:
    actors_by_split = {split: set() for split in ("train", "val", "test")}
    families_by_split = {split: set() for split in ("train", "val", "test")}
    for record in records:
        split = record["split"]
        actors_by_split[split].add(record["actor_id"])
        families_by_split[split].add(record["source_parent_uid"])
    actor_pairs = []
    family_pairs = []
    names = tuple(families_by_split)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            actor_pairs.extend(sorted(actors_by_split[left] & actors_by_split[right]))
            family_pairs.extend(sorted(families_by_split[left] & families_by_split[right]))
    per_split = {}
    for split in names:
        selected = [record for record in records if record["split"] == split]
        per_split[split] = {
            "actors": len(actors_by_split[split]),
            "motions": len(selected),
            "hours": sum(record["frames"] / record["fps"] for record in selected) / 3600.0,
            "body_only_hours": sum(record["frames"] / record["fps"] for record in selected if not record["is_hoi"]) / 3600.0,
            "hoi_hours": sum(record["frames"] / record["fps"] for record in selected if record["is_hoi"]) / 3600.0,
            "mirror_count": sum(bool(record["is_mirror"]) for record in selected),
        }
    return {
        "status": "PASS" if not actor_pairs and not family_pairs else "FAIL",
        "per_split": per_split,
        "actor_leakage": sorted(set(actor_pairs)),
        "actor_leakage_count": len(set(actor_pairs)),
        "family_leakage": sorted(set(family_pairs)),
        "family_leakage_count": len(set(family_pairs)),
    }


def _percentiles(values: torch.Tensor, names=(50, 95, 99)) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if not flat.numel():
        return {f"p{name}": 0.0 for name in names}
    return {f"p{name}": float(torch.quantile(flat, name / 100.0)) for name in names}


def kinematic_qa(features: torch.Tensor, local: torch.Tensor, root: torch.Tensor, skeleton, fps: float) -> dict:
    _, joints, _ = skeleton.fk(local, root)
    root_velocity = torch.diff(root, dim=0) * fps
    root_accel = torch.diff(root_velocity, dim=0) * fps
    joint_velocity = torch.diff(joints, dim=0) * fps
    joint_accel = torch.diff(joint_velocity, dim=0) * fps
    joint_jerk = torch.diff(joint_accel, dim=0) * fps
    root_speed = root_velocity.norm(dim=-1)
    joint_speed = joint_velocity.norm(dim=-1)
    joint_accel_norm = joint_accel.norm(dim=-1)
    joint_jerk_norm = joint_jerk.norm(dim=-1)
    contacts = features[:, -4:] > 0.5
    foot_indices = list(skeleton.foot_joint_idx)
    planar_foot_speed = torch.diff(joints[:, foot_indices][:, :, [0, 2]], dim=0).norm(dim=-1) * fps

    def slip(indices: slice) -> dict[str, float]:
        mask = contacts[1:, indices]
        values = planar_foot_speed[:, indices][mask]
        return _percentiles(values, names=(50, 95))

    heading = torch.atan2(features[:, 4], features[:, 3]).detach().cpu().numpy()
    heading_speed = np.abs(np.diff(np.unwrap(heading))) * fps
    switches = (contacts[1:] != contacts[:-1]).float()
    return {
        "root_speed": {**_percentiles(root_speed), "max": float(root_speed.max()) if root_speed.numel() else 0.0},
        "root_accel": _percentiles(root_accel.norm(dim=-1), names=(95, 99)),
        "joint_speed": _percentiles(joint_speed, names=(95, 99)),
        "joint_accel": _percentiles(joint_accel_norm, names=(95, 99)),
        "joint_jerk": _percentiles(joint_jerk_norm, names=(95, 99)),
        "foot_contact_ratio": float(contacts.float().mean()),
        "foot_slip_left": slip(slice(0, 2)),
        "foot_slip_right": slip(slice(2, 4)),
        "penetration_depth_min": float(joints[:, foot_indices, 1].min()),
        "contact_switch_rate": float(switches.mean()) if switches.numel() else 0.0,
        "heading_angular_speed": {
            "p95": float(np.quantile(heading_speed, 0.95)) if len(heading_speed) else 0.0,
            "p99": float(np.quantile(heading_speed, 0.99)) if len(heading_speed) else 0.0,
        },
        "root_displacement_m": float((root[-1] - root[0]).norm()),
        "root_planar_extent_m": float((root[:, [0, 2]].max(dim=0).values - root[:, [0, 2]].min(dim=0).values).norm()),
    }


def geometry_status(qa: dict) -> str:
    if not qa["finite"]:
        return "fail"
    for field, limit in GEOMETRY_LIMITS.items():
        if field == "minimum_foot_height":
            if qa[field] < limit:
                return "fail"
        elif qa[field] > limit:
            return "fail"
    return "pass"


class HiPHIConverter:
    def __init__(self, adapter: HiPHIAdapter, target_fps: float = 20.0):
        from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
        from ardy.skeleton import CoreSkeleton27

        self.adapter = adapter
        self.target_fps = float(target_fps)
        self.skeleton = CoreSkeleton27(load=True)
        self.motion_rep = ArdyMotionRep(self.skeleton, fps=self.target_fps)

    def convert(self, record: HiPHIRecord) -> dict:
        native = self.adapter.parse(record)
        if native.local_rot_mats.shape[0] != record.frame_count:
            raise ValueError(
                f"metadata frame_count={record.frame_count} but BVH has {len(native.local_rot_mats)}"
            )
        sampled = resample_native_motion(native, self.target_fps)
        local, root, mapping = retarget_hiphi_to_core27(sampled, self.skeleton)
        _, posed, _ = self.skeleton.fk(local, root)
        feet = posed[:, list(self.skeleton.foot_joint_idx), 1]
        root = root.clone()
        root[:, 1] -= feet.min()
        features = self.motion_rep(
            local[None], root[None], to_normalize=False, to_canonicalize=True
        )[0].contiguous()
        if features.shape[1] != 330:
            raise RuntimeError(f"expected ARDY 330-D, got {tuple(features.shape)}")
        canonical = self.motion_rep.inverse(features[None], is_normalized=False)
        canonical_local = canonical["local_rot_mats"][0].contiguous()
        canonical_root = canonical["root_positions"][0].contiguous()
        geometry = validate_ardy_features(features, self.skeleton, int(round(self.target_fps)))
        kinematics = kinematic_qa(
            features, canonical_local, canonical_root, self.skeleton, self.target_fps
        )
        kinematics["ground_offset_m"] = -float(feet.min())
        source_contract = dict(native.metadata["bvh_contract"])
        if record.actor_height_cm:
            source_contract["actor_height_m"] = record.actor_height_cm * 0.01
            source_contract["skeleton_to_actor_height_ratio"] = (
                source_contract["source_skeleton_height_m"] / source_contract["actor_height_m"]
            )
        return {
            "features": features,
            "local_rot_mats": canonical_local,
            "root_positions": canonical_root,
            "geometry_qa": geometry,
            "kinematic_qa": kinematics,
            "status": geometry_status(geometry),
            "mapping": mapping,
            "source_contract": source_contract,
            "source_fps": native.fps,
            "fps": self.target_fps,
        }


def cache_key(record: HiPHIRecord, source: Path, target_fps: float, git_commit: str) -> str:
    stat = source.stat()
    payload = {
        "source_path": str(source.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "target_fps": target_fps,
        "mapping_version": "hiphi_core27_global_v1",
        "config": {"ground": "global_min_feet_v1", "heading": "ardy_canonical_v1"},
        "git_commit": git_commit,
        "motion_id": record.motion_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def output_paths(output: str | Path, motion_id: str) -> dict[str, Path]:
    root = Path(output)
    shard = hashlib.sha256(motion_id.encode()).hexdigest()[:2]
    safe_name = motion_id.replace("/", "_")
    return {
        "feature": root / "features" / shard / f"{safe_name}.npy",
        "rotation": root / "rotations" / shard / f"{safe_name}.npy",
        "root": root / "roots" / shard / f"{safe_name}.npy",
        "item": root / "manifests" / "items" / shard / f"{safe_name}.json",
        "qa": root / "qa" / shard / f"{safe_name}.json",
    }


def save_conversion(
    output: str | Path,
    record: HiPHIRecord,
    result: dict,
    split: str,
    key: str,
) -> dict:
    paths = output_paths(output, record.motion_id)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["feature"], result["features"].numpy().astype(np.float32))
    np.save(paths["rotation"], result["local_rot_mats"].numpy().astype(np.float32))
    np.save(paths["root"], result["root_positions"].numpy().astype(np.float32))
    qa = {
        "geometry": result["geometry_qa"],
        "kinematic": result["kinematic_qa"],
        "source_contract": result["source_contract"],
        "mapping": result["mapping"],
    }
    paths["qa"].write_text(json.dumps(qa, indent=2) + "\n")
    manifest = {
        "id": f"hiphi:{record.motion_id}",
        "sample_uid": f"hiphi:{record.motion_id}",
        "source_dataset": "hiphi",
        "source_id": record.motion_id,
        "source_parent_uid": record.source_parent_uid,
        "source_group_uid": record.source_group_uid,
        "source_archive": record.archive_name,
        "source_path": str(result.get("source_path", "")),
        "feature_path": str(paths["feature"]),
        "rotation_path": str(paths["rotation"]),
        "root_path": str(paths["root"]),
        "frames": int(len(result["features"])),
        "fps": result["fps"],
        "source_fps": result["source_fps"],
        "text": record.text_annotation,
        "captions": [record.text_annotation] if record.text_annotation else [],
        "text_quality": "natural_instruction" if record.text_annotation else "missing",
        "text_segments": [{
            "text": record.text_annotation,
            "source": "hiphi",
            "quality": "natural_instruction" if record.text_annotation else "missing",
            "start_sec": 0.0,
            "end_sec": record.duration_sec,
        }],
        "actor_id": record.actor_id,
        "actor_metadata": {
            "height_cm": record.actor_height_cm,
            "weight_kg": record.actor_weight_kg,
            "gender": record.actor_gender,
        },
        "is_mirror": record.mirrored,
        "mirror_of": record.mirror_of,
        "is_hoi": record.is_hoi,
        "frame": record.frame,
        "lu": record.lu,
        "geometry_qa": result["geometry_qa"],
        "kinematic_qa": result["kinematic_qa"],
        "status": result["status"],
        "split": split,
        "cache_key": key,
        "license_tag": "modalitynet-open-research-license",
        "metadata_provenance": {
            "global": "metadata/hiphi_metadata.csv",
            "package": f"data/{record.frame}/{record.lu}/{record.motion_id}/metadata.json",
        },
        "source_relative_path": f"data/{record.frame}/{record.lu}/{record.motion_id}/motion_actor.bvh",
    }
    paths["item"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
