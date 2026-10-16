"""Metadata-backed HiPHI dataset adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from ardy_data.hiphi_archive import discover_metadata_table
from ardy_data.parsers.hiphi import parse_hiphi_bvh
from ardy_data.schema import NativeMotion


@dataclass(frozen=True)
class HiPHIRecord:
    motion_id: str
    frame: str
    lu: str
    frame_lu: str
    duration_sec: float
    frame_count: int
    actor_id: str
    text_annotation: str
    is_hoi: bool
    mirrored: bool
    archive_name: str
    actor_height_cm: float | None = None
    actor_weight_kg: float | None = None
    actor_gender: str | None = None

    @property
    def source_parent_uid(self) -> str:
        return self.motion_id.removesuffix("__mirror")

    @property
    def source_group_uid(self) -> str:
        return f"actor:{self.actor_id}"

    @property
    def mirror_of(self) -> str | None:
        return self.source_parent_uid if self.mirrored else None


class HiPHIAdapter:
    def __init__(self, hiphi_root: str | Path):
        self.root = Path(hiphi_root)
        metadata = pd.read_csv(discover_metadata_table(self.root))
        actor_tables = []
        for path in sorted((self.root / "metadata").glob("*.csv")):
            columns = set(pd.read_csv(path, nrows=0).columns)
            if {"actor_id", "height_cm", "weight_kg", "gender"} <= columns:
                actor_tables.append(path)
        if len(actor_tables) != 1:
            raise ValueError(f"expected one HiPHI actor table, found {actor_tables}")
        actors = pd.read_csv(actor_tables[0]).set_index("actor_id").to_dict("index")
        parts = pd.read_csv(self.root / "data" / "motion_to_part.csv")
        archive_by_motion = {}
        for row in parts.to_dict("records"):
            archive_by_motion[str(row["motion_id"])] = str(row["archive_name"])
            archive_by_motion[str(row["mirrored_motion_id"])] = str(row["archive_name"])
        self.records = []
        for row in metadata.to_dict("records"):
            motion_id = str(row["motion_id"])
            if motion_id not in archive_by_motion:
                raise ValueError(f"motion {motion_id} is absent from motion_to_part.csv")
            actor_id = str(row["actor_id"])
            actor = actors.get(actor_id, {})
            self.records.append(HiPHIRecord(
                motion_id=motion_id,
                frame=str(row["frame"]),
                lu=str(row["lu"]),
                frame_lu=str(row["frame_lu"]),
                duration_sec=float(row["duration_sec"]),
                frame_count=int(row["frame_count"]),
                actor_id=actor_id,
                text_annotation="" if pd.isna(row["text_annotation"]) else str(row["text_annotation"]).strip(),
                is_hoi=bool(row["is_hoi"]),
                mirrored=bool(row["mirrored"]),
                archive_name=archive_by_motion[motion_id],
                actor_height_cm=float(actor["height_cm"]) if "height_cm" in actor else None,
                actor_weight_kg=float(actor["weight_kg"]) if "weight_kg" in actor else None,
                actor_gender=str(actor["gender"]) if "gender" in actor else None,
            ))

    def iter_records(self, *, body_only: bool = False, include_hoi: bool = False) -> Iterator[HiPHIRecord]:
        if body_only and include_hoi:
            raise ValueError("body_only and include_hoi are mutually exclusive")
        for record in self.records:
            if body_only and record.is_hoi:
                continue
            if not include_hoi and not body_only and record.is_hoi:
                continue
            yield record

    def bvh_path(self, record: HiPHIRecord) -> Path:
        return self.root / "data" / record.frame / record.lu / record.motion_id / "motion_actor.bvh"

    def parse(self, record: HiPHIRecord) -> NativeMotion:
        bvh_path = self.bvh_path(record)
        package_path = bvh_path.parent / "metadata.json"
        package = json.loads(package_path.read_text())
        expected = {
            "motion_id": record.motion_id,
            "frame": record.frame,
            "lu": record.lu,
            "frame_lu": record.frame_lu,
            "frame_count": record.frame_count,
            "actor_id": record.actor_id,
            "text_annotation": record.text_annotation,
            "is_hoi": record.is_hoi,
            "mirrored": record.mirrored,
        }
        mismatches = {
            field: {"global": value, "package": package.get(field)}
            for field, value in expected.items()
            if package.get(field) != value
        }
        if mismatches:
            raise ValueError(f"global/package metadata mismatch for {record.motion_id}: {mismatches}")
        metadata = {
            "motion_id": record.motion_id,
            "actor_id": record.actor_id,
            "frame": record.frame,
            "lu": record.lu,
            "frame_lu": record.frame_lu,
            "text_annotation": record.text_annotation,
            "is_hoi": record.is_hoi,
            "mirrored": record.mirrored,
            "source_parent_uid": record.source_parent_uid,
            "archive_name": record.archive_name,
            "actor_height_cm": record.actor_height_cm,
            "actor_weight_kg": record.actor_weight_kg,
            "actor_gender": record.actor_gender,
            "package_metadata_path": str(package_path),
            "package_metadata": package,
        }
        motion = parse_hiphi_bvh(bvh_path, metadata)
        package_fps = float(package["fps"])
        if abs(package_fps - motion.fps) > 1e-3:
            raise ValueError(f"package FPS {package_fps} != BVH FPS {motion.fps} for {record.motion_id}")
        return motion
