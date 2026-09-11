import pandas as pd

from ardy_data.hiphi_archive import discover_metadata_table, metadata_audit


def test_metadata_discovery_and_audit_use_released_schema(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "data").mkdir()
    rows = pd.DataFrame([
        {"motion_id": "m1", "frame": "F", "lu": "l", "frame_lu": "F-l", "duration_sec": 10.0, "frame_count": 900, "actor_id": "A001", "text_annotation": "Move", "is_hoi": False, "object_categories": "", "mirrored": False},
        {"motion_id": "m1__mirror", "frame": "F", "lu": "l", "frame_lu": "F-l", "duration_sec": 10.0, "frame_count": 900, "actor_id": "A001", "text_annotation": "Move", "is_hoi": False, "object_categories": "", "mirrored": True},
    ])
    rows.to_csv(tmp_path / "metadata" / "unexpected_name.csv", index=False)
    pd.DataFrame([{"archive_name": "a.tar.zst", "package_count": 2, "compressed_bytes": 1, "uncompressed_bytes": 2}]).to_csv(tmp_path / "data" / "archive_manifest.csv", index=False)
    pd.DataFrame([{"motion_id": "m1", "mirrored_motion_id": "m1__mirror", "archive_name": "a.tar.zst", "uncompressed_bytes": 2}]).to_csv(tmp_path / "data" / "motion_to_part.csv", index=False)
    assert discover_metadata_table(tmp_path).name == "unexpected_name.csv"
    audit = metadata_audit(tmp_path)
    assert audit["total_motions"] == 2
    assert audit["original_only_hours"] == 10.0 / 3600.0
    assert audit["text_coverage"] == 1.0
