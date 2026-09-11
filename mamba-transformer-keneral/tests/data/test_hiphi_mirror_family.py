from ardy_data.adapters.hiphi import HiPHIRecord
from ardy_data.hiphi import split_audit


def make_record(motion_id, mirrored):
    return HiPHIRecord(
        motion_id=motion_id,
        frame="Motion",
        lu="move",
        frame_lu="Motion-move",
        duration_sec=1.0,
        frame_count=90,
        actor_id="A001",
        text_annotation="Move.",
        is_hoi=False,
        mirrored=mirrored,
        archive_name="part.tar.zst",
    )


def test_original_and_mirror_share_family():
    original = make_record("Motion-move_0001", False)
    mirror = make_record("Motion-move_0001__mirror", True)
    assert original.source_parent_uid == mirror.source_parent_uid
    assert mirror.mirror_of == original.motion_id


def test_split_audit_detects_mirror_family_leakage():
    base = {"actor_id": "A001", "frames": 20, "fps": 20, "is_hoi": False}
    audit = split_audit([
        {**base, "source_parent_uid": "m1", "split": "train", "is_mirror": False},
        {**base, "actor_id": "A002", "source_parent_uid": "m1", "split": "test", "is_mirror": True},
    ])
    assert audit["status"] == "FAIL"
    assert audit["family_leakage"] == ["m1"]
