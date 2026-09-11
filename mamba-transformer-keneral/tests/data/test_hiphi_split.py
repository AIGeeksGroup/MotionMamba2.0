from ardy_data.hiphi import actor_split_map, split_audit


def record(actor, family, split):
    return {
        "actor_id": actor,
        "source_parent_uid": family,
        "split": split,
        "frames": 20,
        "fps": 20,
        "is_hoi": False,
        "is_mirror": False,
    }


def test_actor_split_is_deterministic_and_roughly_80_10_10():
    actors = [f"A{index:03d}" for index in range(100)]
    first = actor_split_map(actors, 7)
    assert first == actor_split_map(list(reversed(actors)), 7)
    assert list(first.values()).count("train") == 80
    assert list(first.values()).count("val") == 10
    assert list(first.values()).count("test") == 10


def test_split_audit_detects_pairwise_actor_leakage():
    audit = split_audit([record("A001", "m1", "train"), record("A001", "m2", "val")])
    assert audit["status"] == "FAIL"
    assert audit["actor_leakage"] == ["A001"]
