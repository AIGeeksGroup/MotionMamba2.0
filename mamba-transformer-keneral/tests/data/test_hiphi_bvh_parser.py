from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ardy_data.parsers.hiphi import parse_hiphi_bvh


def write_bvh(path):
    path.write_text("""HIERARCHY
ROOT Hips
{
  OFFSET 0 100 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 5 0
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.011111111111111112
0 100 0 0 0 0 0 0 0 0 0 0
50 100 0 45 0 0 0 0 0 10 20 30
100 100 0 90 0 0 0 0 0 20 40 60
""")


def test_bvh_parser_reads_hierarchy_timing_euler_and_units(tmp_path):
    path = tmp_path / "motion_actor.bvh"
    write_bvh(path)
    motion = parse_hiphi_bvh(path)
    assert motion.joint_names == ("Hips", "Spine")
    assert motion.parents == (-1, 0)
    assert motion.euler_orders == ("ZXY", "ZXY")
    assert motion.fps == 90.0
    np.testing.assert_allclose(motion.root_positions_m[:, 0], [0.0, 0.5, 1.0])
    expected = Rotation.from_euler("ZXY", [[10, 20, 30]], degrees=True).as_matrix()[0]
    np.testing.assert_allclose(motion.local_rot_mats[1, 1], expected, atol=1e-6)


def test_bvh_parser_rejects_dynamic_non_root_translation(tmp_path):
    path = tmp_path / "motion_actor.bvh"
    write_bvh(path)
    text = path.read_text().replace("0 0 0 10 20 30", "1 0 0 10 20 30")
    path.write_text(text)
    try:
        parse_hiphi_bvh(path)
    except ValueError as exc:
        assert "non-root translations" in str(exc)
    else:
        raise AssertionError("dynamic non-root translations must not be silently discarded")
