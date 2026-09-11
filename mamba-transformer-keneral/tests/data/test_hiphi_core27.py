from pathlib import Path

import torch

from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
from ardy.skeleton import CoreSkeleton27
from ardy_data.humanml3d import CORE_JOINTS
from ardy_data.retarget.core27 import HIPHI_TO_CORE27, retarget_hiphi_to_core27
from ardy_data.schema import NativeMotion


def test_hiphi_core27_mapping_is_complete_and_semantic():
    assert set(HIPHI_TO_CORE27) == set(CORE_JOINTS)
    assert len(HIPHI_TO_CORE27) == 27
    assert HIPHI_TO_CORE27["RightHandEnd"] == "RightHandMiddle1"
    assert HIPHI_TO_CORE27["LeftHandEnd"] == "LeftHandMiddle1"
    assert HIPHI_TO_CORE27["Hips"] == "Hips"


def test_retargeted_core27_builds_330d_and_preserves_so3():
    skeleton = CoreSkeleton27(load=True)
    source_names = tuple(HIPHI_TO_CORE27[name] for name in skeleton.bone_order_names)
    frames = 5
    identity = torch.eye(3).repeat(frames, 27, 1, 1)
    motion = NativeMotion(
        joint_names=source_names,
        parents=tuple(int(value) for value in skeleton.joint_parents.tolist()),
        offsets_m=torch.zeros(27, 3),
        neutral_joints_m=skeleton.neutral_joints.detach().clone(),
        local_rot_mats=identity,
        root_positions_m=torch.zeros(frames, 3),
        fps=20.0,
        euler_orders=("ZXY",) * 27,
        channels=tuple(("Zrotation", "Xrotation", "Yrotation") for _ in range(27)),
        source_path=Path("synthetic.bvh"),
    ).validate()
    local, root, provenance = retarget_hiphi_to_core27(motion, skeleton)
    features = ArdyMotionRep(skeleton, fps=20)(
        local[None], root[None], to_normalize=False, to_canonicalize=False
    )[0]
    assert features.shape == (frames, 330)
    assert provenance["mapping_version"] == "hiphi_core27_global_v1"
    orthogonal = local.transpose(-1, -2) @ local
    torch.testing.assert_close(orthogonal, torch.eye(3).expand_as(orthogonal))
    torch.testing.assert_close(torch.linalg.det(local), torch.ones(frames, 27))
