"""Convert ARDY CoreSkeleton27 geometry to the official HumanML3D 263-D form."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from ardy_data.humanml3d import SMPL22_TO_CORE


DEFAULT_MDM_EVAL_ROOT = Path("/data3/zts/packages/motion-diffusion-model-eval")
DEFAULT_HUMANML3D_ROOT = Path("/data3/zts/datasets/HumanML3D/HumanML3D")


def core27_to_smpl22_positions(core_joints: np.ndarray) -> np.ndarray:
    """Select the exact SMPL22 correspondences used by the forward retargeter."""
    values = np.asarray(core_joints, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (27, 3):
        raise ValueError(f"expected [T,27,3] Core joints, got {values.shape}")
    output = np.empty((len(values), 22, 3), dtype=np.float32)
    for smpl_joint, core_joint in SMPL22_TO_CORE.items():
        output[:, smpl_joint] = values[:, core_joint]
    return output


def core27_to_humanml263(
    core_joints: np.ndarray,
    *,
    mdm_eval_root: str | Path = DEFAULT_MDM_EVAL_ROOT,
    humanml3d_root: str | Path = DEFAULT_HUMANML3D_ROOT,
) -> tuple[np.ndarray, np.ndarray]:
    """Run HumanML3D's released ``process_file`` on mapped Core27 joints.

    ``process_file`` emits one fewer feature frame than position frames. The
    final pose is repeated once so a T-frame generated motion remains T frames
    in the evaluator representation.
    """
    mdm_root = Path(mdm_eval_root)
    if str(mdm_root) not in sys.path:
        sys.path.insert(0, str(mdm_root))
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.scripts import motion_process
    from data_loaders.humanml.utils.paramUtil import (
        t2m_kinematic_chain,
        t2m_raw_offsets,
    )

    # The released evaluator code predates NumPy 1.20 and uses ``np.float``.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    # ``motion_process.py`` defines these only inside its dataset-conversion
    # __main__ block. Set the same HumanML3D globals explicitly for library use.
    motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    motion_process.kinematic_chain = t2m_kinematic_chain
    motion_process.face_joint_indx = [2, 1, 17, 16]
    motion_process.fid_r, motion_process.fid_l = [8, 11], [7, 10]
    motion_process.l_idx1, motion_process.l_idx2 = 5, 8

    target_path = Path(humanml3d_root) / "new_joints" / "000021.npy"
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    target_positions = np.load(target_path).astype(np.float32)
    target_skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
    motion_process.tgt_offsets = target_skeleton.get_offsets_joints(
        torch.from_numpy(target_positions[0])
    )
    smpl22 = core27_to_smpl22_positions(core_joints)
    padded = np.concatenate((smpl22, smpl22[-1:]), axis=0)
    features, global_positions, _, _ = motion_process.process_file(padded, 0.002)
    if features.shape != (len(core_joints), 263):
        raise RuntimeError(f"unexpected HumanML3D feature shape: {features.shape}")
    return features.astype(np.float32), global_positions[:-1].astype(np.float32)
