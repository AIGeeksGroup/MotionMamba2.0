import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from ardy_data.resample import resample_rotations_and_root


@pytest.mark.parametrize("target_fps", [20.0, 25.0])
def test_rotation_aware_resample_constant_velocities(target_fps):
    source_fps = 90.0
    frames = 91
    time = np.arange(frames) / source_fps
    matrices = Rotation.from_euler("y", (90.0 * time)[:, None], degrees=True).as_matrix()
    rotations = torch.from_numpy(matrices[:, None]).float()
    root = torch.from_numpy(np.stack((2.0 * time, np.ones_like(time), -time), axis=-1)).float()
    sampled_rot, sampled_root = resample_rotations_and_root(rotations, root, source_fps, target_fps)
    target_time = np.arange(len(sampled_root)) / target_fps
    np.testing.assert_allclose(sampled_root[:, 0], 2.0 * target_time, atol=1e-6)
    angles = np.rad2deg(Rotation.from_matrix(sampled_rot[:, 0].numpy()).magnitude())
    np.testing.assert_allclose(angles, 90.0 * target_time, atol=2e-4)
    assert len(sampled_rot) == int(target_fps) + 1
