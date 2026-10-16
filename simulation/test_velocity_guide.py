"""Read-only guide matches the official velocity-transition integration."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ardy'))
from scripts.interactive_demo.velocity_guide import velocity_path


class VelocityGuideTests(unittest.TestCase):
    def test_path_matches_official_loop_and_preserves_inputs(self):
        root = np.array([3., .8, 4.])
        velocity = np.array([0., 0., .7])
        target = np.array([.8, 0., 0.])
        future = []
        position = root.copy()
        for i in range(50):
            alpha = (i+1)/50
            position = position + ((1-alpha)*velocity+alpha*target)/25
            future.append(position.copy())
        expected = np.array(future)[9::10]
        expected[:, 1] = 0.
        np.testing.assert_allclose(velocity_path(root, velocity, target, 25), expected, atol=1e-12)
        np.testing.assert_array_equal(root, [3., .8, 4.])
        np.testing.assert_array_equal(target, [.8, 0., 0.])
