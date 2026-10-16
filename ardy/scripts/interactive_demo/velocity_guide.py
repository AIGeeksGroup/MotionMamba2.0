"""Read-only keyboard guide using ARDY's original waypoint/path renderer."""
import numpy as np
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from scipy.interpolate import interp1d

from ardy.motion_rep.smooth_root import get_smooth_root_pos
from ardy.viz.viser_utils import RootKeyframe2DSet


def velocity_path(root, velocity, target, fps, duration=2., interval=10):
    """Use the original keyboard guide's two-second linear velocity transition."""
    count = max(1, int(fps * duration))
    alpha = np.arange(1, count + 1, dtype=float)[:, None] / count
    velocities = (1-alpha)*np.asarray(velocity) + alpha*np.asarray(target)
    future = np.asarray(root) + np.cumsum(velocities, axis=0)/fps
    # The official guide starts at its first future waypoint, not at the
    # character's feet. Do not add a sixth dot or an extra connecting segment.
    points = future[interval-1::interval].copy()
    points[:, 1] = 0.
    return points


class VelocityGuide:
    def __init__(self, client, name, skeleton):
        self.client, self.name = client, name
        # This private renderer is never added to session.constraints and
        # therefore cannot introduce goals into the generator.
        self.visual = RootKeyframe2DSet(name, client, skeleton)
        self.visual.set_dense_path(True)
        self.last_frame = None
        self.last_target = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='velocity-guide')
        self._pending = None
        self._working = False
        self._closed = False
        self._visible = False

    def update(self, root, velocity, target, fps, frame, force=False):
        target = tuple(float(v) for v in target)
        with self._lock:
            if self._closed:
                return
            self.set_visibility(True)
            if (not force and self.last_frame is not None
                    and 0 <= frame-self.last_frame < 4 and target == self.last_target):
                return
            self.last_frame, self.last_target = frame, target
            # Keep at most one newest request. Smoothing must not run while
            # holding a scene/pose atomic block or build a queue behind playback.
            self._pending = (np.array(root, copy=True), np.array(velocity, copy=True), target, fps)
            if not self._working:
                self._working = True
                self._executor.submit(self._draw_pending)

    def _draw_pending(self):
        while True:
            with self._lock:
                if self._closed or self._pending is None:
                    self._working = False
                    return
                root, velocity, target, fps = self._pending
                self._pending = None
            try:
                points = velocity_path(root, velocity, target, fps)
                offsets = np.arange(10, 10 * len(points) + 1, 10)
                # Same interpolation and smoother as RootKeyframe2DSet, outside
                # the playback thread and outside the scene publication lock.
                t = np.arange(offsets[0], offsets[-1] + 1)
                dense = np.stack([interp1d(offsets, points[:, 0])(t),
                                  np.zeros_like(t),
                                  interp1d(offsets, points[:, 2])(t)], axis=1)
                if len(points) >= 3:
                    dense = get_smooth_root_pos(torch.from_numpy(dense[None]))[0].numpy()
                segments = np.stack([dense[:-1], dense[1:]], axis=1)
                # Playback already holds client.atomic() when it calls update;
                # keep the same lock order here to prevent a lock inversion.
                with self.client.atomic(), self._lock:
                    if self._closed or not self._visible or target != self.last_target:
                        continue
                    # Reuse official spheres and 5px pure-red line. The scene
                    # update is brief and all annotation parts arrive together.
                    with self.visual._data_lock:
                        for offset in set(self.visual.keyframes) - set(offsets):
                            self.visual.remove_keyframe(f'target_vel_waypoint_{offset}', offset)
                        for offset, point in zip(offsets, points):
                            self.visual.add_keyframe(
                                f'target_vel_waypoint_{offset}', int(offset), point,
                                viz_label=False, exists_ok=True, update_path=False,
                                add_annulus=False)
                        self.visual.line_segments.points = segments
                        self.set_visibility(True)
            except Exception:
                logging.exception('Failed to draw ARDY velocity guide')

    def set_visibility(self, visible):
        with self._lock:
            self._visible = visible
            if self.visual.line_segments is not None:
                self.visual.line_segments.visible = visible
            for offset in tuple(self.visual.scene_elements):
                self.visual.set_keyframe_visibility(offset, visible)

    def clear(self):
        with self.client.atomic(), self._lock:
            self._closed = True
            self._pending = None
            self.visual.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
