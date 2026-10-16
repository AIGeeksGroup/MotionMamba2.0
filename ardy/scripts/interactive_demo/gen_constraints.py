# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constraint computation for InteractiveTimelineDemo (split from generation)."""

from .common import *  # noqa: F401,F403


def _to_device_tensor(x, device) -> torch.Tensor:
    """Accept a numpy array or tensor and return a tensor on ``device``."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device)


class GenConstraintsMixin:
    def compute_model_constraints_lst(
        self,
        session: ClientSession,
        num_frames: int,
        history_end_idx: int,
    ):
        """Compute the lst of constraints for the model based on the constraints in viser."""
        if not session.constraints:
            return []

        device = self.device
        dense_root_pos_2d = None

        model_constraints = []
        for track_name in ["2D Root", "Full-Body", "End-Effectors"]:
            if track_name not in session.constraints:
                continue
            constraint = session.constraints[track_name]
            constraint_info = constraint.get_constraint_info()
            frame_idx = constraint_info["frame_idx"]
            # drop any constraints outside the generation range
            valid_info = [(i, fi) for i, fi in enumerate(frame_idx) if fi < num_frames and fi > history_end_idx]
            valid_idx = [i for i, _ in valid_info]
            valid_frame_idx = [fi for _, fi in valid_info]

            if len(valid_frame_idx) == 0:
                continue
            if len(valid_frame_idx) != len(frame_idx):
                print(f"Dropping {len(frame_idx) - len(valid_frame_idx)} constraints outside the generation range")

            frame_indices = torch.tensor(valid_frame_idx)
            # TODO: for root keyframe, probably want to constrain hips to be 0 as well,
            #        not just the smoothed root which could be offset
            # For dense 2D root I think it is okay, maybe we could do this for sparse 2D roots?
            if track_name == "2D Root":
                root_pos_2d = _to_device_tensor(constraint_info["root_pos"][valid_idx][:, [0, 2]], device)
                model_constraints.append(
                    Root2DConstraintSet(
                        session.motion_rep.skeleton,
                        frame_indices,
                        root_pos_2d,
                        global_root_heading=(
                            _to_device_tensor(constraint_info['global_root_heading'][valid_idx], device)
                            if 'global_root_heading' in constraint_info else None),
                    )
                )

                # get the full 2d root for other tracks
                if session.constraints["2D Root"].dense_path:
                    dense_root_pos_2d = _to_device_tensor(constraint_info["root_pos"][:, [0, 2]], device)
            elif track_name == "Full-Body":
                constraint_joints_pos = _to_device_tensor(constraint_info["joints_pos"][valid_idx], device)
                constraint_joints_rot = _to_device_tensor(constraint_info["joints_rot"][valid_idx], device)

                root_pos_2d = None
                if dense_root_pos_2d is not None:
                    root_pos_2d = dense_root_pos_2d[frame_indices]

                model_constraints.append(
                    FullBodyConstraintSet(
                        session.motion_rep.skeleton,
                        frame_indices,
                        constraint_joints_pos,
                        constraint_joints_rot,
                        root_2d=root_pos_2d,
                    )
                )
            elif track_name == "End-Effectors":
                constraint_joints_pos = _to_device_tensor(constraint_info["joints_pos"][valid_idx], device)
                constraint_joints_rot = _to_device_tensor(constraint_info["joints_rot"][valid_idx], device)

                end_effector_type_set_lst = [
                    end_effector_type_set
                    for i, end_effector_type_set in enumerate(constraint_info["end_effector_type"])
                    if i in valid_idx
                ]

                # regroup the end effector data by type
                cls_idx = defaultdict(list)
                for idx, end_effector_type_set in enumerate(end_effector_type_set_lst):
                    for end_effector_type in end_effector_type_set:
                        cls_idx[TYPE_TO_CLASS[end_effector_type]].append(idx)

                for cls, lst_idx in cls_idx.items():
                    frame_indices_cls = frame_indices[lst_idx]
                    root_pos_2d = None
                    if dense_root_pos_2d is not None:
                        root_pos_2d = dense_root_pos_2d[frame_indices_cls]

                    constraint_joints_pos_el = constraint_joints_pos[lst_idx]
                    constraint_joints_rot_el = constraint_joints_rot[lst_idx]

                    model_constraints.append(
                        cls(
                            session.motion_rep.skeleton,
                            frame_indices_cls,
                            constraint_joints_pos_el,
                            constraint_joints_rot_el,
                            root_2d=root_pos_2d,
                        )
                    )
            else:
                raise ValueError(f"Unsupported constraint type: {constraint.display_name}")
        return model_constraints

    def compute_constraint_mask(
        self,
        session: ClientSession,
        num_samples: int,
        num_frames: int,
        history_end_idx: int,
    ):
        """Compute the mask for the constraints."""
        if len(session.constraints) == 0:
            print("No constraints to compute constraint mask")
            return None, None

        device = self.device
        model_constraints = self.compute_model_constraints_lst(session, num_frames, history_end_idx)

        if len(model_constraints) == 0:
            print("No valid constraints to compute constraint mask")
            return None, None

        observed_motion, motion_mask = session.motion_rep.create_conditions_from_constraints(
            model_constraints,
            length=num_frames,
            to_normalize=False,
            device=device,
        )

        observed_motion = session.motion_rep.normalize(observed_motion)
        observed_motion = (
            observed_motion * motion_mask
        )  # mask out the unobserved frames that are non zero due to normalization

        # Repeat for num_samples
        observed_motion = repeat(observed_motion, "t d -> b t d", b=num_samples)
        motion_mask = repeat(motion_mask, "t d -> b t d", b=num_samples)

        return motion_mask, observed_motion

    def _update_root_constraints_from_target_velocity(
        self, client_id: int, current_frame_idx: int, target_velocity: np.ndarray
    ):
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        lock = session.__dict__.setdefault('velocity_waypoint_lock', threading.RLock())
        with lock, session.client.atomic():
            key = (int(current_frame_idx), tuple(float(v) for v in target_velocity),
                   bool(session.gui_elements.gui_use_target_heading_checkbox.value))
            if getattr(session, 'velocity_waypoint_key', None) == key:
                return
            self._update_velocity_waypoints_impl(client_id, current_frame_idx, target_velocity)
            session.velocity_waypoint_key = key

    def _update_velocity_waypoints_impl(
        self, client_id: int, current_frame_idx: int, target_velocity: np.ndarray
    ):
        """Predict future root positions using target velocity and update 2D root constraints.

        Args:
            client_id: Client ID
            current_frame_idx: Current frame index
            target_velocity: Target velocity [vx, vy, vz]
        """
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        # Check if we have motion data
        if session.joints_pos is None or session.joints_pos.shape[0] == 0:
            return

        # Get model FPS and calculate prediction duration
        fps = session.model_fps
        num_future_frames = int(fps * VELOCITY_TRANSITION_DURATION)

        # Predict future root positions using target velocity with smooth transition
        # Start from current root position (first character)
        root_idx = session.motion_rep.skeleton.root_idx
        current_root_pos = session.joints_pos[0, current_frame_idx, root_idx].clone()  # [3]

        # Get current velocity from motion data
        if current_frame_idx > 0 and session.joints_pos.shape[1] > current_frame_idx:
            prev_root_pos = session.joints_pos[0, current_frame_idx - 1, root_idx]
            current_velocity = (current_root_pos - prev_root_pos) * fps  # velocity = displacement * fps
        else:
            current_velocity = torch.zeros_like(current_root_pos)

        # Convert target_velocity to torch tensor
        target_velocity_tensor = torch.tensor(
            target_velocity, dtype=torch.float32, device=current_root_pos.device
        )  # [3]

        # Calculate number of frames for velocity transition
        transition_duration = VELOCITY_TRANSITION_DURATION  # seconds
        num_transition_frames = int(fps * transition_duration)
        num_transition_frames = min(num_transition_frames, num_future_frames)

        # Generate interpolated velocities and positions
        dt = 1.0 / fps
        future_root_pos = []
        interpolated_velocities = []

        for i in range(num_future_frames):
            # Calculate interpolation factor (0 to 1 over transition period)
            if i < num_transition_frames:
                alpha = (i + 1) / num_transition_frames  # Linear interpolation
            else:
                alpha = 1.0  # Full target velocity after transition

            # Interpolate velocity
            interp_velocity = (1 - alpha) * current_velocity + alpha * target_velocity_tensor
            interpolated_velocities.append(interp_velocity)

            # Calculate position incrementally
            if i == 0:
                new_pos = current_root_pos + interp_velocity * dt
            else:
                new_pos = future_root_pos[-1] + interp_velocity * dt

            future_root_pos.append(new_pos)

        # Stack into tensor
        future_root_pos = torch.stack(future_root_pos, dim=0)  # [F, 3]
        interpolated_velocities = torch.stack(interpolated_velocities, dim=0)  # [F, 3]

        recycled = {}
        root_constraint = session.constraints['2D Root']
        # Keep the original waypoint meshes alive while their absolute frame
        # numbers advance. Recreating five subdivision-3 spheres every update
        # resends their entire geometry and stalls browser rendering.
        if getattr(session.model, 'is_remote_mm2', False):
            with root_constraint._data_lock:
                for frame, ids in list(root_constraint.frame2keyid.items()):
                    if len(ids) == 1 and str(ids[0]).startswith('target_vel_waypoint_'):
                        key = ids[0]
                        recycled[key] = (frame, root_constraint.scene_elements.pop(frame))
                        root_constraint.keyframes.pop(frame)
                        root_constraint.frame2keyid.pop(frame)
                        root_constraint.root_headings.pop(frame, None)

        # The remote keyboard guide uses the original sparse waypoints. Do not
        # force the independent dense-path smoother used by the old overlay.
        if getattr(session.model, 'is_remote_mm2', False):
            session.constraints['2D Root'].smooth_path = False
        self._clear_target_velocity_waypoints(client_id, update_path=False)

        # Add new 2D root waypoint constraints at regular intervals (e.g., every 4 frames)
        waypoint_interval = TARGET_VELOCITY_GOAL_FRAME_INTERVAL  # Add waypoint every N frames
        root_constraint = session.constraints["2D Root"]

        num_new_waypoints = 0
        timeline_waypoints = {}
        for i in range(waypoint_interval, num_future_frames + 1, waypoint_interval):
            if i > len(future_root_pos):
                break

            future_frame_idx = current_frame_idx + i
            waypoint_pos = future_root_pos[i - 1]  # [3]
            root_pos = torch.tensor(
                [waypoint_pos[0].item(), 0.0, waypoint_pos[2].item()],
                dtype=torch.float32,
            )

            # Create unique waypoint ID
            waypoint_id = f"target_vel_waypoint_{i}"

            # Calculate root heading from interpolated velocity direction if enabled
            gui_elements = session.gui_elements
            global_root_heading = None
            if gui_elements.gui_use_target_heading_checkbox.value:
                # Get the interpolated velocity for this frame
                velocity_for_frame = interpolated_velocities[i - 1]  # [3]
                vel_x = velocity_for_frame[0].item()
                vel_z = velocity_for_frame[2].item()

                # WaypointMesh's visual heading is atan2(z, x), as in the
                # referenced repository. The remote model's +Z-zero heading
                # remains encoded separately by controls(); these preview
                # keyframes are excluded from mouse/model conditioning.
                import math

                global_root_heading = (math.atan2(vel_z, vel_x)
                    if getattr(session.model, 'is_remote_mm2', False)
                    else math.atan2(vel_x, vel_z))

            with root_constraint._data_lock:
                old = recycled.pop(waypoint_id, None)
                if old is not None:
                    _, scene = old
                    if future_frame_idx not in root_constraint.keyframes:
                        root_constraint.scene_elements[future_frame_idx] = scene
                        root_constraint.keyframes[future_frame_idx] = root_pos.numpy()
                        root_constraint.frame2keyid[future_frame_idx] = []
                        if global_root_heading is None:
                            for name in ('arrow_base', 'arrow_head'):
                                mesh = getattr(scene['waypoint'], name, None)
                                if mesh is not None:
                                    session.client.scene.remove_by_name(mesh.name)
                                    setattr(scene['waypoint'], name, None)
                    else:
                        scene['waypoint'].clear()
                root_constraint.add_keyframe(
                    keyframe_id=waypoint_id,
                    frame_idx=future_frame_idx,
                    root_pos=root_pos,
                    global_root_heading=global_root_heading,
                    viz_label=False,
                    exists_ok=True,
                    update_path=False,
                    add_annulus=False,
                )
                if getattr(session.model, 'is_remote_mm2', False):
                    root_constraint.scene_elements[future_frame_idx]['waypoint'].sphere.visible = True
            num_new_waypoints += 1
            if getattr(session.model, 'is_remote_mm2', False):
                timeline_waypoints[waypoint_id] = future_frame_idx
            else:
                self.add_keyframe_to_timeline(client_id, "2D Root", future_frame_idx, waypoint_id)

        for key, (frame, scene) in recycled.items():
            scene['waypoint'].clear()
            self.remove_keyframe_from_timeline(client_id, '2D Root', frame, key)

        if timeline_waypoints:
            self._sync_velocity_waypoint_timeline(client_id, timeline_waypoints)

        if getattr(session.model, 'is_remote_mm2', False):
            # Draw the sparse route between the same five registered waypoints;
            # no extra root segment, dense smoothing or second set of dots.
            points = future_root_pos[waypoint_interval - 1::waypoint_interval].cpu().numpy().copy()
            points[:, 1] = 0.
            if root_constraint.line_segments is None:
                root_constraint.line_segments = session.client.scene.add_line_segments(
                    name=f'/{root_constraint.name}/line_segments',
                    points=np.zeros((1, 2, 3)), colors=(255, 0, 0), line_width=5.0)
            root_constraint.line_segments.points = np.stack([points[:-1], points[1:]], axis=1)
            root_constraint.line_segments.visible = True

    def _sync_velocity_waypoint_timeline(self, client_id, waypoints):
        """Move real 2D Root keys without replacing their browser identities.

        This Viser release has no public batch/update API. Use its dataclass
        state for an existing set and publish once; other releases fall back
        to the public add API with stable UUIDs.
        """
        from dataclasses import is_dataclass, replace

        session = self.client_sessions[client_id]
        data = session.timeline_data
        timeline = getattr(session.client, 'timeline', None)
        if data is None or timeline is None:
            return
        track_id = data.get('tracks_ids', {}).get('2D Root')
        if track_id is None:
            return
        with data['keyframe_update_lock']:
            tracked = data['keyframes']
            state = getattr(timeline, '_keyframes', {})
            publish = getattr(timeline, '_send_timeline_update', None)
            if callable(publish) and all(
                key in tracked and is_dataclass(state.get(tracked[key]['timeline_uuid']))
                for key in waypoints
            ):
                changed = False
                for key, frame in waypoints.items():
                    entry = tracked[key]
                    if entry['frame'] != frame:
                        uuid = entry['timeline_uuid']
                        state[uuid] = replace(state[uuid], frame=frame)
                        entry['frame'] = frame
                        changed = True
                if changed:
                    publish()
                return
            for key, frame in waypoints.items():
                entry = tracked.get(key)
                if entry is not None and entry['frame'] == frame:
                    continue
                kwargs = {'uuid': entry['timeline_uuid']} if entry else {}
                uuid = timeline.add_keyframe(track_id, frame, **kwargs)
                tracked[key] = dict(frame=frame, track_id=track_id, timeline_uuid=uuid)

    def _clear_target_velocity_waypoints(self, client_id: int, *, update_path: bool = True):
        """Remove only keyboard-generated 2D Root keys and their timeline IDs."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        lock = session.__dict__.setdefault('velocity_waypoint_lock', threading.RLock())
        with lock, session.client.atomic():
            self._clear_velocity_waypoints_impl(client_id, update_path=update_path)
            session.velocity_waypoint_key = None

    def _clear_velocity_waypoints_impl(self, client_id: int, *, update_path: bool):
        session = self.client_sessions[client_id]
        root_constraint = session.constraints.get('2D Root')
        if root_constraint is None:
            return
        keyframes_to_remove = []

        # Collect all target velocity waypoints to remove (make a copy to avoid modification during iteration)
        with root_constraint._data_lock:
            existing_ids = [(frame, list(ids)) for frame, ids in root_constraint.frame2keyid.items()]
        for frame_idx, keyframe_ids in existing_ids:
            for keyframe_id in list(keyframe_ids):
                if isinstance(keyframe_id, str) and keyframe_id.startswith("target_vel_waypoint_"):
                    keyframes_to_remove.append((frame_idx, keyframe_id))

        # Remove collected keyframes
        for frame_idx, keyframe_id in keyframes_to_remove:
            # Remove from constraint data structures
            root_constraint.remove_keyframe(keyframe_id, frame_idx)

            # Remove from timeline GUI using the proper API
            self.remove_keyframe_from_timeline(
                client_id=client_id,
                constraint_type="2D Root",
                frame_idx=frame_idx,
                constraint_id=keyframe_id,
            )

        if update_path and getattr(session.model, 'is_remote_mm2', False):
            if root_constraint.line_segments is not None:
                if root_constraint.dense_path and len(root_constraint.keyframes) >= 2:
                    root_constraint.update_line_segments()
                else:
                    root_constraint.line_segments.visible = False
