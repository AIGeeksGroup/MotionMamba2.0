# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from .common import *  # noqa: F401,F403


class PlaybackMixin:
    @staticmethod
    def _frame_display_due(session, now):
        """Coalesce passive labels/timeline, never manual seek or pose updates."""
        if (getattr(session.model, 'is_remote_mm2', False)
                and getattr(session, 'mm2_rendering', False) and session.playing):
            if now - getattr(session, 'mm2_display_updated_at', float('-inf')) < .1:
                return False
        session.mm2_display_updated_at = now
        return True

    @staticmethod
    def _playback_rate(session) -> float:
        """Return source presentation FPS after any SONIC retiming."""
        playback_fps = float(session.model_fps)
        return playback_fps

    def _anchor_playback(self, session, frame_float: float | None = None) -> None:
        if frame_float is None:
            frame_float = float(max(0, session.frame_idx))
        frame_float = max(0.0, min(float(frame_float), float(max(0, session.max_frame_idx))))
        now_ns = time.monotonic_ns()
        session.playback_epoch_ns = now_ns
        session.playback_start_frame = frame_float
        session.frame_float = frame_float
        session.playback_epoch_id += 1

    def _log_playback_metrics(self, session, now_ns: int, target_fps: float) -> None:
        if not session.playback_metrics_anchor_ns:
            session.playback_metrics_anchor_ns = now_ns
            session.playback_last_report_ns = now_ns
            return
        if now_ns - session.playback_last_report_ns < 1_000_000_000:
            return
        samples = session.playback_set_frame_ms
        if samples:
            values = np.asarray(samples, dtype=np.float64)
            p50, p95, p99 = np.percentile(values, [50, 95, 99])
            over_40 = int(np.count_nonzero(values > 40.0))
        else:
            p50 = p95 = p99 = 0.0
            over_40 = 0
        elapsed_s = max(1e-9, (now_ns - session.playback_metrics_anchor_ns) / 1e9)
        print(
            "[ARDY playback diagnostic] "
            f"target_fps={target_fps:.3f} set_frame_hz={session.playback_pose_updates / elapsed_s:.2f} "
            f"set_frame_ms_p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} "
            f"over_40ms={over_40} expired_frames={session.playback_expired_frames} "
            f"epoch_id={session.playback_epoch_id}",
            flush=True,
        )
        session.playback_pose_updates = 0
        session.playback_expired_frames = 0
        session.playback_set_frame_ms.clear()
        session.playback_metrics_anchor_ns = now_ns
        session.playback_last_report_ns = now_ns

    def run_client_playback(self, client_id: int):
        """Follow the current model after failed loads or explicit reloads."""
        while self.client_active(client_id):
            session = self.client_sessions[client_id]
            if session.stop_playback:
                break
            if getattr(session.model, "is_remote_mm2", False):
                session.model.playback(self, session)
            else:
                self._run_client_local_playback(client_id)

    def _run_client_local_playback(self, client_id: int):
        """Playback loop for a specific client."""
        print(f"Starting playback loop for client {client_id}")

        elapsed_history = []
        session = self.client_sessions[client_id]
        self._anchor_playback(session)
        last_render_frame = None

        while True:
            # Check if client is still active and should continue
            if not self.client_active(client_id):
                print(f"Client {client_id} no longer active, stopping playback loop")
                break

            session = self.client_sessions[client_id]
            if session.stop_playback:
                print(f"Stop signal received for client {client_id}")
                break
            if getattr(session.model, "is_remote_mm2", False):
                break

            now_ns = time.monotonic_ns()
            playback_fps = float(session.model_fps)
            if getattr(self, "sonic_output", None) is not None:
                playback_fps = self.sonic_output.source_playback_fps(
                    session.model_fps
                )
            if session.playing:
                if not session.playback_epoch_ns:
                    self._anchor_playback(session)
                frame_float = session.playback_start_frame + (
                    (now_ns - session.playback_epoch_ns) / 1e9 * playback_fps
                )
                max_frame = float(max(0, session.max_frame_idx))
                if frame_float >= max_frame:
                    frame_float = max_frame
                if frame_float + 1e-6 < session.frame_float:
                    self._anchor_playback(session, session.frame_float)
                    frame_float = session.frame_float
                if last_render_frame is None or abs(frame_float - session.frame_float) > 1e-5:
                    previous_integer_frame = int(session.frame_idx)
                    before_ns = time.perf_counter_ns()
                    self.set_frame(client_id, frame_float)
                    session.playback_set_frame_ms.append(
                        (time.perf_counter_ns() - before_ns) / 1e6
                    )
                    session.playback_pose_updates += 1
                    last_render_frame = frame_float
                    # Count source samples skipped by a blocking GUI/replan call.
                    session.playback_expired_frames += max(
                        0, int(frame_float) - previous_integer_frame - 1
                    )
            else:
                last_render_frame = None

            self._log_playback_metrics(session, now_ns, playback_fps)
            next_wakeup = time.monotonic() + min(0.01, 0.5 / max(playback_fps, 1.0))
            time.sleep(max(0.0, next_wakeup - time.monotonic()))

            # Track moving average of actual fps
            elapsed = max(1e-9, time.monotonic_ns() / 1e9 - now_ns / 1e9)
            elapsed_history.append(elapsed)
            if len(elapsed_history) > 10:
                elapsed_history.pop(0)

            if self.client_active(client_id):
                session.gui_elements.gui_actual_fps.value = 1.0 / (sum(elapsed_history) / len(elapsed_history))

        print(f"Playback loop ended for client {client_id}")

    def run(self):
        """Main dummy loop to keep server alive."""
        print("Main server loop started")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Server shutting down...")
            # Signal all playback threads to stop
            for session in self.client_sessions.values():
                session.stop_playback = True

    def set_playback_state(self, client_id, playing, reason='user'):
        """One path for keyboard and GUI; never change the consumer cursor."""
        session = self.client_sessions[client_id]
        owner = getattr(self, 'mm2_physical_owner', None)
        if owner in self.client_sessions and owner != client_id:
            session.client.add_notification(title='Physical control owner', body='Another viewer owns SONIC.', auto_close_seconds=3)
            return
        remote = getattr(session.model, 'is_remote_mm2', False)
        if remote and not session.model.owns_bridge(self, session):
            return
        if remote and playing and getattr(session.model, 'planning', False):
            session.model.end_planning(self, session)
        session.playing = bool(playing)
        session.pause_reason = None if playing else reason
        self._anchor_playback(session, session.frame_float)
        bridge = getattr(self, 'sonic_output', None)
        if bridge is not None and not getattr(session.model, 'reference_preview', False):
            if remote:
                bridge.viewer_playback_control('play' if playing else 'pause')
            else:
                bridge.set_playing(session.playing, source_frame=max(0, session.frame_idx), source_fps=session.model_fps)
        g = session.gui_elements
        g.gui_play_pause_button.label = 'Pause' if playing else 'Play'
        g.gui_next_frame_button.disabled = bool(playing)
        g.gui_prev_frame_button.disabled = bool(playing)
        if remote and not playing and reason == 'user' and not session.model.failed:
            # Prepare exactly one reversible future window while paused. Do
            # not publish it; a later text/control update rewinds this draft.
            session.model.prepare_requested = True

    def set_frame(self, client_id: int, frame_idx: float, trigger_by_gui_timeline: bool = False, *, render_only: bool = False, preview_target_velocity: bool = False):
        """Set the current frame for a client."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client

        owner = getattr(self, 'mm2_physical_owner', None)
        if owner in self.client_sessions and owner != client_id:
            return
        # Visual control refresh must preserve even the subframe cursor.
        frame_float = float(session.frame_float if render_only else frame_idx)
        max_frame = max(0, int(session.max_frame_idx))
        frame_float = max(0.0, min(frame_float, float(max_frame)))
        render_frame_idx = int(np.floor(frame_float))
        alpha = frame_float - render_frame_idx
        session.frame_float = frame_float
        session.frame_idx = render_frame_idx
        if (getattr(session.model, 'is_remote_mm2', False)
                and not render_only
                and not getattr(session, 'mm2_rendering', False)
                and session.motion_tensor is not None):
            session.model.seek_playback(self, session, render_frame_idx)
        if getattr(self, "sonic_output", None) is not None:
            if not render_only and (trigger_by_gui_timeline or not session.playing) and not getattr(session, "mm2_rendering", False) and not getattr(session.model, 'is_remote_mm2', False):
                self.sonic_output.seek(
                    source_frame=render_frame_idx, source_fps=session.model_fps
                )
        if trigger_by_gui_timeline:
            self._anchor_playback(session, frame_float)
        # Update the Viser timeline GUI if not triggered by it
        refresh_display = self._frame_display_due(session, time.monotonic())
        if refresh_display and not trigger_by_gui_timeline and hasattr(client, "timeline"):
            try:
                if getattr(client.timeline, '_current_frame', None) != render_frame_idx:
                    client.timeline.set_current_frame(render_frame_idx)
                # Frame numbers are global. The Viser client pans its viewport
                # to reveal an out-of-view cursor without moving this origin.
                window_end = max(session.max_frame_idx, TIMELINE_WINDOW_AFTER)
                if getattr(session.model, 'sequential_preload', False):
                    window_end = max(0, session.max_frame_idx)
                if getattr(session.model, 'planning', False) and not getattr(session.model, 'sequential_preload', False):
                    window_end = max(window_end, int(session.model.consumed_frame)+1500)
                if getattr(session.model, 'is_remote_mm2', False):
                    velocity = session.model.display_target_velocity_at(session, render_frame_idx)
                    if velocity is not None:
                        window_end = max(window_end, render_frame_idx + int(session.model_fps * VELOCITY_TRANSITION_DURATION))
                if (getattr(client.timeline, '_start_frame', None),
                        getattr(client.timeline, '_end_frame', None)) != (0, window_end):
                    client.timeline.set_frame_range(start_frame=0, end_frame=window_end)
            except (AttributeError, Exception) as e:
                print(f"Could not update timeline frame: {e}")

        session.cur_time = frame_float / session.model_fps
        if refresh_display:
            session.gui_elements.gui_current_time.value = session.cur_time
            session.gui_elements.gui_frame_idx_input.value = render_frame_idx
        if getattr(session.model, 'is_remote_mm2', False) and not session.playing:
            session.gui_elements.gui_prev_frame_button.disabled = render_frame_idx == 0
            session.gui_elements.gui_next_frame_button.disabled = render_frame_idx >= max_frame

        # Check if approaching end of timeline
        thresh = session.gui_elements.gui_replan_trigger_thresh.value
        enable_auto_replan = session.gui_elements.gui_enable_auto_replan_checkbox.value
        if (not getattr(session.model, 'is_remote_mm2', False)
            and not trigger_by_gui_timeline and enable_auto_replan
            and session.max_frame_idx - render_frame_idx <= thresh):
            # Cheap pre-check to avoid spawning a thread while a replan runs;
            # skip_if_busy makes the trigger drop atomically if another thread
            # won the race between this check and the lock acquisition.
            if not session.replan_lock.locked():
                threading.Thread(
                    target=self.on_replan_trigger,
                    args=(client_id,),
                    kwargs={"skip_if_busy": True},
                    daemon=True,
                ).start()

        # Update constraint visibility - show constraints that have been reached (frame_idx <= current frame)
        show_hand_orientations = session.gui_elements.gui_viz_hand_orientations_checkbox.value
        hide_distant_constraints = session.gui_elements.gui_viz_hide_distant_constraints_checkbox.value

        # Calculate max visible future frame if hiding distant constraints
        max_future_frame = float("inf")
        if hide_distant_constraints:
            future_crop = session.gui_elements.gui_future_crop_length.value
            gen_horizon = session.gen_horizon_len
            max_future_frame = render_frame_idx + future_crop + gen_horizon

        for track_name, constraint in session.constraints.items():
            # A snapshot of keys alone is insufficient: a concurrent delete
            # can remove a scene element before its visibility is updated.
            with constraint._data_lock:
                for constraint_frame_idx in constraint.scene_elements.keys():
                    visibility = constraint_frame_idx >= render_frame_idx
                    if hide_distant_constraints and constraint_frame_idx > max_future_frame:
                        visibility = False
                    if track_name == "End-Effectors":
                        constraint.set_keyframe_visibility(
                            constraint_frame_idx, visibility,
                            show_rotation_axes=show_hand_orientations,
                        )
                    else:
                        constraint.set_keyframe_visibility(constraint_frame_idx, visibility)
                if track_name == "2D Root":
                    constraint.set_interval_labels_visibility(render_frame_idx)

        # Update character poses and get root position for target velocity arrow
        # Note: Each character has its own actual velocity arrow (blue) shown on their skeleton
        # We also have ONE shared target velocity arrow (orange) for the entire session
        root_pos_for_target = None
        with session.characters_lock:
            if render_frame_idx >= 0 and render_frame_idx <= session.max_frame_idx:
                for character_idx, character in enumerate(session.characters.values()):
                    next_idx = min(render_frame_idx + 1, session.max_frame_idx)
                    if alpha > 0.0 and next_idx != render_frame_idx:
                        joints_pos = torch.lerp(
                            session.joints_pos[character_idx, render_frame_idx],
                            session.joints_pos[character_idx, next_idx], alpha,
                        )
                        joints_rot = self._interpolate_rotation_matrices(
                            session.joints_rot[character_idx, render_frame_idx],
                            session.joints_rot[character_idx, next_idx],
                            alpha,
                        )
                    else:
                        joints_pos = session.joints_pos[character_idx, render_frame_idx]
                        joints_rot = session.joints_rot[character_idx, render_frame_idx]
                    # Get actual root velocity for this frame (pass as tensor)
                    root_velocity = None
                    if session.root_velocities is not None:
                        root_velocity = session.root_velocities[character_idx, render_frame_idx]

                    # Keyboard velocity annotations remain visible when the
                    # diagnostic skeleton is hidden behind the STL robot.
                    if getattr(session.model, 'is_remote_mm2', False) and character.skeleton_mesh is not None:
                        recorded = getattr(session.model, 'display_target_velocity_at',
                                           session.model.target_velocity_at)(session, render_frame_idx)
                        character.skeleton_mesh.velocity_arrow_visible = (
                            recorded is not None or
                            (preview_target_velocity and session.gui_elements.gui_use_target_velocity_checkbox.value))

                    foot_contacts = (
                        session.foot_contacts[character_idx, render_frame_idx] > 0.5
                        if session.foot_contacts is not None
                        else None
                    )
                    character.set_pose(
                        joints_pos,
                        joints_rot,
                        foot_contacts=foot_contacts,
                        root_velocity=root_velocity,
                    )

                    # Store root position from first character for target velocity visualization
                    if character_idx == 0 and root_pos_for_target is None:
                        root_pos_for_target = joints_pos[character.skeleton.root_idx]

        # Update reference motion character
        if (
            session.ref_character is not None
            and session.ref_joints_pos is not None
            and session.gui_elements.gui_viz_ref_motion_checkbox.value
        ):
            ref_frame = min(render_frame_idx, session.ref_joints_pos.shape[0] - 1)
            if ref_frame >= 0:
                session.ref_character.set_pose(
                    session.ref_joints_pos[ref_frame],
                    session.ref_joints_rot[ref_frame],
                )

        # Update hand gizmos if enabled
        if session.gui_elements.gui_viz_hand_orientations_checkbox.value:
            if not session.hand_gizmos:
                self.create_hand_gizmos(client_id)
            self.update_hand_gizmos(client_id, render_frame_idx)
        elif session.hand_gizmos:
            # Hide hand gizmos if checkbox is off
            self.set_hand_gizmos_visibility(client_id, False)

        # Update target velocity arrow visualization (orange, user-specified target)
        if session.target_velocity_arrow is not None:
            gui_elements = session.gui_elements
            remote = getattr(session.model, 'is_remote_mm2', False)
            target_vel_xz = (getattr(session.model, 'display_target_velocity_at',
                                    session.model.target_velocity_at)(session, render_frame_idx) if remote
                             else (gui_elements.gui_target_root_velocity.value
                                   if gui_elements.gui_use_target_velocity_checkbox.value else None))
            # Editing previews the requested direction immediately. Normal
            # playback/seek still uses accepted controls at the rendered frame.
            if preview_target_velocity:
                target_vel_xz = (gui_elements.gui_target_root_velocity.value
                                if gui_elements.gui_use_target_velocity_checkbox.value else None)
            # Show target velocity arrow if enabled and we have a valid root position
            if target_vel_xz is not None and root_pos_for_target is not None:
                # Remote replay uses accepted controls at the rendered frame.
                target_velocity = np.array([target_vel_xz[0], 0.0, target_vel_xz[1]])

                # Update target velocity arrow
                session.target_velocity_arrow.update(
                    root_velocity=target_velocity,
                    root_pos=root_pos_for_target,
                    visible=True,
                )

                # Use the repository's real 2D Root waypoints and timeline
                # entries, rather than a second, independently drawn guide.
                if (render_frame_idx % TARGET_VELOCITY_UPDATE_INTERVAL == 0
                    or (remote and preview_target_velocity)):
                    self._update_root_constraints_from_target_velocity(client_id, render_frame_idx, target_velocity)
            else:
                # Hide target velocity arrow if disabled or no character available
                session.target_velocity_arrow.set_visibility(False)
                if remote:
                    self._clear_target_velocity_waypoints(client_id)

            marker = getattr(session, 'start_direction_marker', None)
            if remote and marker is not None:
                # Do not confuse the fixed initial-pose marker with the live
                # blue velocity arrow at the character's current feet.
                marker.set_visibility(target_vel_xz is None and
                    gui_elements.gui_show_start_direction_checkbox.value)

        # Update camera
        if session.gui_elements.gui_viz_auto_camera_checkbox.value:
            self.update_camera_follow(client_id, render_frame_idx)

    @staticmethod
    def _interpolate_rotation_matrices(first, second, alpha):
        """Interpolate global rotations and project back to SO(3)."""
        if alpha <= 0.0:
            return first
        if alpha >= 1.0:
            return second
        blended = first * (1.0 - alpha) + second * alpha
        try:
            u, _, vh = torch.linalg.svd(blended)
            rotation = u @ vh
            negative = torch.det(rotation) < 0
            if bool(torch.any(negative)):
                u = u.clone()
                u[negative, :, -1] *= -1
                rotation = u @ vh
            return rotation
        except (RuntimeError, AttributeError):
            return blended
