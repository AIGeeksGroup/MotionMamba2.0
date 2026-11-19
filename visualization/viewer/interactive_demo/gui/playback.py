# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: Playback tab (split from create_gui)."""

from ..common import *  # noqa: F401,F403


class GuiPlaybackMixin:
    def _build_playback_tab(self, client, client_id, tab_group, g, timeline, default_prompt):
        with tab_group.add_tab("Playback", viser.Icon.PLAYER_PLAY):
            g.gui_play_pause_button = client.gui.add_button("Play", disabled=False)
            g.gui_next_frame_button = client.gui.add_button(
                "Next Frame", disabled=False, icon=viser.Icon.PLAYER_TRACK_NEXT_FILLED
            )
            g.gui_prev_frame_button = client.gui.add_button(
                "Prev Frame", disabled=True, icon=viser.Icon.PLAYER_TRACK_PREV_FILLED
            )
            g.gui_actual_fps = client.gui.add_number("Actual FPS", initial_value=30.0, step=0.0001, disabled=True)
            g.gui_current_time = client.gui.add_number("Current Time (s)", initial_value=0.0, step=0.01, disabled=True)
            g.gui_model_fps = client.gui.add_number("Native FPS", initial_value=30, disabled=True)
            g.gui_frame_idx_input = client.gui.add_number("Frame Index", initial_value=0, min=0, max=199, step=1)
            g.gui_enable_auto_replan_checkbox = client.gui.add_checkbox("Enable Auto Replan", initial_value=True)

            @g.gui_frame_idx_input.on_update
            def _(event) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if event.client is None and getattr(session.model, 'is_remote_mm2', False):
                    return  # renderer updates are not user seek requests
                requested_frame = int(g.gui_frame_idx_input.value)
                if (event.client is not None and getattr(session.model, 'is_remote_mm2', False)
                        and not session.playing):
                    if not session.model.planning:
                        session.model.begin_planning(self, session, True)
                    session.model.select_planning_frame(self, session, requested_frame)
                    return
                frontier = max(0, session.max_frame_idx)
                # Viser clamps values above ``max`` before invoking this callback.
                # Reaching the frontier from a real browser event is therefore
                # also the signal that the user requested a frame beyond it.
                target_frame = max(0, min(requested_frame, frontier))
                if target_frame != session.frame_idx:
                    self.set_frame(client_id, target_frame, trigger_by_gui_timeline=True)
                    # Update button states
                    g.gui_prev_frame_button.disabled = target_frame == 0
                    g.gui_next_frame_button.disabled = target_frame == frontier
                if event.client is not None and hasattr(client, "timeline"):
                    try:
                        # Frame Index is a separate GUI control, so explicitly
                        # synchronize the bottom timeline cursor and its view.
                        client.timeline.set_current_frame(target_frame)
                        window_start = 0
                        window_end = max(frontier, TIMELINE_WINDOW_AFTER)
                        client.timeline.set_frame_range(
                            start_frame=window_start,
                            end_frame=window_end,
                        )
                    except (AttributeError, Exception) as exc:
                        print(f"Could not sync Frame Index to timeline: {exc}")
                if (
                    event.client is not None
                    and requested_frame >= frontier
                    and g.gui_enable_auto_replan_checkbox.value
                    and not session.replan_lock.locked()
                ):
                    threading.Thread(
                        target=self.on_replan_trigger,
                        args=(client_id,),
                        kwargs={"skip_if_busy": True},
                        daemon=True,
                    ).start()

            @g.gui_play_pause_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                self.set_playback_state(client_id, not session.playing)

            @g.gui_next_frame_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if session.frame_idx < session.max_frame_idx:
                    new_frame = session.frame_idx + 1
                    self.set_frame(client_id, new_frame)
                    g.gui_prev_frame_button.disabled = False
                    if new_frame == session.max_frame_idx:
                        g.gui_next_frame_button.disabled = True

            @g.gui_prev_frame_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if session.frame_idx > 0:
                    new_frame = session.frame_idx - 1
                    self.set_frame(client_id, new_frame)
                    g.gui_next_frame_button.disabled = False
                    if new_frame == 0:
                        g.gui_prev_frame_button.disabled = True

        #
        # Text tab
        #
