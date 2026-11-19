# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API for timeline management in Viser."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple

from . import _messages
from ._threadpool_exceptions import print_threadpool_errors

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop
    from concurrent.futures import ThreadPoolExecutor

    from ._viser import ClientHandle, ViserServer
    from .infra._infra import ClientId, WebsockMessageHandler

logger = logging.getLogger(__name__)

def _make_timeline_uuid() -> str:
    """Generate a unique identifier for timeline prompts."""
    import uuid
    return str(uuid.uuid4())


# Primary colors for prompts (cycling) - matches frontend
PROMPT_COLORS: list[tuple[int, int, int]] = [
    (82, 133, 166),  # Default (teal-blue)
    (239, 68, 68),   # Red
    (249, 115, 22),  # Orange
    (234, 179, 8),   # Yellow
    (34, 197, 94),   # Green
    (59, 130, 246),  # Blue
    (168, 85, 247),  # Purple
    (236, 72, 153),  # Pink
]


def _colors_equal(
    a: tuple[int, int, int] | None, b: tuple[int, int, int] | None
) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def _color_for_compare(
    color: tuple[int, int, int] | None, has_prompt: bool
) -> tuple[int, int, int] | None:
    if not has_prompt:
        return color
    return PROMPT_COLORS[0] if color is None else color


class TimelineApi:
    """API for managing the timeline UI in Viser.
    
    The timeline displays a horizontal timeline with configurable prompts,
    similar to a Gantt chart. Each prompt has text content, a time range
    (start_frame to end_frame), and an optional color.
    """

    def __init__(
        self,
        owner: ViserServer | ClientHandle,
        websock_interface: WebsockMessageHandler,
        thread_executor: ThreadPoolExecutor,
        event_loop: AbstractEventLoop,
    ) -> None:
        self._owner = owner
        self._websock_interface = websock_interface
        self._thread_executor = thread_executor
        self._event_loop = event_loop
        self._enabled = False
        self._start_frame = 0
        self._end_frame = 100
        self._current_frame = 0
        # Split-text is the only supported timeline behavior.
        self._constraints_enabled = True  # Whether constraint editing is enabled
        self._min_prompt_duration = 1  # Minimum number of frames for a prompt
        self._max_prompt_duration = None  # Maximum number of frames for a prompt (None = no limit)
        self._default_prompt_text: str = ""
        self._default_prompt_duration: int = 30
        self._default_num_frames_zoom: int = 300
        self._max_frames_zoom: int = 1000
        self._fps: float = 30.0
        self._next_color_index: int = 0  # Track next color index for cycling
        # Arrow-key overlay state (UI-only, rendered above timeline on the left).
        self._arrow_overlay_enabled: bool = False
        self._arrow_overlay_highlighted: tuple[_messages.ArrowKeyType, ...] = ()
        self._arrow_overlay_base_opacity: float = 1.0
        self._arrow_overlay_highlight_opacity: float = 1.0
        self._arrow_overlay_position: _messages.ArrowKeyOverlayPosition = "bottom_left"
        self._prompts: dict[str, _messages.TimelinePrompt] = {}
        self._tracks: dict[str, _messages.TimelineTrack] = {}
        self._keyframes: dict[str, _messages.TimelineKeyframe] = {}
        self._intervals: dict[str, _messages.TimelineInterval] = {}
        self._frame_change_cb: list[Callable[[int], None]] = []
        self._keyframe_add_cb: list[Callable[[str, str, int], None]] = []  # (keyframe_id, track_id, frame)
        self._keyframe_move_cb: list[Callable[[str, int], None]] = []  # (keyframe_id, new_frame)
        self._keyframe_delete_cb: list[Callable[[str], None]] = []  # (keyframe_id,)
        self._interval_add_cb: list[Callable[[str, str, int, int], None]] = []  # (interval_id, track_id, start, end)
        self._interval_move_cb: list[Callable[[str, int, int], None]] = []  # (interval_id, new_start, new_end)
        self._interval_delete_cb: list[Callable[[str], None]] = []  # (interval_id,)

        # Prompt callbacks: these are triggered by *UI interactions* (client->server messages),
        # mirroring the keyframe/interval callback behavior.
        self._prompt_add_cb: list[
            Callable[[str, int, int, str, Tuple[int, int, int] | None], None]
        ] = []  # (prompt_id, start_frame, end_frame, text, color)
        self._prompt_delete_cb: list[Callable[[str], None]] = []  # (prompt_id,)
        self._prompt_update_cb: list[Callable[[str, str], None]] = []  # (prompt_id, new_text)
        self._prompt_resize_cb: list[Callable[[str, int, int], None]] = []  # (prompt_id, new_start, new_end)
        self._prompt_move_cb: list[Callable[[str, int, int], None]] = []  # (prompt_id, new_start, new_end)
        self._prompt_swap_cb: list[Callable[[str, str], None]] = []  # (prompt_id_1, prompt_id_2)
        self._prompt_split_cb: list[Callable[[str, str, str, int], None]] = []  # (original_id, left_id, right_id, split_frame)
        self._prompt_merge_cb: list[Callable[[str, str], None]] = []  # (left_id, right_id)
        
        # Track whether this timeline has ever been used/configured
        # This prevents unused client timelines from interfering with server timelines
        self._has_been_used = False
        
        # Register handlers for timeline updates from client
        self._websock_interface.register_handler(
            _messages.TimelineUpdateMessage, self._handle_timeline_update
        )
        self._websock_interface.register_handler(
            _messages.TimelineKeyframeAddMessage, self._handle_keyframe_add
        )
        self._websock_interface.register_handler(
            _messages.TimelineKeyframeMoveMessage, self._handle_keyframe_move
        )
        self._websock_interface.register_handler(
            _messages.TimelineKeyframeDeleteMessage, self._handle_keyframe_delete
        )
        self._websock_interface.register_handler(
            _messages.TimelineIntervalAddMessage, self._handle_interval_add
        )
        self._websock_interface.register_handler(
            _messages.TimelineIntervalMoveMessage, self._handle_interval_move
        )
        self._websock_interface.register_handler(
            _messages.TimelineIntervalDeleteMessage, self._handle_interval_delete
        )
        self._websock_interface.register_handler(
            _messages.TimelinePromptUpdateMessage, self._handle_prompt_update
        )
        self._websock_interface.register_handler(
            _messages.TimelinePromptResizeMessage, self._handle_prompt_resize
        )
        self._websock_interface.register_handler(
            _messages.TimelinePromptMoveMessage, self._handle_prompt_move
        )
        self._websock_interface.register_handler(
            _messages.TimelinePromptSwapMessage, self._handle_prompt_swap
        )
        self._websock_interface.register_handler(
            _messages.TimelinePromptDeleteMessage, self._handle_prompt_delete
        )
        self._websock_interface.register_handler(
            _messages.TimelinePromptAddMessage, self._handle_prompt_add
        )
        self._websock_interface.register_handler(
            _messages.TimelinePromptSplitMessage, self._handle_prompt_split
        )
        self._websock_interface.register_handler(
            _messages.TimelinePromptMergeMessage, self._handle_prompt_merge
        )

    def set_visible(self, visible: bool) -> None:
        """Show or hide the timeline UI.
        
        Args:
            visible: Whether to show the timeline.
        """
        self._enabled = visible
        self._has_been_used = True
        self._normalize_prompts(force_contiguous=True)
        self._send_timeline_update()

    def set_frame_range(self, start_frame: int, end_frame: int) -> None:
        """Set the frame range of the timeline.
        
        Args:
            start_frame: Starting frame number.
            end_frame: Ending frame number (determines initial viewport, not scroll limit).
        """
        self._start_frame = start_frame
        self._end_frame = end_frame
        
        self._send_timeline_update()


    def set_current_frame(self, frame: int) -> None:
        """Set the current frame position.
        
        Args:
            frame: Frame number to set as current.
        """
        self._current_frame = frame
        self._send_timeline_update()

    def configure_arrow_key_overlay(
        self,
        *,
        enabled: bool | None = None,
        highlighted: Sequence[str] | None = None,
        base_opacity: float | None = None,
        highlight_opacity: float | None = None,
        position: _messages.ArrowKeyOverlayPosition | None = None,
    ) -> None:
        """Configure the arrow-key overlay rendered above the timeline (UI-only).

        Args:
            enabled: Whether the arrow overlay should be shown.
            highlighted: Arrow keys to highlight (any subset of ArrowUp/Down/Left/Right).
            base_opacity: Opacity for non-highlighted keys (0..1).
            highlight_opacity: Opacity for highlighted keys (0..1).
            position: Overlay placement. One of:
                bottom_left, bottom_center, bottom_right, top_center, top_right.
        """
        if enabled is not None:
            self._arrow_overlay_enabled = bool(enabled)
        if highlighted is not None:
            # Normalize and validate.
            allowed = {"ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"}
            for k in highlighted:
                if k not in allowed:
                    raise ValueError(
                        f"Invalid arrow key '{k}'. Must be one of {sorted(allowed)}."
                    )
            self._arrow_overlay_highlighted = tuple(  # type: ignore[assignment]
                highlighted
            )
        if base_opacity is not None:
            self._arrow_overlay_base_opacity = float(base_opacity)
        if highlight_opacity is not None:
            self._arrow_overlay_highlight_opacity = float(highlight_opacity)
        if position is not None:
            allowed_positions = {
                "bottom_left",
                "bottom_center",
                "bottom_right",
                "top_center",
                "top_right",
            }
            if position not in allowed_positions:
                raise ValueError(
                    "Invalid arrow overlay position "
                    f"'{position}'. Must be one of {sorted(allowed_positions)}."
                )
            self._arrow_overlay_position = position

        self._websock_interface.queue_message(
            _messages.ArrowKeyOverlayMessage(
                enabled=self._arrow_overlay_enabled,
                highlighted=self._arrow_overlay_highlighted,
                base_opacity=self._arrow_overlay_base_opacity,
                highlight_opacity=self._arrow_overlay_highlight_opacity,
                position=self._arrow_overlay_position,
            )
        )

    def set_highlighted_arrow_keys(
        self, highlighted: Sequence[str] | None
    ) -> None:
        """Convenience wrapper to update highlighted arrow keys."""
        self.configure_arrow_key_overlay(
            highlighted=() if highlighted is None else highlighted
        )

    def set_defaults(
        self,
        default_text: str = "",
        default_duration: int = 30,
        *,
        min_duration: int = 1,
        max_duration: int | None = None,
        default_num_frames_zoom: int = 300,
        max_frames_zoom: int = 1000,
        fps: float = 30.0,
    ) -> None:
        """Set default split-text prompt settings and (if needed) initialize prompts.
        
        Splittext behavior is the default:
        - Starts with a single text prompt with a fixed duration
        - Users can Shift+click on a prompt to split it into two prompts at that frame
        - Text content can be edited for each prompt
        - Prompts cannot be moved or resized (except splittext-specific interactions), but can be split and deleted
        - The timeline can be scrolled infinitely to the right
        - Calling this again with a non-empty `default_text` will append a prompt using defaults.
        - Calling this again with empty `default_text` will extend the last prompt by `default_duration`.
        
        Args:
            default_text: Default text content for new prompts.
            default_duration: Default prompt duration in frames.
            min_duration: Minimum number of frames a prompt can have (default: 1).
            max_duration: Maximum number of frames a prompt can have (default: None = no limit).
        """
        self._has_been_used = True
        self._min_prompt_duration = max(1, int(min_duration))
        self._max_prompt_duration = int(max_duration) if max_duration is not None else None
        if self._max_prompt_duration is not None and self._max_prompt_duration < self._min_prompt_duration:
            raise ValueError("max_duration must be >= min_duration (or None).")

        self._default_prompt_text = default_text
        self._default_prompt_duration = int(default_duration)

        self._default_num_frames_zoom = max(1, int(default_num_frames_zoom))
        self._max_frames_zoom = max(self._default_num_frames_zoom, int(max_frames_zoom))

        fps_f = float(fps)
        if not (fps_f > 0.0):
            raise ValueError("fps must be > 0.")
        self._fps = fps_f

        # Ensure the timeline range supports the max zoom-out bound.
        # (Prompts can extend beyond, but this defines the zoom-out envelope and initial view.)
        self._end_frame = max(self._end_frame, self._max_frames_zoom)
        
        # If first time or no prompts exist, create a new one
        if len(self._prompts) == 0:
            start = self._start_frame
            uuid = _make_timeline_uuid()
            self._prompts[uuid] = _messages.TimelinePrompt(
                uuid=uuid,
                text=default_text,
                start_frame=start,
                end_frame=start + self._default_prompt_duration,
                color=None,  # Will use default color
            )
        else:
            # Find the last prompt and extend it or add new one
            sorted_prompts = sorted(self._prompts.values(), key=lambda p: p.end_frame)
            last_prompt = sorted_prompts[-1]
            
            if default_text:
                # Add new prompt after the last one
                new_id = _make_timeline_uuid()
                self._prompts[new_id] = _messages.TimelinePrompt(
                    uuid=new_id,
                    text=default_text,
                    start_frame=last_prompt.end_frame,
                    end_frame=last_prompt.end_frame + self._default_prompt_duration,
                    color=None,
                )
            else:
                # Extend the last prompt
                self._prompts[last_prompt.uuid] = _messages.TimelinePrompt(
                    uuid=last_prompt.uuid,
                    text=last_prompt.text,
                    start_frame=last_prompt.start_frame,
                    end_frame=last_prompt.end_frame + self._default_prompt_duration,
                    color=last_prompt.color,
                )
        
        self._send_timeline_update()

    def _pick_next_prompt_color(
        self,
        left_color: tuple[int, int, int] | None,
        right_color: tuple[int, int, int] | None,
    ) -> tuple[int, int, int]:
        color_index = self._next_color_index
        color = PROMPT_COLORS[color_index]
        attempts = 0

        while attempts < len(PROMPT_COLORS):
            if not _colors_equal(color, left_color) and not _colors_equal(
                color, right_color
            ):
                break
            color_index = (color_index + 1) % len(PROMPT_COLORS)
            color = PROMPT_COLORS[color_index]
            attempts += 1

        self._next_color_index = (color_index + 1) % len(PROMPT_COLORS)
        return color

    def _normalize_prompts(self, *, force_contiguous: bool = True) -> None:
        if not self._prompts:
            return

        min_dur = max(1, int(self._min_prompt_duration))
        max_dur = int(self._max_prompt_duration) if self._max_prompt_duration is not None else None

        sorted_prompts = sorted(
            self._prompts.values(), key=lambda p: (p.start_frame, p.end_frame)
        )
        normalized: dict[str, _messages.TimelinePrompt] = {}
        prev_end: int | None = None
        for prompt in sorted_prompts:
            start = min(prompt.start_frame, prompt.end_frame)
            end = max(prompt.start_frame, prompt.end_frame)
            duration = end - start

            if force_contiguous and prev_end is not None:
                start = prev_end
                end = start + max(duration, min_dur)

            if end - start < min_dur:
                end = start + min_dur
            if max_dur is not None and end - start > max_dur:
                end = start + max_dur

            normalized[prompt.uuid] = _messages.TimelinePrompt(
                uuid=prompt.uuid,
                text=prompt.text,
                start_frame=start,
                end_frame=end,
                color=prompt.color,
            )
            prev_end = end

        self._prompts = normalized

    def set_zoom_settings(
        self,
        *,
        default_num_frames_zoom: int | None = None,
        max_frames_zoom: int | None = None,
    ) -> None:
        """Update timeline zoom configuration without mutating prompts.

        This is useful when application state changes (e.g. loading a new motion)
        and we want the initial visible window to adapt, without changing the
        timeline's prompt content/durations.

        Args:
            default_num_frames_zoom: Desired initial visible frame window (>= 1).
            max_frames_zoom: Max number of frames visible when fully zoomed out (>= default_num_frames_zoom).
        """
        self._has_been_used = True

        if default_num_frames_zoom is not None:
            self._default_num_frames_zoom = max(1, int(default_num_frames_zoom))

        if max_frames_zoom is not None:
            self._max_frames_zoom = max(int(max_frames_zoom), self._default_num_frames_zoom)
        else:
            # Maintain the invariant max_frames_zoom >= default_num_frames_zoom.
            self._max_frames_zoom = max(self._max_frames_zoom, self._default_num_frames_zoom)

        # Ensure the zoom-out envelope is representable by the timeline range.
        self._end_frame = max(self._end_frame, self._max_frames_zoom)

        self._send_timeline_update()

    def set_fps(self, fps: float) -> None:
        """Set the frames-per-second timebase used for UI-only frame->seconds formatting."""
        fps_f = float(fps)
        if not (fps_f > 0.0):
            raise ValueError("fps must be > 0.")
        self._fps = fps_f
        self._send_timeline_update()

    def disable_constraints(self) -> None:
        """Disable constraint editing on the timeline.
        
        When disabled, the timeline shows a gray overlay and all add/delete/move/hover/drag
        operations for keyframes, intervals, and prompts are disabled. The timeline still
        displays all constraints but prevents user interactions with them.
        """
        self._constraints_enabled = False
        self._send_timeline_update()

    def enable_constraints(self) -> None:
        """Enable constraint editing on the timeline.
        
        When enabled (default), users can freely add, delete, move, and interact with
        keyframes, intervals, and prompts on the timeline.
        """
        self._constraints_enabled = True
        self._send_timeline_update()

    def on_frame_change(
        self, callback: Callable[[int], None]
    ) -> Callable[[int], None]:
        """Register a callback for when the current frame changes.
        
        This callback is triggered when the user clicks on a frame in the timeline UI.
        
        Args:
            callback: Function to call with the new frame number.
            
        Returns:
            The callback function (for convenience).
            
        Example:
            >>> @server.timeline.on_frame_change
            >>> def handle_frame_change(frame: int):
            >>>     print(f"Frame changed to: {frame}")
        """
        self._frame_change_cb.append(callback)
        return callback

    def on_keyframe_add(
        self, callback: Callable[[str, str, int], None]
    ) -> Callable[[str, str, int], None]:
        """Register a callback for when a keyframe is added.
        
        This callback is triggered when the user adds a keyframe in the timeline UI.
        
        Args:
            callback: Function to call with (keyframe_id, track_id, frame).
            
        Returns:
            The callback function (for convenience).
            
        Example:
            >>> @server.timeline.on_keyframe_add
            >>> def handle_keyframe_add(keyframe_id: str, track_id: str, frame: int):
            >>>     print(f"Keyframe {keyframe_id} added to track {track_id} at frame {frame}")
        """
        self._keyframe_add_cb.append(callback)
        return callback

    def on_keyframe_move(
        self, callback: Callable[[str, int], None]
    ) -> Callable[[str, int], None]:
        """Register a callback for when a keyframe is moved.
        
        This callback is triggered when the user drags a keyframe to a new frame.
        
        Args:
            callback: Function to call with (keyframe_id, new_frame).
            
        Returns:
            The callback function (for convenience).
            
        Example:
            >>> @server.timeline.on_keyframe_move
            >>> def handle_keyframe_move(keyframe_id: str, new_frame: int):
            >>>     print(f"Keyframe {keyframe_id} moved to frame {new_frame}")
        """
        self._keyframe_move_cb.append(callback)
        return callback

    def on_keyframe_delete(
        self, callback: Callable[[str], None]
    ) -> Callable[[str], None]:
        """Register a callback for when a keyframe is deleted.
        
        This callback is triggered when the user deletes a keyframe.
        
        Args:
            callback: Function to call with (keyframe_id,).
            
        Returns:
            The callback function (for convenience).
            
        Example:
            >>> @server.timeline.on_keyframe_delete
            >>> def handle_keyframe_delete(keyframe_id: str):
            >>>     print(f"Keyframe {keyframe_id} deleted")
        """
        self._keyframe_delete_cb.append(callback)
        return callback

    def on_interval_add(
        self, callback: Callable[[str, str, int, int], None]
    ) -> Callable[[str, str, int, int], None]:
        """Register a callback for when an interval is added.
        
        This callback is triggered when the user creates an interval in the timeline UI.
        
        Args:
            callback: Function to call with (interval_id, track_id, start_frame, end_frame).
            
        Returns:
            The callback function (for convenience).
            
        Example:
            >>> @server.timeline.on_interval_add
            >>> def handle_interval_add(interval_id: str, track_id: str, start_frame: int, end_frame: int):
            >>>     print(f"Interval {interval_id} added to track {track_id}: {start_frame}-{end_frame}")
        """
        self._interval_add_cb.append(callback)
        return callback

    def on_interval_move(
        self, callback: Callable[[str, int, int], None]
    ) -> Callable[[str, int, int], None]:
        """Register a callback for when an interval is moved or resized.
        
        This callback is triggered when the user drags or resizes an interval.
        
        Args:
            callback: Function to call with (interval_id, new_start_frame, new_end_frame).
            
        Returns:
            The callback function (for convenience).
            
        Example:
            >>> @server.timeline.on_interval_move
            >>> def handle_interval_move(interval_id: str, new_start: int, new_end: int):
            >>>     print(f"Interval {interval_id} moved to: {new_start}-{new_end}")
        """
        self._interval_move_cb.append(callback)
        return callback

    def on_interval_delete(
        self, callback: Callable[[str], None]
    ) -> Callable[[str], None]:
        """Register a callback for when an interval is deleted.
        
        This callback is triggered when the user deletes an interval.
        
        Args:
            callback: Function to call with (interval_id,).
            
        Returns:
            The callback function (for convenience).
            
        Example:
            >>> @server.timeline.on_interval_delete
            >>> def handle_interval_delete(interval_id: str):
            >>>     print(f"Interval {interval_id} deleted")
        """
        self._interval_delete_cb.append(callback)
        return callback

    def on_prompt_add(
        self, callback: Callable[[str, int, int, str, Tuple[int, int, int] | None], None]
    ) -> Callable[[str, int, int, str, Tuple[int, int, int] | None], None]:
        """Register a callback for when a prompt is added from the UI.

        Args:
            callback: Function to call with (prompt_id, start_frame, end_frame, text, color).
        """
        self._prompt_add_cb.append(callback)
        return callback

    def on_prompt_delete(
        self, callback: Callable[[str], None]
    ) -> Callable[[str], None]:
        """Register a callback for when a prompt is deleted from the UI.

        Args:
            callback: Function to call with (prompt_id,).
        """
        self._prompt_delete_cb.append(callback)
        return callback

    def on_prompt_update(
        self, callback: Callable[[str, str], None]
    ) -> Callable[[str, str], None]:
        """Register a callback for when a prompt's text is edited from the UI.

        Args:
            callback: Function to call with (prompt_id, new_text).
        """
        self._prompt_update_cb.append(callback)
        return callback

    def on_prompt_resize(
        self, callback: Callable[[str, int, int], None]
    ) -> Callable[[str, int, int], None]:
        """Register a callback for when a prompt is resized from the UI.

        Args:
            callback: Function to call with (prompt_id, new_start_frame, new_end_frame).
        """
        self._prompt_resize_cb.append(callback)
        return callback

    def on_prompt_move(
        self, callback: Callable[[str, int, int], None]
    ) -> Callable[[str, int, int], None]:
        """Register a callback for when a prompt is moved from the UI.

        Args:
            callback: Function to call with (prompt_id, new_start_frame, new_end_frame).
        """
        self._prompt_move_cb.append(callback)
        return callback

    def on_prompt_swap(
        self, callback: Callable[[str, str], None]
    ) -> Callable[[str, str], None]:
        """Register a callback for when two prompts are swapped from the UI.

        Args:
            callback: Function to call with (prompt_id_1, prompt_id_2).
        """
        self._prompt_swap_cb.append(callback)
        return callback

    def on_prompt_split(
        self, callback: Callable[[str, str, str, int], None]
    ) -> Callable[[str, str, str, int], None]:
        """Register a callback for when a prompt is split from the UI.

        Args:
            callback: Function to call with (original_prompt_id, left_prompt_id, right_prompt_id, split_frame).
        """
        self._prompt_split_cb.append(callback)
        return callback

    def on_prompt_merge(
        self, callback: Callable[[str, str], None]
    ) -> Callable[[str, str], None]:
        """Register a callback for when two adjacent prompts are merged from the UI.

        Args:
            callback: Function to call with (left_prompt_id, right_prompt_id).
        """
        self._prompt_merge_cb.append(callback)
        return callback

    def add_prompt(
        self,
        text: str,
        start_frame: int,
        end_frame: int,
        color: Optional[Tuple[int, int, int]] = None,
        uuid: Optional[str] = None,
    ) -> str:
        """Add a text prompt to the timeline.
        
        Args:
            text: Text content of the prompt.
            start_frame: Starting frame for this prompt.
            end_frame: Ending frame for this prompt.
            color: Optional RGB color tuple (0-255). If None, uses default blue color.
            uuid: Optional unique identifier. If None, one will be generated.
            
        Returns:
            The UUID of the added prompt.
        """
        if uuid is None:
            uuid = _make_timeline_uuid()
        
        prompt = _messages.TimelinePrompt(
            uuid=uuid,
            text=text,
            start_frame=start_frame,
            end_frame=end_frame,
            color=color,
        )
        self._prompts[uuid] = prompt
        self._normalize_prompts(force_contiguous=True)
        self._send_timeline_update()
        return uuid

    def update_prompt(
        self,
        uuid: str,
        text: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Update an existing prompt.
        
        Args:
            uuid: UUID of the prompt to update.
            text: New text content (None to keep unchanged).
            start_frame: New start frame (None to keep unchanged).
            end_frame: New end frame (None to keep unchanged).
            color: New RGB color (None to keep unchanged).
        """
        if uuid not in self._prompts:
            raise ValueError(f"Prompt with UUID {uuid} not found")
        
        prompt = self._prompts[uuid]
        self._prompts[uuid] = _messages.TimelinePrompt(
            uuid=uuid,
            text=text if text is not None else prompt.text,
            start_frame=start_frame if start_frame is not None else prompt.start_frame,
            end_frame=end_frame if end_frame is not None else prompt.end_frame,
            color=color if color is not None else prompt.color,
        )
        self._normalize_prompts(force_contiguous=True)
        self._send_timeline_update()

    def remove_prompt(self, uuid: str) -> None:
        """Remove a prompt from the timeline.
        
        Args:
            uuid: UUID of the prompt to remove.
        """
        if uuid in self._prompts:
            del self._prompts[uuid]
            self._normalize_prompts(force_contiguous=True)
            self._send_timeline_update()

    def clear_prompts(self) -> None:
        """Remove all prompts from the timeline."""
        self._prompts.clear()
        self._send_timeline_update()

    def add_track(
        self,
        name: str,
        track_type: str = "keyframe",
        color: Optional[Tuple[int, int, int]] = None,
        height_scale: float = 1.0,
        uuid: Optional[str] = None,
    ) -> str:
        """Add a track to the timeline.
        
        Args:
            name: Display name of the track (e.g., "FullBody", "Left Hand").
            track_type: Type of track - "keyframe" for pose targets, "prompt" for text.
            color: Optional RGB color tuple (0-255). If None, uses default color.
            height_scale: Height scale multiplier (default 1.0). Use 0.75 for 75% height, etc.
            uuid: Optional unique identifier. If None, one will be generated.
            
        Returns:
            The UUID of the added track.
        """
        if uuid is None:
            uuid = _make_timeline_uuid()
        
        track = _messages.TimelineTrack(
            uuid=uuid,
            name=name,
            track_type=track_type,
            color=color,
            height_scale=height_scale,
        )
        self._tracks[uuid] = track
        self._send_timeline_update()
        return uuid

    def remove_track(self, uuid: str) -> None:
        """Remove a track from the timeline.
        
        Also removes all keyframes associated with this track.
        
        Args:
            uuid: UUID of the track to remove.
        """
        if uuid in self._tracks:
            del self._tracks[uuid]
            # Remove all keyframes for this track
            self._keyframes = {
                k_id: k for k_id, k in self._keyframes.items()
                if k.track_id != uuid
            }
            self._send_timeline_update()

    def clear_tracks(self) -> None:
        """Remove all tracks and their keyframes from the timeline."""
        self._tracks.clear()
        self._keyframes.clear()
        self._send_timeline_update()

    def add_keyframe(
        self,
        track_id: str,
        frame: int,
        value: Optional[float] = None,
        opacity: float = 1.0,
        locked: bool = False,
        uuid: Optional[str] = None,
    ) -> str:
        """Add a keyframe marker to a track.
        
        Args:
            track_id: UUID of the track to add the keyframe to.
            frame: Frame number where the keyframe should appear.
            value: Optional value associated with the keyframe.
            opacity: Opacity of the keyframe (0.0 to 1.0). Default is 1.0.
            locked: Whether this keyframe is locked from interface modifications.
                   When True, the keyframe can only be modified via Python API.
            uuid: Optional unique identifier. If None, one will be generated.
            
        Returns:
            The UUID of the added keyframe.
            
        Raises:
            ValueError: If the track_id doesn't exist.
        """
        if track_id not in self._tracks:
            raise ValueError(f"Track with UUID {track_id} not found")
        
        if uuid is None:
            uuid = _make_timeline_uuid()
        
        keyframe = _messages.TimelineKeyframe(
            uuid=uuid,
            track_id=track_id,
            frame=frame,
            value=value,
            opacity=opacity,
            locked=locked,
        )
        self._keyframes[uuid] = keyframe
        self._send_timeline_update()
        return uuid
    
    def add_locked_keyframe(
        self,
        track_id: str,
        frame: int,
        opacity: float = 0.3,
        value: Optional[float] = None,
        uuid: Optional[str] = None,
    ) -> str:
        """Add a locked, transparent keyframe that cannot be modified via the UI.
        
        This is a convenience method for creating keyframes that:
        - Have transparency (default opacity 0.3)
        - Cannot be deleted via the interface
        - Cannot be moved via the interface
        - Can only be modified through Python API
        
        Args:
            track_id: UUID of the track to add the keyframe to.
            frame: Frame number where the keyframe should appear.
            opacity: Opacity of the keyframe (0.0 to 1.0). Default is 0.3.
            value: Optional value associated with the keyframe.
            uuid: Optional unique identifier. If None, one will be generated.
        
        Returns:
            The UUID of the added keyframe.
        
        Raises:
            ValueError: If the track_id doesn't exist.
            
        Example:
            >>> track_id = server.timeline.add_track("MyTrack")
            >>> # Add a locked background keyframe
            >>> keyframe_id = server.timeline.add_locked_keyframe(
            ...     track_id, frame=25, opacity=0.2
            ... )
        """
        return self.add_keyframe(
            track_id=track_id,
            frame=frame,
            value=value,
            opacity=opacity,
            locked=True,
            uuid=uuid,
        )

    def remove_keyframe(self, uuid: str) -> None:
        """Remove a keyframe from its track.
        
        Args:
            uuid: UUID of the keyframe to remove.
        """
        if uuid in self._keyframes:
            del self._keyframes[uuid]
            self._send_timeline_update()

    def clear_keyframes(self, track_id: Optional[str] = None) -> None:
        """Remove keyframes from the timeline.
        
        Args:
            track_id: If provided, only remove keyframes from this track.
                     If None, remove all keyframes.
        """
        if track_id is None:
            self._keyframes.clear()
        else:
            self._keyframes = {
                k_id: k for k_id, k in self._keyframes.items()
                if k.track_id != track_id
            }
        self._send_timeline_update()

    def _send_timeline_update(self) -> None:
        """Send the current timeline state to the client."""
        message = _messages.TimelineMessage(
            enabled=self._enabled,
            fps=self._fps,
            start_frame=self._start_frame,
            end_frame=self._end_frame,
            current_frame=self._current_frame,
            prompts=tuple(self._prompts.values()),
            tracks=tuple(self._tracks.values()),
            keyframes=tuple(self._keyframes.values()),
            intervals=tuple(self._intervals.values()),
            mode="splittext",
            constraints_enabled=self._constraints_enabled,
            min_prompt_duration=self._min_prompt_duration,
            max_prompt_duration=self._max_prompt_duration,
            default_text=self._default_prompt_text,
            default_duration=self._default_prompt_duration,
            default_num_frames_zoom=self._default_num_frames_zoom,
            max_frames_zoom=self._max_frames_zoom,
        )
        self._websock_interface.queue_message(message)

    async def _handle_timeline_update(
        self, client_id: ClientId, message: _messages.TimelineUpdateMessage
    ) -> None:
        """Handle timeline updates from the client (e.g., dragging the timeline)."""
        # Only process if this timeline has been explicitly used/configured
        # This prevents unused client timelines from interfering with server timelines
        if not self._has_been_used:
            return
        
        # Update the timeline state
        self._current_frame = message.current_frame
        self._send_timeline_update()
        
        # Call registered callbacks
        for cb in self._frame_change_cb:
            if asyncio.iscoroutinefunction(cb):
                await cb(self._current_frame)
            else:
                self._thread_executor.submit(
                    cb, self._current_frame
                ).add_done_callback(print_threadpool_errors)
    
    async def _handle_keyframe_add(
        self, client_id: ClientId, message: _messages.TimelineKeyframeAddMessage
    ) -> None:
        """Handle request to add a new keyframe."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Add keyframe (correct parameter order: track_id, frame)
        keyframe_id = self.add_keyframe(message.track_id, message.frame)
        # Note: add_keyframe already calls _send_timeline_update()
        
        # Call registered callbacks
        if keyframe_id:
            for cb in self._keyframe_add_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(keyframe_id, message.track_id, message.frame)
                else:
                    self._thread_executor.submit(
                        cb, keyframe_id, message.track_id, message.frame
                    ).add_done_callback(print_threadpool_errors)
    
    async def _handle_keyframe_move(
        self, client_id: ClientId, message: _messages.TimelineKeyframeMoveMessage
    ) -> None:
        """Handle request to move a keyframe."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Update keyframe frame number
        if message.keyframe_id in self._keyframes:
            old_kf = self._keyframes[message.keyframe_id]
            
            # Don't allow moving locked keyframes
            if old_kf.locked:
                return
            
            self._keyframes[message.keyframe_id] = _messages.TimelineKeyframe(
                uuid=old_kf.uuid,
                track_id=old_kf.track_id,
                frame=message.new_frame,
                value=old_kf.value,
                opacity=old_kf.opacity,
                locked=old_kf.locked,
            )
            self._send_timeline_update()
            
            # Call registered callbacks
            for cb in self._keyframe_move_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(message.keyframe_id, message.new_frame)
                else:
                    self._thread_executor.submit(
                        cb, message.keyframe_id, message.new_frame
                    ).add_done_callback(print_threadpool_errors)
    
    async def _handle_keyframe_delete(
        self, client_id: ClientId, message: _messages.TimelineKeyframeDeleteMessage
    ) -> None:
        """Handle request to delete a keyframe."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Remove keyframe if it exists and is not locked
        if message.keyframe_id in self._keyframes:
            # Don't allow deleting locked keyframes
            if self._keyframes[message.keyframe_id].locked:
                return
            
            self.remove_keyframe(message.keyframe_id)
            # Note: remove_keyframe already calls _send_timeline_update()
            
            # Call registered callbacks
            for cb in self._keyframe_delete_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(message.keyframe_id)
                else:
                    self._thread_executor.submit(
                        cb, message.keyframe_id
                    ).add_done_callback(print_threadpool_errors)
    
    async def _handle_interval_add(
        self, client_id: ClientId, message: _messages.TimelineIntervalAddMessage
    ) -> None:
        """Handle request to add an interval."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Add interval
        interval_id = self.add_interval(
            track_id=message.track_id,
            start_frame=message.start_frame,
            end_frame=message.end_frame,
            uuid=message.uuid,
        )
        self._send_timeline_update()
        
        # Call registered callbacks
        if interval_id:
            for cb in self._interval_add_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(interval_id, message.track_id, message.start_frame, message.end_frame)
                else:
                    self._thread_executor.submit(
                        cb, interval_id, message.track_id, message.start_frame, message.end_frame
                    ).add_done_callback(print_threadpool_errors)
    
    async def _handle_interval_move(
        self, client_id: ClientId, message: _messages.TimelineIntervalMoveMessage
    ) -> None:
        """Handle request to move an interval."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Update interval frames
        if message.interval_id in self._intervals:
            old_interval = self._intervals[message.interval_id]
            
            # Don't allow moving locked intervals
            if old_interval.locked:
                return
            
            self._intervals[message.interval_id] = _messages.TimelineInterval(
                uuid=old_interval.uuid,
                track_id=old_interval.track_id,
                start_frame=message.new_start_frame,
                end_frame=message.new_end_frame,
                value=old_interval.value,
                opacity=old_interval.opacity,
                locked=old_interval.locked,
            )
            self._send_timeline_update()
            
            # Call registered callbacks
            for cb in self._interval_move_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(message.interval_id, message.new_start_frame, message.new_end_frame)
                else:
                    self._thread_executor.submit(
                        cb, message.interval_id, message.new_start_frame, message.new_end_frame
                    ).add_done_callback(print_threadpool_errors)
    
    async def _handle_interval_delete(
        self, client_id: ClientId, message: _messages.TimelineIntervalDeleteMessage
    ) -> None:
        """Handle request to delete an interval."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Remove interval if it exists and is not locked
        if message.interval_id in self._intervals:
            # Don't allow deleting locked intervals
            if self._intervals[message.interval_id].locked:
                return
            
            self.remove_interval(message.interval_id)
            self._send_timeline_update()
            
            # Call registered callbacks
            for cb in self._interval_delete_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(message.interval_id)
                else:
                    self._thread_executor.submit(
                        cb, message.interval_id
                    ).add_done_callback(print_threadpool_errors)
    
    async def _handle_prompt_update(
        self, client_id: ClientId, message: _messages.TimelinePromptUpdateMessage
    ) -> None:
        """Handle request to update a prompt's text."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Update prompt text if it exists
        if message.prompt_id in self._prompts:
            old_prompt = self._prompts[message.prompt_id]
            self._prompts[message.prompt_id] = _messages.TimelinePrompt(
                uuid=old_prompt.uuid,
                text=message.new_text,
                start_frame=old_prompt.start_frame,
                end_frame=old_prompt.end_frame,
                color=old_prompt.color,
            )
            self._normalize_prompts(force_contiguous=True)
            self._send_timeline_update()

            # Call registered callbacks
            for cb in self._prompt_update_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(message.prompt_id, message.new_text)
                else:
                    self._thread_executor.submit(
                        cb, message.prompt_id, message.new_text
                    ).add_done_callback(print_threadpool_errors)
        else:
            logger.warning("Prompt update ignored; prompt %s not found.", message.prompt_id)
    
    async def _handle_prompt_resize(
        self, client_id: ClientId, message: _messages.TimelinePromptResizeMessage
    ) -> None:
        """Handle request to resize a prompt."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Update prompt frames if it exists
        if message.prompt_id in self._prompts:
            old_prompt = self._prompts[message.prompt_id]
            self._prompts[message.prompt_id] = _messages.TimelinePrompt(
                uuid=old_prompt.uuid,
                text=old_prompt.text,
                start_frame=message.new_start_frame,
                end_frame=message.new_end_frame,
                color=old_prompt.color,
            )
            self._normalize_prompts(force_contiguous=True)
            self._send_timeline_update()

            # Call registered callbacks
            for cb in self._prompt_resize_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(message.prompt_id, message.new_start_frame, message.new_end_frame)
                else:
                    self._thread_executor.submit(
                        cb, message.prompt_id, message.new_start_frame, message.new_end_frame
                    ).add_done_callback(print_threadpool_errors)
        else:
            logger.warning("Prompt resize ignored; prompt %s not found.", message.prompt_id)
    
    async def _handle_prompt_move(
        self, client_id: ClientId, message: _messages.TimelinePromptMoveMessage
    ) -> None:
        """Handle request to move a prompt."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Update prompt frames if it exists
        if message.prompt_id in self._prompts:
            old_prompt = self._prompts[message.prompt_id]
            self._prompts[message.prompt_id] = _messages.TimelinePrompt(
                uuid=old_prompt.uuid,
                text=old_prompt.text,
                start_frame=message.new_start_frame,
                end_frame=message.new_end_frame,
                color=old_prompt.color,
            )
            self._normalize_prompts(force_contiguous=True)
            self._send_timeline_update()

            # Call registered callbacks
            for cb in self._prompt_move_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(message.prompt_id, message.new_start_frame, message.new_end_frame)
                else:
                    self._thread_executor.submit(
                        cb, message.prompt_id, message.new_start_frame, message.new_end_frame
                    ).add_done_callback(print_threadpool_errors)
        else:
            logger.warning("Prompt move ignored; prompt %s not found.", message.prompt_id)
    
    async def _handle_prompt_swap(
        self, client_id: ClientId, message: _messages.TimelinePromptSwapMessage
    ) -> None:
        """Handle request to swap two prompts."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Swap the order of two prompts (they keep their own sizes)
        if message.prompt_id_1 in self._prompts and message.prompt_id_2 in self._prompts:
            prompt1 = self._prompts[message.prompt_id_1]
            prompt2 = self._prompts[message.prompt_id_2]
            
            # Calculate durations (each box keeps its own size)
            duration1 = prompt1.end_frame - prompt1.start_frame
            duration2 = prompt2.end_frame - prompt2.start_frame
            
            # Determine the overall block boundaries
            block_start = min(prompt1.start_frame, prompt2.start_frame)
            block_end = max(prompt1.end_frame, prompt2.end_frame)
            
            # Swap order: boxes switch left-to-right position but keep their sizes
            if prompt1.start_frame < prompt2.start_frame:
                # prompt1 was on left, moves to right
                new_prompt1_start = block_end - duration1
                new_prompt1_end = block_end
                new_prompt2_start = block_start
                new_prompt2_end = block_start + duration2
            else:
                # prompt1 was on right, moves to left
                new_prompt1_start = block_start
                new_prompt1_end = block_start + duration1
                new_prompt2_start = block_end - duration2
                new_prompt2_end = block_end
            
            # Constrain both prompts to timeline bounds
            if new_prompt1_start < self._start_frame:
                new_prompt1_start = self._start_frame
                new_prompt1_end = self._start_frame + duration1
            if new_prompt1_end > self._end_frame:
                new_prompt1_end = self._end_frame
                new_prompt1_start = self._end_frame - duration1
            
            if new_prompt2_start < self._start_frame:
                new_prompt2_start = self._start_frame
                new_prompt2_end = self._start_frame + duration2
            if new_prompt2_end > self._end_frame:
                new_prompt2_end = self._end_frame
                new_prompt2_start = self._end_frame - duration2
            
            # Update both prompts with their new positions
            self._prompts[message.prompt_id_1] = _messages.TimelinePrompt(
                uuid=prompt1.uuid,
                text=prompt1.text,
                start_frame=new_prompt1_start,
                end_frame=new_prompt1_end,
                color=prompt1.color,
            )
            self._prompts[message.prompt_id_2] = _messages.TimelinePrompt(
                uuid=prompt2.uuid,
                text=prompt2.text,
                start_frame=new_prompt2_start,
                end_frame=new_prompt2_end,
                color=prompt2.color,
            )
            self._send_timeline_update()

            # Call registered callbacks
            for cb in self._prompt_swap_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(message.prompt_id_1, message.prompt_id_2)
                else:
                    self._thread_executor.submit(
                        cb, message.prompt_id_1, message.prompt_id_2
                    ).add_done_callback(print_threadpool_errors)
    
    async def _handle_prompt_delete(
        self, client_id: ClientId, message: _messages.TimelinePromptDeleteMessage
    ) -> None:
        """Handle request to delete a prompt."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Prevent deleting the last remaining prompt.
        if len(self._prompts) <= 1:
            logger.warning(
                "Cannot delete the last remaining prompt in splittext timeline."
            )
            return
        
        # Delete the prompt if it exists
        if message.prompt_id in self._prompts:
            deleted_prompt = self._prompts[message.prompt_id]
            del self._prompts[message.prompt_id]
            
            # Splittext: shift all prompts to the right leftward to fill the gap.
            if len(self._prompts) > 0:
                # Get all prompts sorted by start frame
                sorted_prompts = sorted(self._prompts.values(), key=lambda p: p.start_frame)

                # Calculate the gap size
                gap_size = deleted_prompt.end_frame - deleted_prompt.start_frame

                # Find all prompts that come after the deleted prompt and shift them left
                for prompt in sorted_prompts:
                    if prompt.start_frame >= deleted_prompt.end_frame:
                        # This prompt is to the right of the deleted one - shift it left
                        self._prompts[prompt.uuid] = _messages.TimelinePrompt(
                            uuid=prompt.uuid,
                            text=prompt.text,
                            start_frame=prompt.start_frame - gap_size,
                            end_frame=prompt.end_frame - gap_size,
                            color=prompt.color,
                        )
            
            self._normalize_prompts(force_contiguous=True)
            self._send_timeline_update()

            # Call registered callbacks
            for cb in self._prompt_delete_cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(message.prompt_id)
                else:
                    self._thread_executor.submit(
                        cb, message.prompt_id
                    ).add_done_callback(print_threadpool_errors)
        else:
            logger.warning("Prompt delete ignored; prompt %s not found.", message.prompt_id)
    
    async def _handle_prompt_add(
        self, client_id: ClientId, message: _messages.TimelinePromptAddMessage
    ) -> None:
        """Handle request to add a new prompt."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Find the left and right neighbor prompts
        sorted_prompts = sorted(self._prompts.values(), key=lambda p: p.start_frame)
        left_neighbor = None
        right_neighbor = None
        for p in sorted_prompts:
            if p.end_frame == message.start_frame:
                left_neighbor = p
            if p.start_frame == message.end_frame:
                right_neighbor = p
        
        left_color = left_neighbor.color if left_neighbor else None
        right_color = right_neighbor.color if right_neighbor else None

        # In the UI, `None` is rendered using the default prompt color (PROMPT_COLORS[0]).
        # For color de-duplication, treat an explicit neighbor with color=None as that default.
        left_color_for_compare = _color_for_compare(left_color, left_neighbor is not None)
        right_color_for_compare = _color_for_compare(right_color, right_neighbor is not None)
        
        # Prefer a color provided by the client (keeps optimistic UI in sync),
        # but fall back to server-side color selection for backwards compatibility.
        color: tuple[int, int, int] | None = None
        if message.color is not None and len(message.color) == 3:
            try:
                candidate = (int(message.color[0]), int(message.color[1]), int(message.color[2]))
                # Clamp to valid 8-bit range.
                candidate = (max(0, min(255, candidate[0])),
                             max(0, min(255, candidate[1])),
                             max(0, min(255, candidate[2])))
                color = candidate
                # Keep server color cycling roughly aligned if candidate is one of our palette colors.
                if candidate in PROMPT_COLORS:
                    self._next_color_index = (PROMPT_COLORS.index(candidate) + 1) % len(PROMPT_COLORS)
            except Exception:
                color = None

        if color is None:
            color = self._pick_next_prompt_color(
                left_color_for_compare, right_color_for_compare
            )
        
        # Create a new prompt
        uuid = _make_timeline_uuid()
        new_prompt = _messages.TimelinePrompt(
            uuid=uuid,
            text=message.text,
            start_frame=min(message.start_frame, message.end_frame),
            end_frame=max(message.start_frame, message.end_frame),
            color=color,
        )
        self._prompts[uuid] = new_prompt
        self._normalize_prompts(force_contiguous=True)
        self._send_timeline_update()

        # Call registered callbacks
        for cb in self._prompt_add_cb:
            if asyncio.iscoroutinefunction(cb):
                await cb(uuid, new_prompt.start_frame, new_prompt.end_frame, new_prompt.text, new_prompt.color)
            else:
                self._thread_executor.submit(
                    cb, uuid, new_prompt.start_frame, new_prompt.end_frame, new_prompt.text, new_prompt.color
                ).add_done_callback(print_threadpool_errors)
    
    async def _handle_prompt_split(
        self, client_id: ClientId, message: _messages.TimelinePromptSplitMessage
    ) -> None:
        """Handle request to split a prompt at a given frame."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Check if the prompt exists
        if message.prompt_id not in self._prompts:
            logger.warning("Prompt split ignored; prompt %s not found.", message.prompt_id)
            return
        
        # Get the original prompt
        original_prompt = self._prompts[message.prompt_id]
        
        # Validate split frame is within prompt range.
        split_frame = int(message.split_frame)
        if split_frame <= original_prompt.start_frame or split_frame >= original_prompt.end_frame:
            logger.warning(
                "Prompt split ignored; split frame %s is outside prompt range.",
                split_frame,
            )
            return  # Invalid split frame

        min_dur = max(1, int(self._min_prompt_duration))

        # Ensure the LEFT segment meets min duration by clamping the split point.
        if split_frame - original_prompt.start_frame < min_dur:
            split_frame = original_prompt.start_frame + min_dur

        # Ensure the RIGHT segment meets min duration.
        # If it's too short, extend the original prompt end and shift all subsequent prompts right.
        original_end = original_prompt.end_frame
        if original_end - split_frame < min_dur:
            new_end = split_frame + min_dur
            shift = new_end - original_end
            if shift > 0:
                # Shift all prompts that start at/after the original end to the right, preserving durations.
                for p in sorted(self._prompts.values(), key=lambda p: p.start_frame, reverse=True):
                    if p.uuid == original_prompt.uuid:
                        continue
                    if p.start_frame >= original_end:
                        self._prompts[p.uuid] = _messages.TimelinePrompt(
                            uuid=p.uuid,
                            text=p.text,
                            start_frame=p.start_frame + shift,
                            end_frame=p.end_frame + shift,
                            color=p.color,
                        )
                # Update our local view of the original prompt end.
                original_end = new_end

        # Respect max_duration by clamping end if configured (note: may reduce right segment if user
        # asked for something too large; we still guarantee min_dur above).
        if self._max_prompt_duration is not None:
            max_dur = int(self._max_prompt_duration)
            max_end = original_prompt.start_frame + max_dur
            original_end = min(original_end, max_end)
            if original_end - split_frame < min_dur:
                # If clamping would violate min duration, pull the split left.
                split_frame = max(original_prompt.start_frame + min_dur, original_end - min_dur)
        
        right_neighbor = None
        for p in self._prompts.values():
            if p.uuid == original_prompt.uuid:
                continue
            if p.start_frame == original_prompt.end_frame:
                right_neighbor = p
                break
        
        left_color = original_prompt.color
        right_neighbor_color = right_neighbor.color if right_neighbor else None

        # In the UI, `None` is rendered using the default prompt color (PROMPT_COLORS[0]).
        # For color de-duplication, treat an explicit neighbor with color=None as that default.
        left_color_for_compare = _color_for_compare(left_color, True)
        right_neighbor_color_for_compare = _color_for_compare(
            right_neighbor_color, right_neighbor is not None
        )

        right_color = self._pick_next_prompt_color(
            left_color_for_compare, right_neighbor_color_for_compare
        )

        # Create two new prompts
        # First prompt: from original start to split frame
        first_uuid = _make_timeline_uuid()
        first_prompt = _messages.TimelinePrompt(
            uuid=first_uuid,
            text=original_prompt.text,
            start_frame=original_prompt.start_frame,
            end_frame=split_frame,
            color=original_prompt.color,
        )
        
        # Second prompt: from split frame to original end
        second_uuid = _make_timeline_uuid()
        second_prompt = _messages.TimelinePrompt(
            uuid=second_uuid,
            text=original_prompt.text,
            start_frame=split_frame,
            end_frame=original_end,
            color=right_color,
        )
        
        # Remove the original prompt and add the two new prompts
        del self._prompts[message.prompt_id]
        self._prompts[first_uuid] = first_prompt
        self._prompts[second_uuid] = second_prompt
        
        self._normalize_prompts(force_contiguous=True)
        self._send_timeline_update()

        # Call registered callbacks
        for cb in self._prompt_split_cb:
            if asyncio.iscoroutinefunction(cb):
                await cb(message.prompt_id, first_uuid, second_uuid, split_frame)
            else:
                self._thread_executor.submit(
                    cb, message.prompt_id, first_uuid, second_uuid, split_frame
                ).add_done_callback(print_threadpool_errors)
    
    async def _handle_prompt_merge(
        self, client_id: ClientId, message: _messages.TimelinePromptMergeMessage
    ) -> None:
        """Handle request to merge two adjacent prompts."""
        # Only process if this timeline has been explicitly used/configured
        if not self._has_been_used:
            return
        
        # Check if both prompts exist
        if message.prompt_id_left not in self._prompts or message.prompt_id_right not in self._prompts:
            logger.warning(
                "Prompt merge ignored; missing prompt(s): %s, %s.",
                message.prompt_id_left,
                message.prompt_id_right,
            )
            return
        
        # Get both prompts
        left_prompt = self._prompts[message.prompt_id_left]
        right_prompt = self._prompts[message.prompt_id_right]
        
        # Verify they are adjacent
        if left_prompt.end_frame != right_prompt.start_frame:
            logger.warning(
                "Prompt merge ignored; prompts are not adjacent: %s -> %s.",
                left_prompt.end_frame,
                right_prompt.start_frame,
            )
            return  # Not adjacent
        
        # Create merged prompt (keeping left prompt's text and extending to right's end)
        merged_prompt = _messages.TimelinePrompt(
            uuid=left_prompt.uuid,
            text=left_prompt.text,
            start_frame=left_prompt.start_frame,
            end_frame=right_prompt.end_frame,
            color=left_prompt.color,
        )
        
        # Update left prompt and delete right prompt
        self._prompts[left_prompt.uuid] = merged_prompt
        del self._prompts[message.prompt_id_right]
        
        self._normalize_prompts(force_contiguous=True)
        self._send_timeline_update()

        # Call registered callbacks
        for cb in self._prompt_merge_cb:
            if asyncio.iscoroutinefunction(cb):
                await cb(message.prompt_id_left, message.prompt_id_right)
            else:
                self._thread_executor.submit(
                    cb, message.prompt_id_left, message.prompt_id_right
                ).add_done_callback(print_threadpool_errors)
    
    def add_interval(
        self,
        track_id: str,
        start_frame: int,
        end_frame: int,
        value: Optional[float] = None,
        opacity: float = 1.0,
        locked: bool = False,
        uuid: Optional[str] = None,
    ) -> str:
        """Add an interval constraint to a track.
        
        Args:
            track_id: UUID of the track to add the interval to.
            start_frame: Start frame of the interval (inclusive).
            end_frame: End frame of the interval (inclusive).
            value: Optional value associated with this interval.
            opacity: Opacity of the interval (0.0 to 1.0). Default is 1.0.
            locked: Whether this interval is locked from interface modifications.
                   When True, the interval can only be modified via Python API.
            uuid: Optional UUID for the interval. If not provided, one will be generated.
        
        Returns:
            UUID of the created interval.
        
        Raises:
            ValueError: If the track does not exist.
        """
        if track_id not in self._tracks:
            raise ValueError(f"Track with UUID {track_id} not found")
        
        if uuid is None:
            uuid = _make_timeline_uuid()
        
        interval = _messages.TimelineInterval(
            uuid=uuid,
            track_id=track_id,
            start_frame=min(start_frame, end_frame),
            end_frame=max(start_frame, end_frame),
            value=value,
            opacity=opacity,
            locked=locked,
        )
        self._intervals[uuid] = interval
        self._send_timeline_update()
        return uuid
    
    def add_locked_interval(
        self,
        track_id: str,
        start_frame: int,
        end_frame: int,
        opacity: float = 0.3,
        value: Optional[float] = None,
        uuid: Optional[str] = None,
    ) -> str:
        """Add a locked, transparent interval that cannot be modified via the UI.
        
        This is a convenience method for creating intervals that:
        - Have transparency (default opacity 0.3)
        - Cannot be deleted via the interface
        - Cannot be resized via the interface
        - Cannot be moved via the interface
        - Can only be modified through Python API
        
        Args:
            track_id: UUID of the track to add the interval to.
            start_frame: Start frame of the interval (inclusive).
            end_frame: End frame of the interval (inclusive).
            opacity: Opacity of the interval (0.0 to 1.0). Default is 0.3.
            value: Optional value associated with this interval.
            uuid: Optional UUID for the interval. If not provided, one will be generated.
        
        Returns:
            UUID of the created interval.
        
        Raises:
            ValueError: If the track does not exist.
            
        Example:
            >>> track_id = server.timeline.add_track("MyTrack")
            >>> # Add a locked background interval
            >>> interval_id = server.timeline.add_locked_interval(
            ...     track_id, start_frame=10, end_frame=50, opacity=0.2
            ... )
        """
        return self.add_interval(
            track_id=track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            value=value,
            opacity=opacity,
            locked=True,
            uuid=uuid,
        )
    
    def remove_interval(self, uuid: str) -> None:
        """Remove an interval from its track.
        
        Args:
            uuid: UUID of the interval to remove.
        """
        if uuid in self._intervals:
            del self._intervals[uuid]
            self._send_timeline_update()

    def clear_intervals(self, track_id: Optional[str] = None) -> None:
        """Remove intervals from the timeline.
        
        Args:
            track_id: If provided, only remove intervals from this track.
                     If None, remove all intervals.
        """
        if track_id is None:
            self._intervals.clear()
        else:
            self._intervals = {
                i_id: i for i_id, i in self._intervals.items()
                if i.track_id != track_id
            }
        self._send_timeline_update()
