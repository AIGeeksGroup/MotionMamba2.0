"""Remote generation adapter for the original ARDY InteractiveTimelineDemo.

No web server or GUI is created here. The training environment owns weights,
RNG and persistent memory; this process retains the original renderer/bridge.
"""
import json
from contextlib import nullcontext
import hashlib
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import zmq

from ardy.motion_rep import ArdyMotionRep
from ardy.skeleton import G1Skeleton34
from .reference_playback_clock import ReferencePlaybackClock
from .disk_motion_history import DiskMotionHistory
from .control_history import ControlHistory


class RemoteMotionMambaModel:
    is_remote_mm2 = True
    # Prepare one unpublished window early; keep the existing publication
    # frontier so prefetching does not add latency to text/control changes.
    prefetch_frames = 104
    publish_ahead_frames = 52

    def __init__(self, encoder):
        torch.set_num_threads(4)
        self.url = os.environ.get("MM2_GENERATOR_URL", "http://127.0.0.1:8766")
        health = self.api("/health")
        self.inference_backend = health.get('inference_backend', 'reference')
        if (health["frames_per_window"], health["fps"]) != (52, 25):
            raise ValueError("Generator does not meet G1 52-frame / 25 FPS contract")
        self.checkpoint = health["checkpoint"]
        stats_dir = Path(os.environ.get("MM2_STATS_DIR") or health.get(
            "stats_dir", str(Path(__file__).resolve().parents[3]/"simulation/assets/stats")))
        if health.get("stats_sha256"):
            fingerprint = hashlib.sha256()
            for name in ("mean.npy", "std.npy"):
                fingerprint.update(name.encode())
                fingerprint.update((stats_dir/name).read_bytes())
            if fingerprint.hexdigest() != health["stats_sha256"]:
                raise ValueError("Viewer normalization differs from generator checkpoint")
        self.motion_rep = ArdyMotionRep(G1Skeleton34(), 25, stats_path=str(stats_dir))
        self.model_label = os.environ.get("MM2_MODEL_LABEL", health.get("model_label", "Motion Mamba 2.0"))
        self.skeleton = self.motion_rep.skeleton
        self.denoiser = SimpleNamespace(num_frames_per_token=4)
        self.diffusion = SimpleNamespace(num_base_steps=10)
        self.gen_horizon_len = 52
        self.text_encoder = encoder
        self.sid = None
        self.segments = []
        self.control_history = ControlHistory()
        self.velocity_preview = None
        self.window = 0
        self.revision = 0
        self.limit = int(os.environ.get("MM2_SESSION_WINDOWS", "0"))
        self.intent_lock = threading.RLock()
        # Seek/restart and generation must observe one committed motion frontier.
        # Keep input receipt independent so new text can cancel in-flight work.
        self.motion_update_lock = threading.RLock()
        self.generation_lock = threading.RLock()
        self.telemetry_lock = threading.RLock()
        self.telemetry_epoch = None
        self.telemetry_ready = False
        self._control_timer = None
        self._last_control = None
        self.intent_version = 0
        self.last_error = None
        self.error_kind = None
        self.draft = None
        self.prepare_requested = False
        self.planning = False
        self.sequential_preload = False
        self.planning_frame = 0
        self.plan_until = 0
        self.consumed_frame = 0.0
        self.scheduled_prompts = {}
        self.live_replan_requested = False
        self.replan_target = None
        self.live_replanning = bool(health.get('live_replanning'))
        self.replan_budgets = {0:1.2, 1:max(1.6,float(health.get('live_replan_prepare_s', .7))+.8)}
        self.output = Path(os.environ.get("MM2_VIEWER_OUTPUT",
            str(Path(__file__).resolve().parents[3]/"runtime/original_viewer")))
        self.output.mkdir(parents=True, exist_ok=True)
        self.failed = False
        self.completed_at_reference_frame = None
        self.actual = {}
        self.reference = {}
        self.reference_clock = ReferencePlaybackClock()

    def api(self, path, data=None):
        request = urllib.request.Request(self.url + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"})
        attempts = 2 if path in ('/next', '/publish', '/replan/accept') else 1
        for attempt in range(attempts):
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=180) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(exc.read().decode()) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt+1 == attempts:
                    raise

    @staticmethod
    def notice(session, message, color="orange"):
        session.client.add_notification(title="Motion Mamba 2.0", body=message,
                                        color=color, auto_close_seconds=8)

    @property
    def reference_preview(self):
        # The original ARDY UI has one timeline, not selectable playback modes.
        return False

    def configure(self, demo, session):
        g = session.gui_elements
        g.gui_chosen_model_md.content = f"**Model:** {self.model_label} (G1 / 52)"
        acceleration_label = ('CUDA Graph (optimized)'
                              if self.inference_backend == 'optimized' else 'None')
        g.gui_compile_mode.options = (acceleration_label,)
        g.gui_compile_mode.value = acceleration_label
        g.gui_history_crop_length.min = 0
        g.gui_history_crop_length.value = 0
        # Preserve all original controls and layout, but do not imply that
        # unsupported sampler/constraint settings affect the remote model.
        for name in ("gui_skeleton_dropdown", "gui_horizon_dropdown", "gui_compile_mode",
                     "gui_diffusion_steps_slider", "gui_cfg_text_weight", "gui_cfg_constraint_weight",
                     "gui_history_crop_length", "gui_future_crop_length"):
            getattr(g, name).disabled = True
        g.gui_replan_trigger_thresh.value = 100
        self.prompt = g.gui_prompt_text.value
        self.received_prompt = self.prompt
        if demo.sonic_output and self.owns_bridge(demo, session):
            demo.sonic_output.catch_up = False
            demo.sonic_output.source_control = lambda action: self.source_playback_control(demo, session, action)
            with demo.sonic_output._lock:
                demo.sonic_output._state.epoch = int(time.time() * 1000)

    def source_playback_control(self, demo, session, action):
        """SONIC buttons use the same owner and accepted history as the viewer."""
        with self.generation_lock, self.motion_update_lock:
            client_id = session.client.client_id
            if (not demo.client_active(client_id) or not self.sid or not self.window
                    or getattr(demo, 'mm2_physical_owner', None) != client_id):
                return False
            if action == 'restart':
                demo.set_playback_state(client_id, False, 'bridge_restart')
                self.planning = self.sequential_preload = False
                self.consumed_frame = 0.0
                self.seek_playback(demo, session, 0, force=True)
                demo._anchor_playback(session, 0)
                demo.set_frame(client_id, 0, render_only=True)
            else:
                demo.set_playback_state(client_id, action == 'play')
            return True

    def owns_bridge(self, demo, session):
        if not demo.sonic_output or self.reference_preview:
            return True
        owner = getattr(demo, 'mm2_physical_owner', None)
        if owner not in demo.client_sessions:
            demo.mm2_physical_owner = session.client.client_id
            return True
        if owner == session.client.client_id:
            return True
        self.notice(session, '物理仿真由另一连接控制；请关闭该连接后重试。')
        return False

    def event(self, name, **fields):
        if hasattr(self, 'run_dir'):
            record = dict(event=name, wall_time=time.time(), monotonic=time.monotonic(),
                          session=self.sid, revision=self.revision, request_version=self.intent_version, **fields)
            self.write_trace('events.jsonl', record)

    def write_trace(self, filename, record):
        """Diagnostics must never terminate the playback/control worker."""
        folder = getattr(self, 'run_dir', None)
        if folder is None:
            return
        try:
            with (folder/filename).open('a') as stream:
                stream.write(json.dumps(record)+'\n')
        except OSError as exc:
            now = time.monotonic()
            if now-getattr(self, '_trace_error_time', -float('inf')) > 5:
                self._trace_error_time = now
                print(f'[MM2 trace] {filename}: {exc}; playback continues', flush=True)

    def close(self):
        if self._control_timer is not None:
            self._control_timer.cancel()
            self._control_timer = None
        if self.sid:
            sid, self.sid = self.sid, None
            try:
                self.api("/close", dict(session=sid))
            except Exception as exc:
                print(f"[MM2] close failed: {exc}", flush=True)

    def reset(self, session):
        with self.generation_lock, self.motion_update_lock:
            return self._reset_impl(session)

    def _reset_impl(self, session):
        self.close()
        self.reference_clock.reset()
        result = self.api("/session", dict(seed=int(session.gui_elements.gui_seed.value), windows=self.limit,
            translation=np.asarray(session.init_global_translation).reshape(3).tolist(),
            heading=float(session.init_first_heading_angle),owner_pid=os.getpid()))
        sid = result["session"]
        run_dir = self.output / sid
        run_dir.mkdir(parents=True, exist_ok=True)
        # Publish only an existing directory to the concurrently running logger.
        self.run_dir = run_dir
        self.sid = sid
        self.window = self.revision = 0
        self.failed = False
        self.last_error = self.error_kind = None
        self.draft = None
        self.prepare_requested = False
        self.completed_at_reference_frame = None
        self.consumed_frame = 0.0
        self.live_replan_requested = False
        self.replan_target = None
        self.planning = False
        self.sequential_preload = False
        self.planning_frame = 0
        self.plan_until = 0
        self.scheduled_prompts = {}
        self.prompt = session.gui_elements.gui_prompt_text.value
        self.received_prompt = self.prompt
        self.segments = []
        self.control_history = ControlHistory()
        self.velocity_preview = None
        self.reference = {}
        self._last_control = None
        session.mm2_waypoint_route = None
        session.mm2_live_waypoint = None
        print(f"[MM2] session={self.sid} seed={session.gui_elements.gui_seed.value} memory reset", flush=True)

    def begin_planning(self, demo, session, sequential):
        # Paused timeline editing is implicit, as in the reproduced ARDY UI.
        if not self.planning or self.sequential_preload != sequential:
            demo.set_playback_state(session.client.client_id, False, 'user')
            self.planning = True
            self.sequential_preload = sequential
            self.planning_frame = max(0, session.frame_idx)
            self.plan_until = self.window * 52
        frontier = max(0, session.max_frame_idx)
        limit = frontier if sequential else max(frontier, int(self.consumed_frame)+1500)
        session.gui_elements.gui_frame_idx_input.max = limit
        if hasattr(session.client, 'timeline'):
            session.client.timeline.set_frame_range(start_frame=0, end_frame=limit)

    def select_planning_frame(self, demo, session, requested):
        """Clamp to loaded frames; an explicit frontier request loads one window."""
        sequential = getattr(self, 'sequential_preload', False)
        frontier = max(0, session.max_frame_idx)
        limit = frontier if sequential else int(self.consumed_frame) + 1500
        frame = max(0, min(int(requested), limit))
        self.planning_frame = frame
        target = ((self.window+1)*52 if sequential and int(requested)>=frontier
                  else self.window*52 if sequential else (frame//52+1)*52)
        self.plan_until = max(self.plan_until, target)
        self.prepare_requested = self.window * 52 < self.plan_until
        self.event('planning_selected', requested_frame=int(requested),
                   planning_frame=frame, plan_until=self.plan_until)
        if frame <= session.max_frame_idx:
            demo.set_frame(session.client.client_id, frame)
        else:
            session.gui_elements.gui_frame_idx_input.value = frame
        if self.prepare_requested:
            threading.Thread(target=demo.on_replan_trigger,
                args=(session.client.client_id,), kwargs={'skip_if_busy': True}, daemon=True).start()

    def end_planning(self, demo, session):
        self.planning = False
        self.sequential_preload = False
        session.gui_elements.gui_frame_idx_input.max = max(0,session.max_frame_idx)
        # Play starts at the selected cursor, including an explicit frame zero.
        self.reference_clock.reset()

    def seek_playback(self, demo, session, frame, *, force=False):
        with self.generation_lock, self.motion_update_lock:
            return self._seek_playback_impl(demo, session, frame, force=force)

    def _seek_playback_impl(self, demo, session, frame, *, force=False):
        """Replay saved generated motion, without re-running or resetting the model."""
        self.velocity_preview = None
        # A subsequent keyboard edit starts from this frame's direction, rather
        # than the final direction left in the live planning input.
        session.mm2_keyboard_edit_frame = int(frame)
        pressed = getattr(session, 'mm2_arrow_keys_pressed', None)
        if pressed:
            pressed.clear()
            session.client.timeline.set_highlighted_arrow_keys([])
        bridge = demo.sonic_output
        if bridge is None or not self.window or not self.owns_bridge(demo, session):
            return
        target = int(frame) * 2
        with bridge._lock:
            if not force and bridge._state.qpos is not None and bridge._state.frame == target:
                return
        qpos = np.concatenate([np.load(self.run_dir/f'qpos_w{i:06d}.npy') for i in range(self.window)])
        bridge.restore_source_playback(qpos, source_frame=int(frame),
            name='Motion Mamba 2.0: ' + self.prompt_at(int(frame)), playing=session.playing,
            stream_complete=bool(self.limit and self.window == self.limit))
        self.reference_clock.reset()
        self.event('playback_seek', source_frame=int(frame), generated_frames=len(qpos))

    def target_velocity_at(self, session, frame):
        """Render what was accepted at this frame; seeking never changes intent."""
        if self.window:
            return self.control_history.at(frame)
        g = session.gui_elements
        return tuple(g.gui_target_root_velocity.value) if g.gui_use_target_velocity_checkbox.value else None

    def display_target_velocity_at(self, session, frame):
        """Live input stays visible while replanning; seeking clears its preview."""
        preview = getattr(self, 'velocity_preview', None)
        return preview if preview is not None else self.target_velocity_at(session, frame)

    def record_control(self, frame, control):
        self.control_history.replace_tail(frame, control)
        self.event('control_accepted', source_start=int(frame), control=control)

    def prompt_at(self, frame):
        values = [(k, v) for k, v in getattr(self, 'scheduled_prompts', {}).items() if k <= frame]
        return max(values)[1] if values else self.prompt

    def update_prompt(self, demo, session, trigger_replan=True):
        prompt = session.gui_elements.gui_prompt_text.value.strip()
        if not prompt:
            return self.notice(session, "请输入非空文本。")
        if self.window and not session.playing and not self.planning:
            self.begin_planning(demo, session, True)
        # Receipt must not wait for encoding or the generation/replan lock.
        # The generator encodes exactly the prompt captured with this version.
        with self.intent_lock:
            self.intent_version += 1
            self.prepare_requested = True
            self.received_prompt = prompt
            if self.planning:
                target = ((self.planning_frame + 1 + 3)//4)*4
                self.scheduled_prompts[target] = prompt
                self.replan_target = target if target < self.window*52 else None
                self.live_replan_requested = self.replan_target is not None
                self.plan_until = max(self.plan_until, (target//52+1)*52)
            else:
                self.prompt = prompt
                self.scheduled_prompts = {}
                self.replan_target = None
                self.live_replan_requested = self.live_replanning and self.window > 0
            self.failed = False
            self.last_error = self.error_kind = None
            session.gui_elements.gui_active_prompt_label.content = f"**Active Prompt:** {prompt}"
            boundary = target if self.planning else self.window * 52
            self.sync_timeline(demo, session, pending=prompt, pending_start=boundary)
            self.event('input_received', prompt=prompt, earliest_source_frame=boundary, playing=session.playing)
            print(f"[MM2] prompt boundary={boundary}: {prompt}", flush=True)
        if trigger_replan:
            threading.Thread(target=demo.on_replan_trigger, args=(session.client.client_id,), daemon=True).start()

    def sync_timeline(self, demo, session, pending=None, pending_start=None):
        if not hasattr(session.client, "timeline"):
            return
        # Input callbacks and generation completion can update this together.
        # Serialize the replacement and send only its final state to Viser.
        with self.intent_lock:
            atomic = getattr(session.client, 'atomic', nullcontext)
            with atomic():
                self._sync_timeline(demo, session, pending, pending_start)

    def _sync_timeline(self, demo, session, pending=None, pending_start=None):
        # Removing only the last recorded UUID list leaves orphan old labels
        # after overlapping updates. The generation segments are authoritative.
        session.client.timeline.clear_prompts()
        segments = [dict(s) for s in self.segments]
        if pending:
            start = self.window*52 if pending_start is None else pending_start
            segments = [dict(s,end_frame=min(s['end_frame'],start-1))
                        for s in segments if s['start_frame'] < start]
            segments.append(dict(text=pending, start_frame=start, end_frame=99999))
        elif segments:
            segments[-1]["end_frame"] = 99999
            for frame,prompt in sorted(self.scheduled_prompts.items()):
                if frame >= self.window*52:
                    segments[-1]['end_frame']=frame-1
                    segments.append(dict(text=prompt,start_frame=frame,end_frame=99999))
        if getattr(self, 'sequential_preload', False):
            end = max(0, session.max_frame_idx)
            segments = [dict(s, end_frame=min(s['end_frame'], end))
                        for s in segments if s['start_frame'] <= end]
        ids = []
        for i, segment in enumerate(segments):
            # Generation intervals include their last frame; Viser normalizes
            # adjacent prompt intervals with an exclusive end.
            visible = dict(segment, end_frame=segment['end_frame'] + 1)
            ids.append(session.client.timeline.add_prompt(**visible, color=demo.get_prompt_color(i)))
        session.timeline_data["prompt_uuid_list"] = ids
        session.timeline_data["prompt_counter"] = len(ids)

    def needs_generation(self, session):
        if self.failed or session.stop_playback or (self.limit and self.window >= self.limit):
            return False
        if getattr(self, 'live_replan_requested', False):
            return True
        if getattr(self, 'planning', False):
            return self.window * 52 < self.plan_until
        ahead = self.window * 52 - session.frame_idx
        return ahead <= self.prefetch_frames and (
            self.draft is None or self.prepare_requested or ahead <= self.publish_ahead_frames)

    def generate(self, demo, session):
        # Serialize model/RNG/history transactions with seek/reset, while
        # leaving the renderer and consumer clock running during inference.
        with self.generation_lock:
            return self._generate_impl(demo, session)

    def _generate_impl(self, demo, session):
        if self.sid is None:
            return
        if self.failed or session.stop_playback or (self.limit and self.window >= self.limit):
            return
        if not self.owns_bridge(demo, session):
            return
        if getattr(self, 'live_replan_requested', False):
            return self.apply_live_replan(demo, session)
        # Cursor callbacks can already be queued behind the replan lock.
        # Recheck the requested frontier here, not only in the polling loop:
        # stale triggers must never publish past a completed paused plan.
        if getattr(self, 'planning', False) and self.window * 52 >= self.plan_until:
            self.prepare_requested = False
            return
        # Do not turn repeated Replan presses into an unbounded old prefix.
        if (not getattr(self, 'planning', False) and self.window and not self.prepare_requested and (
                self.window * 52 - session.frame_idx > self.prefetch_frames
                or (self.draft is not None and self.window * 52 - session.frame_idx > self.publish_ahead_frames))):
            return
        try:
            for _ in range(4):  # coalesce stale in-flight drafts, at most 4 here
                if session.stop_playback:
                    return
                started = time.monotonic()
                with self.intent_lock:
                    prompt, version = self.prompt_at(self.window*52), self.intent_version
                    control = self.controls(session)
                self.event('generation_requested', window=self.window, control=control)
                if self.draft is not None and (self.draft['version'] != version or self.draft['control'] != control):
                    rewound = self.api('/rewind', dict(session=self.sid, boundary=self.window, revision=self.revision))
                    self.revision = rewound['revision']
                    self.draft = None
                result = self.draft['result'] if self.draft is not None else self.api("/next", dict(
                    session=self.sid, prompt=prompt, window=self.window, revision=self.revision,
                    include_motion=True, request_version=version, control=control))
                # A scheduled transition inside a window keeps its preceding
                # tokens and samples the suffix under the new text.
                transitions = sorted(k for k in getattr(self, 'scheduled_prompts', {})
                    if self.window*52 < k < (self.window+1)*52)
                for target in (transitions if self.draft is None else []):
                    result = self.api('/replan/prepare', dict(session=self.sid,
                        source_frame=target, revision=self.revision,
                        prompt=self.scheduled_prompts[target], control=control, request_version=version))
                    accepted = self.api('/replan/accept', dict(session=self.sid,
                        token=result['replan_token'], revision=self.revision))
                    self.revision = accepted['revision']
                    prompt = self.scheduled_prompts[target]
                prompt = result.get('prompt', prompt)
                with self.intent_lock:
                    if version != self.intent_version or control != self.controls(session):
                        rewound = self.api('/rewind', dict(session=self.sid, boundary=self.window, revision=self.revision))
                        self.revision = rewound['revision']
                        self.draft = None
                        self.event('draft_cancelled', obsolete_version=version, boundary=self.window)
                        continue
                    if self.window and not getattr(self, 'planning', False) and (not session.playing or self.window * 52 - session.frame_idx > self.publish_ahead_frames):
                        self.draft = dict(result=result, control=control, version=version)
                        self.prepare_requested = False
                        self.event('draft_ready', boundary=self.window, prompt=prompt)
                        return
                samples = torch.tensor(result["normalized_motion"], dtype=torch.float32).unsqueeze(0)
                if samples.shape != (1, 52, 414) or not torch.isfinite(samples).all():
                    raise ValueError("Invalid normalized motion from generator")
                decoded = self.motion_rep.inverse(samples, is_normalized=True)
                converted = np.asarray(session.mujoco_converter.dict_to_qpos(decoded, device="cpu", numpy=True)).reshape(52, 36)
                difference = float(np.max(np.abs(converted - np.asarray(result["qpos"]))))
                if difference > 1e-3:
                    raise ValueError(f"Original viewer decoder differs from generator qpos: {difference}")
                raw = self.motion_rep.unnormalize(samples)
                velocities = raw[:, :, self.motion_rep.slice_dict["velocities"]].reshape(1, 52, self.skeleton.nbjoints, 3)
                new = dict(motion_tensor=samples, joints_pos=decoded["posed_joints"],
                    joints_rot=decoded["global_rot_mats"], foot_contacts=decoded["foot_contacts"],
                    root_velocities=velocities[:, :, self.skeleton.root_idx])
                # Recheck after CPU decode. Publication is the irrevocable
                # linearization point; input receipt never waits for inference.
                with self.intent_lock:
                    if version != self.intent_version or control != self.controls(session):
                        rewound = self.api('/rewind', dict(session=self.sid, boundary=self.window, revision=self.revision))
                        self.revision = rewound['revision']
                        self.draft = None
                        continue
                    self.api("/publish", dict(session=self.sid, boundary=self.window + 1, revision=self.revision))
                    self.draft = None
                    self.prepare_requested = False
                with self.motion_update_lock:
                    first = session.motion_tensor is None
                    with session.motion_tensor_lock:
                        for name, tensor in new.items():
                            previous = getattr(session, name)
                            if previous is None:
                                previous = DiskMotionHistory(self.run_dir/'source', name)
                                setattr(session, name, previous)
                            previous.append(tensor)
                        session.max_frame_idx = session.motion_tensor.shape[1] - 1
                    if first:
                        demo.add_character(session.client.client_id, self.skeleton, 0)
                        demo.update_camera_follow(session.client.client_id, 0, use_smoothing=False)
                    np.save(self.run_dir/f'qpos_w{self.window:06d}.npy', converted)
                    self.record_control(self.window * 52, control)
                    start = self.window * 52
                    cuts = [start, *transitions, start+52]
                    for lo,hi in zip(cuts,cuts[1:]):
                        segment_prompt = self.prompt_at(lo)
                        if self.segments and self.segments[-1]['text'] == segment_prompt:
                            self.segments[-1]['end_frame'] = hi-1
                        else:
                            self.segments.append(dict(text=segment_prompt,start_frame=lo,end_frame=hi-1))
                    if demo.sonic_output and not self.reference_preview:
                        demo.sonic_output.append_source_window(converted, source_start=start,
                            name="Motion Mamba 2.0: " + prompt,
                            stream_complete=bool(self.limit and self.window + 1 == self.limit))
                    self.window += 1
                    session.gui_elements.gui_frame_idx_input.max = (max(session.max_frame_idx, int(self.consumed_frame)+1500)
                        if getattr(self, 'planning', False) and not getattr(self, 'sequential_preload', False)
                        else session.max_frame_idx)
                    self.sync_timeline(demo, session)
                    self.segments = self.segments[-32:]
                record = dict(session=self.sid, window=self.window, prompt=prompt, start_frame=start,
                    generation_seconds=result["generation_seconds"], total_seconds=time.monotonic()-started,
                    qpos_decoder_max_abs_difference=difference, accepted_commits=result["accepted_commits"],
                    wall_time=time.time(), request_version=version, revision=self.revision,
                    control=control, condition_nonzero=result.get('condition_nonzero', 0))
                self.write_trace('windows.jsonl', record)
                print("[MM2] " + json.dumps(record), flush=True)
                self.event('published', source_start=start, source_end=self.window*52,
                           prompt=prompt, published_version=version)
                if getattr(self, 'planning', False):
                    self.prepare_requested = self.window*52 < self.plan_until
                    if self.planning_frame <= session.max_frame_idx:
                        session.mm2_rendering = True
                        try:
                            demo.set_frame(session.client.client_id, self.planning_frame)
                        finally:
                            session.mm2_rendering = False
                return
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.failed = True
            self.last_error = str(exc)
            self.error_kind = type(exc).__name__
            self.event('error', error=self.last_error, error_kind=self.error_kind)
            demo.set_playback_state(session.client.client_id, False, 'generation_error')
            self.notice(session, f"生成已暂停：{exc}", "red")

    def apply_live_replan(self, demo, session):
        """Prepare on a history fork; adopt only a current, safe future tail."""
        with self.intent_lock:
            version = self.intent_version
            target = self.replan_target
            prompt = self.prompt_at(target) if target is not None else self.prompt
        bridge = demo.sonic_output if not self.planning else None
        rejection = 'replacement not accepted'
        for attempt in range(2):
            if target is None:
                guard = int(np.ceil(self.replan_budgets[0]*25)) if session.playing else 4
                frame = ((int(self.consumed_frame) + guard + 3)//4)*4
                if session.playing and frame%52 > 20:
                    guard = int(np.ceil(self.replan_budgets[1]*25))
                    frame = ((int(self.consumed_frame)+guard+3)//4)*4
            else:
                frame = target
            if frame >= self.window*52:
                self.live_replan_requested = False
                self.prepare_requested = True
                return
            boundary, prefix = divmod(frame, 52)
            # Paused planning regenerates later windows through prompt_at;
            # do not let a guard window overwrite a second scheduled event.
            lookahead = int(prefix > 20 and not self.planning)
            started = time.monotonic()
            control = self.controls(session, start_frame=boundary*52)
            following_control = self.controls(session, start_frame=(boundary+1)*52)
            token = None
            try:
                result = self.api('/replan/prepare', dict(session=self.sid, source_frame=frame,
                    revision=self.revision, prompt=prompt, request_version=version, control=control,
                    lookahead_windows=lookahead, lookahead_control=following_control))
                token = result['replan_token']
                samples = torch.tensor(result['normalized_motion'], dtype=torch.float32)[None]
                decoded = self.motion_rep.inverse(samples, is_normalized=True)
                converted = np.asarray(session.mujoco_converter.dict_to_qpos(decoded,
                    device='cpu', numpy=True)).reshape(52,36)
                if not np.isfinite(converted).all() or np.max(np.abs(converted-np.asarray(result['qpos']))) > 1e-3:
                    raise ValueError('replacement codec mismatch')
                raw = self.motion_rep.unnormalize(samples)
                velocities = raw[:,:,self.motion_rep.slice_dict['velocities']].reshape(1,52,self.skeleton.nbjoints,3)
                tensors = dict(motion_tensor=samples, joints_pos=decoded['posed_joints'],
                    joints_rot=decoded['global_rot_mats'], foot_contacts=decoded['foot_contacts'],
                    root_velocities=velocities[:,:,self.skeleton.root_idx])
                extra_tensors = []
                extra_qpos = []
                for following in result.get('following_windows', []):
                    x = torch.tensor(following['normalized_motion'],dtype=torch.float32)[None]
                    d = self.motion_rep.inverse(x,is_normalized=True)
                    q = np.asarray(session.mujoco_converter.dict_to_qpos(d,device='cpu',numpy=True)).reshape(52,36)
                    if np.max(np.abs(q-np.asarray(following['qpos']))) > 1e-3:
                        raise ValueError('replacement lookahead codec mismatch')
                    v = self.motion_rep.unnormalize(x)[:,:,self.motion_rep.slice_dict['velocities']].reshape(1,52,self.skeleton.nbjoints,3)
                    extra_tensors.append(dict(motion_tensor=x,joints_pos=d['posed_joints'],
                        joints_rot=d['global_rot_mats'],foot_contacts=d['foot_contacts'],
                        root_velocities=v[:,:,self.skeleton.root_idx]))
                    extra_qpos.append(q)
                with self.intent_lock:
                    if version != self.intent_version:
                        self.api('/replan/cancel', dict(session=self.sid))
                        return
                    def accept():
                        adopted = self.api('/replan/accept', dict(session=self.sid, token=token, revision=self.revision))
                        self.revision = adopted['revision']
                    if bridge is not None:
                        receipt = bridge.replace_source_tail(np.concatenate([converted[prefix:],*extra_qpos]), source_start=frame,
                            name='Motion Mamba 2.0: '+prompt, commit=accept)
                    else:
                        accept()
                        receipt = dict(source_start=frame)
                    with self.motion_update_lock, session.motion_tensor_lock:
                        for name, tensor in tensors.items():
                            getattr(session, name).replace_tail(boundary, tensor)
                        for extra in extra_tensors:
                            for name,tensor in extra.items():getattr(session,name).append(tensor)
                        session.max_frame_idx = (boundary+1+len(extra_tensors))*52-1
                    self.window = boundary+1+len(extra_tensors)
                    self.api('/publish', dict(session=self.sid, boundary=self.window, revision=self.revision))
                    self.draft = None
                    self.live_replan_requested = False
                    self.prepare_requested = self.planning and self.window*52 < self.plan_until
                    self.segments = [dict(s, end_frame=min(s['end_frame'], frame-1))
                                     for s in self.segments if s['start_frame'] < frame]
                    self.segments.append(dict(text=prompt, start_frame=frame, end_frame=self.window*52-1))
                    self.record_control(frame, control)
                    for i in range(len(extra_tensors)):
                        self.record_control((boundary+1+i)*52, following_control)
                    np.save(self.run_dir/f'qpos_w{boundary:06d}.npy', converted)
                    for i,q in enumerate(extra_qpos):np.save(self.run_dir/f'qpos_w{boundary+1+i:06d}.npy',q)
                    if self.planning:
                        self.seek_playback(demo, session, min(self.planning_frame, session.max_frame_idx), force=True)
                    self.sync_timeline(demo, session)
                    self.event('reference_replaced', source_start=frame, source_end=self.window*52,
                        frozen_prefix_frames=prefix, consumed_source_frame=self.consumed_frame,
                        prompt=prompt, elapsed=time.monotonic()-started, seam=receipt)
                    self.replan_budgets[lookahead] = max(.85,time.monotonic()-started+.4)
                    return
            except Exception as exc:
                rejection = str(exc)
                self.replan_budgets[lookahead] = max(self.replan_budgets[lookahead],time.monotonic()-started+.3)
                if token is not None:
                    self.api('/replan/cancel', dict(session=self.sid))
                self.event('replacement_rejected', source_start=frame, error=str(exc))
                if target is not None or 'unsafe replacement seam' in str(exc):
                    break
        # Preserve the valid live plan. This is a rejected candidate, not a
        # generation failure or a reason to disable physical safety.
        self.live_replan_requested = False
        self.prepare_requested = True
        self.sync_timeline(demo,session)
        if 'unsafe replacement seam' in rejection:
            # A candidate rejected by the continuity guard is an internal
            # retry/fallback, not a keyboard input error. Preserve diagnostics
            # and the guard without stacking technical warnings over the scene.
            self.event('replacement_deferred', reason=rejection,
                       fallback='next_window', source_request_version=version)
        else:
            self.notice(session, f'即时替换未完成，保留当前动作：{rejection}', 'orange')

    def controls(self, session, start_frame=None):
        g = session.gui_elements
        for name in ('Full-Body', 'End-Effectors'):
            if name in session.constraints and session.constraints[name].get_constraint_info()['frame_idx']:
                raise ValueError('远程模式目前只接入root位置/heading；请清除Full-Body/EE条件后重试。')
        if g.gui_use_target_velocity_checkbox.value:
            v = list(g.gui_target_root_velocity.value)
            result = dict(version='g1_joint_interaction_v1',
                          intent='decelerate' if np.linalg.norm(v) < 1e-6 else 'move',
                          velocity_xz=v, event_id='viewer_keyboard')
            if g.gui_use_target_heading_checkbox.value and np.linalg.norm(v) > 1e-6:
                result['heading'] = float(np.arctan2(v[0], v[1]))
            return result
        if '2D Root' not in session.constraints:
            return {}
        constraint = session.constraints['2D Root']
        with constraint._data_lock:
            keyboard_frames = {frame for frame, ids in constraint.frame2keyid.items()
                               if ids and all(str(k).startswith('target_vel_waypoint_') for k in ids)}
            # Filter registered keyboard keys before dense interpolation can
            # turn their in-between frames into accidental mouse conditions.
            info = (constraint._get_sparse_constraint_info() if keyboard_frames
                    else constraint.get_constraint_info())
        points = []
        start_frame = self.window*52 if start_frame is None else start_frame
        for i, frame in enumerate(info['frame_idx']):
            # Registered keyboard preview keys belong to velocity control;
            # they must not become mouse targets after keyboard release/seek.
            if frame in keyboard_frames:
                continue
            # Match foundation's bounded future condition (G52 + F<=196).
            # The endpoint keeps its timestamp; it is not moved into this window.
            if start_frame <= frame < start_frame+248:
                point = dict(frame=int(frame-start_frame), xz=np.asarray(info['root_pos'][i])[[0,2]].tolist())
                if 'global_root_heading' in info:
                    point['heading'] = float(info['global_root_heading'][i])
                points.append(point)
        route = getattr(session, 'mm2_waypoint_route', None)
        if route:
            # The final user point and generated sparse guide may coincide.
            # Their timestamps stay absolute until this bounded RPC encoding.
            by_frame = {}
            for p in route:
                if start_frame <= p['frame'] < start_frame+248:
                    by_frame[p['frame']-start_frame] = dict(frame=p['frame']-start_frame,xz=p['xz'])
            # Explicit timeline constraints take precedence over live guides.
            by_frame.update({p['frame']:p for p in points})
            points = sorted(by_frame.values(),key=lambda p:p['frame'])
        return dict(root2d=points) if points else {}

    def replan(self, demo, session):
        self.failed = False
        self.last_error = self.error_kind = None
        self.update_prompt(demo, session, trigger_replan=True)

    def control_changed(self, demo, session):
        if getattr(session, 'mm2_control_batch', False):
            return
        with self.intent_lock:
            control = self.controls(session)
            if control == self._last_control:
                return
            self._last_control = control
            velocity = control.get('velocity_xz')
            self.velocity_preview = tuple(velocity) if velocity is not None else None
            self.intent_version += 1
            self.prepare_requested = True
            self.replan_target = ((self.planning_frame+3)//4)*4 if self.planning else None
            self.live_replan_requested = self.live_replanning and self.window > 0 and (
                self.replan_target is None or self.replan_target < self.window*52)
            self.failed = False
            self.last_error = self.error_kind = None
            self.event('control_received', control=control, playing=session.playing)
            if self._control_timer is not None:
                self._control_timer.cancel()
            self._control_timer = threading.Timer(.08, demo.on_replan_trigger,
                args=(session.client.client_id,), kwargs={'skip_if_busy': True})
            self._control_timer.daemon = True
            self._control_timer.start()

    def receive_playback_state(self, session, bridge, stop):
        """Newest telemetry stays independent of inference and GUI rendering."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"ardy_sonic_web/backend"))
        from protocol import decode_reference_telemetry_packet, decode_packet, ACTUAL_MAGIC
        context = zmq.Context()
        actual = context.socket(zmq.SUB)
        reference = context.socket(zmq.SUB)
        for socket, endpoint, topic in ((actual, int(os.environ.get('TELEMETRY_ZMQ_PORT', '5560')), b""),
                                       (reference, int(os.environ.get('REFERENCE_TELEMETRY_ZMQ_PORT', '5557')), b"g1_debug")):
            socket.setsockopt(zmq.SUBSCRIBE, topic)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.connect(f"tcp://127.0.0.1:{endpoint}")
        poller = zmq.Poller()
        poller.register(actual, zmq.POLLIN)
        poller.register(reference, zmq.POLLIN)
        epoch = None
        ready = False
        epoch_started = 0.
        try:
            while (not stop.is_set() and not session.stop_playback
                   and getattr(session, 'model', self) is self):
                events = dict(poller.poll(20))
                if bridge is None:
                    continue
                with bridge._lock:
                    current_epoch = bridge._state.epoch
                    expected_start = bridge._state.frame
                with self.telemetry_lock:
                    if current_epoch != epoch:
                        epoch, ready = current_epoch, False
                        epoch_started = time.monotonic()
                        self.reference = {}
                        self.telemetry_epoch = epoch
                        self.telemetry_ready = False
                    if actual in events:
                        payload = actual.recv()
                        try:
                            self.actual, _ = decode_packet(payload, ACTUAL_MAGIC)
                        except Exception as exc:
                            self.event('telemetry_packet_rejected', channel='actual', error=str(exc))
                    if reference in events:
                        payload = reference.recv()
                        try:
                            packet = decode_reference_telemetry_packet(payload)
                        except Exception as exc:
                            self.event('telemetry_packet_rejected', channel='reference', error=str(exc))
                            continue
                        fresh = 0 <= time.monotonic()-packet['monotonic_timestamp'] < .15
                        if (fresh and packet['monotonic_timestamp'] >= epoch_started
                                and abs(packet['current_frame_global']-expected_start) <= 4):
                            ready = True
                        if ready and fresh:
                            self.reference = packet
                    packet = self.reference
                    matched = (ready and self.actual.get('startup_epoch') == epoch
                        and packet and 0 <= time.monotonic()-packet['monotonic_timestamp'] < .15)
                    self.telemetry_ready = bool(matched)
                    if matched and packet and bridge.observe_source_consumer(
                            epoch=epoch, frame=packet['current_frame_global'],
                            observed_at=packet['monotonic_timestamp'],
                            received_until=packet['stream_window_start']+packet['buffer_length']):
                        self.consumed_frame = packet['current_frame_global']/2
        except Exception as exc:
            self.event('telemetry_error', error=str(exc))
            self.telemetry_ready = False
        finally:
            actual.close(); reference.close(); context.term()

    def playback(self, demo, session):
        bridge = demo.sonic_output
        stop = threading.Event()
        receiver = threading.Thread(target=self.receive_playback_state,
            args=(session, bridge, stop), name='mm2-latest-telemetry', daemon=True)
        receiver.start()
        epoch = None
        ready = False
        buffering = False
        logged = 0
        fps_anchor = time.monotonic()
        fps_frame = session.frame_idx
        next_tick = time.monotonic()
        render_times = []
        render_intervals = []
        rendered_at = None
        render_report_at = next_tick

        def render(source):
            nonlocal rendered_at
            began = time.monotonic()
            session.mm2_rendering = True
            try:
                # All robot parts, arrows and frame labels share one pose.
                with session.client.atomic():
                    demo.set_frame(session.client.client_id, source)
                session.client.flush()
            finally:
                session.mm2_rendering = False
            render_times.append((time.monotonic()-began)*1000)
            if rendered_at is not None:
                render_intervals.append((began-rendered_at)*1000)
            rendered_at = began
        try:
            while not session.stop_playback and session.model is self:
                if stop.wait(max(0., next_tick-time.monotonic())):
                    break
                # Absolute deadlines: rendering time must not be added to
                # every 20 ms interval. Never burst to catch up missed ticks.
                next_tick = max(next_tick+.02, time.monotonic())
                if bridge is not None:
                    with bridge._lock:
                        current_epoch = bridge._state.epoch
                    failure = getattr(bridge, 'stream_error', None)
                    if failure and failure != getattr(self, '_reported_stream_error', None):
                        self._reported_stream_error = failure
                        demo.set_playback_state(session.client.client_id, False, 'bridge_error')
                        self.notice(session, '参考发送链路已暂停并重连：' + failure, 'red')
                else:
                    current_epoch = 0
                if current_epoch != epoch:
                    epoch, ready, buffering = current_epoch, False, False
                    self.completed_at_reference_frame = None
                with self.telemetry_lock:
                    ready = self.telemetry_ready and self.telemetry_epoch == epoch
                    packet = self.reference
                    actual_state = self.actual
                matched = ready and actual_state.get("startup_epoch") == epoch
                # Restore the synchronized clock used before 17:45. A local
                # wall clock must not run ahead during SONIC startup/buffering.
                # Standalone Viewer playback still has a local clock.
                if bridge is None and not self.planning:
                    source = self.reference_clock.advance(time.monotonic(), session.frame_float,
                        session.max_frame_idx, session.playing, session.model_fps)
                    if source >= 0 and abs(source-session.frame_float) > .1:
                        render(source)
                if bridge is not None and matched and packet:
                    with self.motion_update_lock:
                        # A seek/replay may finish after telemetry was decoded.
                        # Never apply its old frame to the new epoch/cursor.
                        with bridge._lock:
                            same_epoch = bridge._state.epoch == epoch
                        if not same_epoch:
                            continue
                        frame = int(packet["current_frame_global"])
                        ahead = (session.max_frame_idx / 25) - frame / 50
                        if session.playing:
                            if self.actual.get("fallen"):
                                demo.set_playback_state(session.client.client_id, False, 'fallen')
                                self.notice(session, "仿真检测到跌倒，已暂停。", "red")
                            elif self.limit and self.window >= self.limit and frame >= self.window * 104 - 2:
                                self.completed_at_reference_frame = frame
                                demo.set_playback_state(session.client.client_id, False, 'finite_horizon')
                            elif (ahead < 1.1 and session.gui_elements.gui_enable_auto_replan_checkbox.value
                                  and (not self.limit or self.window < self.limit)):
                                with bridge._lock:
                                    active = bridge._state.playing
                                if active:
                                    bridge.viewer_playback_control("pause")
                                buffering = True
                            elif buffering and (ahead >= 1.5 or (self.limit and self.window >= self.limit)):
                                buffering = False
                                bridge.viewer_playback_control("play")
                        elif buffering:
                            buffering = False
                        source = self.synchronized_source_frame(session, frame)
                        if source >= 0 and abs(source - session.frame_float) > .1:
                            render(source)
                if (session.playing or self.prepare_requested or (self.planning and self.window*52 < self.plan_until)) and session.gui_elements.gui_enable_auto_replan_checkbox.value:
                    if self.needs_generation(session):
                        if not session.replan_lock.locked():
                            threading.Thread(target=demo.on_replan_trigger, args=(session.client.client_id,),
                                kwargs={"skip_if_busy": True}, daemon=True).start()
                if time.monotonic() - logged > .25 and self.sid:
                    logged = time.monotonic()
                    state = dict(session=self.sid, windows=self.window, source_frame=session.frame_idx,
                        playing=session.playing, buffering=buffering, epoch=epoch,
                        playback_clock='sonic' if bridge is not None else 'reference', telemetry_ready=ready,
                        epoch_matched=bool(matched), generated_frames=session.max_frame_idx+1,
                        completed_at_reference_frame=self.completed_at_reference_frame,
                        actual=self.actual, reference=self.reference, failed=self.failed,
                        wall_time=time.time(), request_version=self.intent_version, revision=self.revision,
                        pause_reason=getattr(session, 'pause_reason', None), error=self.last_error,
                        error_kind=self.error_kind, published_source_boundary=self.window*52,
                        consumer_locked_source_boundary=int(self.consumed_frame)+1,
                        consumed_source_frame=self.consumed_frame,planning=self.planning,
                        planning_frame=self.planning_frame,
                        displayed_target_velocity=self.target_velocity_at(session, session.frame_idx))
                    self.write_trace('playback.jsonl', state)
                if time.monotonic() - fps_anchor >= 1:
                    executing = next((p['text'] for p in self.segments
                        if p['start_frame'] <= session.frame_idx <= p['end_frame']), self.received_prompt)
                    session.gui_elements.gui_active_prompt_label.content = f'**Active Prompt:** {executing}'
                    now = time.monotonic()
                    session.gui_elements.gui_actual_fps.value = max(0, session.frame_idx - fps_frame) / (now - fps_anchor)
                    fps_anchor, fps_frame = now, session.frame_idx
                if time.monotonic()-render_report_at >= 5:
                    if render_times:
                        self.write_trace('render_metrics.jsonl', dict(
                            wall_time=time.time(), frames=len(render_times),
                            render_ms_p50=float(np.percentile(render_times, 50)),
                            render_ms_p95=float(np.percentile(render_times, 95)),
                            interval_ms_p50=float(np.percentile(render_intervals, 50)) if render_intervals else None,
                            interval_ms_p95=float(np.percentile(render_intervals, 95)) if render_intervals else None))
                    render_times.clear()
                    render_intervals.clear()
                    render_report_at = time.monotonic()
        finally:
            stop.set()
            receiver.join(timeout=1)

    def synchronized_source_frame(self, session, reference_frame):
        """Use the old SONIC consumer clock while preserving paused editing."""
        if self.planning:
            return session.frame_float
        if self.completed_at_reference_frame is not None:
            return session.max_frame_idx
        return max(session.frame_float, min(session.max_frame_idx, reference_frame / 2))
