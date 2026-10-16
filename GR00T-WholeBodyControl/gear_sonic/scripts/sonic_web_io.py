"""Non-blocking actual-state telemetry and viewer controls for MuJoCo."""

from __future__ import annotations

import json
import os
import struct
import time

import mujoco
import numpy as np
import zmq


ACTUAL_MAGIC = b"AST1"
DEBUG_MODES = {
    "reference_only",
    "sonic_official",
    "ardy_offline",
    "ardy_streaming",
}

# Must stay identical to gear_sonic_deploy/.../policy_parameters.hpp. These
# are hardware/MuJoCo-order absolute joint targets used by SONIC's INIT state.
SONIC_DEFAULT_ANGLES = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)


class SonicWebIO:
    """Own ZMQ sockets and access mjData only from the simulation thread."""

    def __init__(
        self,
        simulator,
        camera: mujoco.MjvCamera | None,
        *,
        telemetry_port: int = 5560,
        control_port: int = 5558,
        telemetry_fps: float = 50.0,
        debug_mode: str = "ardy_streaming",
    ) -> None:
        if debug_mode not in DEBUG_MODES:
            raise ValueError(
                f"Unsupported debug mode {debug_mode!r}; expected {sorted(DEBUG_MODES)}"
            )
        self.simulator = simulator
        self.sim_env = simulator.sim_env
        self.camera = camera
        self.debug_mode = debug_mode
        # Headless sessions start before the SONIC policy. Advancing the
        # reset-time suspension harness without controller commands can make
        # the initial state explode before the first browser connection.
        self.physics_enabled = False
        self.physics_requested = False
        self.policy_requested = False
        self.reference_qpos: np.ndarray | None = None
        self.startup_phase = "waiting_reference"
        self.startup_epoch = -1
        # The bridge may still contain a command from the previous episode.
        # Keep MuJoCo frozen only until the restarted controller publishes its
        # first fresh LowCmd, then run INIT under the full suspension harness.
        self.init_physics_pending = True
        self.camera_follow = True
        self.release_pending = False
        self.release_started_at: float | None = None
        self.release_ramp_seconds = float(
            os.environ.get("SONIC_RELEASE_RAMP_SECONDS", "1.5")
        )
        if self.release_ramp_seconds < 0.0:
            raise ValueError("SONIC_RELEASE_RAMP_SECONDS must be non-negative")
        self.release_not_before = 0.0
        self.release_min_seconds = float(
            os.environ.get("SONIC_RELEASE_MIN_SECONDS", "0.1")
        )
        if self.release_min_seconds < 0.0:
            raise ValueError("SONIC_RELEASE_MIN_SECONDS must be non-negative")
        self.policy_ready_steps = 0
        self.release_stable_steps = 1
        self.fallen = False
        self.physics_step_mode = os.environ.get(
            "SONIC_PHYSICS_STEP_MODE", "fixed_lowcmd"
        ).strip()
        if self.physics_step_mode not in {"fixed_lowcmd", "wallclock"}:
            raise ValueError(
                "SONIC_PHYSICS_STEP_MODE must be 'fixed_lowcmd' or 'wallclock', "
                f"got {self.physics_step_mode!r}"
            )
        self.physics_steps_per_low_cmd = int(
            os.environ.get("SONIC_PHYSICS_STEPS_PER_LOWCMD", "4")
        )
        if self.physics_steps_per_low_cmd < 1:
            raise ValueError("SONIC_PHYSICS_STEPS_PER_LOWCMD must be >= 1")
        self.max_catch_up_steps = int(
            os.environ.get("SONIC_MAX_PHYSICS_CATCH_UP_STEPS", "8")
        )
        if self.max_catch_up_steps < 1:
            raise ValueError("SONIC_MAX_PHYSICS_CATCH_UP_STEPS must be >= 1")
        self.physics_wall_anchor = time.monotonic()
        self.physics_sim_anchor = float(self.sim_env.mj_data.time)
        self.last_stepped_low_cmd_sequence = (
            self.sim_env.unitree_bridge.get_low_cmd_sequence()
        )
        self.last_policy_ready_sequence = self.last_stepped_low_cmd_sequence
        self.physics_step_count = 0
        self.consumed_low_cmd_count = 0
        self.dropped_low_cmd_count = 0
        self.stats_wall_anchor = time.monotonic()
        self.stats_sim_anchor = float(self.sim_env.mj_data.time)
        self.stats_step_anchor = 0
        self.stats_command_anchor = self.last_stepped_low_cmd_sequence
        self.stats_command_change_anchor = (
            self.sim_env.unitree_bridge.get_low_cmd_change_sequence()
        )
        self.telemetry_period = 1.0 / telemetry_fps
        self.next_telemetry = time.monotonic()
        self.frame = 0
        self.next_diagnostic = time.monotonic()
        self.context = zmq.Context()
        band = getattr(self.sim_env, "elastic_band", None)
        self.band_nominal_gains = (
            None
            if band is None
            else (band.kp_pos, band.kd_pos, band.kp_ang, band.kd_ang)
        )
        self.safety_harness_scale = float(
            os.environ.get("SONIC_SAFETY_HARNESS_SCALE", "0")
        )
        if not 0.0 <= self.safety_harness_scale <= 1.0:
            raise ValueError(
                "SONIC_SAFETY_HARNESS_SCALE must be between 0 and 1, got "
                f"{self.safety_harness_scale}."
            )
        self._arm_band_at_current_pose()

        self.telemetry = self.context.socket(zmq.PUB)
        self.telemetry.setsockopt(zmq.SNDHWM, 2)
        self.telemetry.setsockopt(zmq.LINGER, 0)
        self.telemetry.bind(f"tcp://127.0.0.1:{telemetry_port}")

        self.control = self.context.socket(zmq.PULL)
        self.control.setsockopt(zmq.RCVHWM, 16)
        self.control.setsockopt(zmq.LINGER, 0)
        self.control.bind(f"tcp://127.0.0.1:{control_port}")
        print(
            f"SONIC web I/O: telemetry=127.0.0.1:{telemetry_port}, "
            f"control=127.0.0.1:{control_port}; physics_step_mode="
            f"{self.physics_step_mode}, steps_per_lowcmd="
            f"{self.physics_steps_per_low_cmd}",
            flush=True,
        )

    def controlled_sim_step(self, original_sim_step) -> None:
        self._drain_controls()
        if self.physics_enabled:
            if self.physics_step_mode == "fixed_lowcmd":
                self._step_for_fresh_low_cmd(original_sim_step)
            else:
                self._step_to_wall_clock(original_sim_step)
        else:
            # Pausing MuJoCo must not look like a robot disconnect. The policy
            # safety loop still expects fresh LowState/IMU samples while the
            # simulation clock and qpos remain frozen.
            self._publish_frozen_low_state()
        self._enable_init_physics_when_ready()
        self._release_when_policy_ready()
        self._advance_release_ramp()
        self._publish_if_due()

    def _step_for_fresh_low_cmd(self, original_sim_step) -> None:
        sequence = self.sim_env.unitree_bridge.get_low_cmd_sequence()
        sequence_delta = sequence - self.last_stepped_low_cmd_sequence
        if sequence_delta <= 0:
            self._publish_frozen_low_state()
            return
        if sequence_delta > 1:
            self.dropped_low_cmd_count += sequence_delta - 1
            print(
                "[SONIC timing warning] skipped LowCmd messages="
                f"{sequence_delta - 1}; applying only the newest command.",
                flush=True,
            )
        self.last_stepped_low_cmd_sequence = sequence
        self.consumed_low_cmd_count += 1
        self._run_physics_steps(original_sim_step, self.physics_steps_per_low_cmd)

    def _step_to_wall_clock(self, original_sim_step) -> None:
        now = time.monotonic()
        target_sim_time = self.physics_sim_anchor + (now - self.physics_wall_anchor)
        sim_dt = float(self.sim_env.sim_dt)
        lag = target_sim_time - float(self.sim_env.mj_data.time)
        step_count = min(
            max(1, int(np.ceil(lag / sim_dt))), self.max_catch_up_steps
        )
        self._run_physics_steps(original_sim_step, step_count)

    def _run_physics_steps(self, original_sim_step, step_count: int) -> None:
        for _ in range(step_count):
            original_sim_step()
            self.physics_step_count += 1
            if bool(getattr(self.sim_env, "fall", False)):
                self.fallen = True
                self.physics_enabled = False
                self.physics_requested = False
                print(
                    "SONIC fall latched; physics frozen for inspection. "
                    "Send restart after correcting the underlying issue.",
                    flush=True,
                )
                break

    def _enable_init_physics_when_ready(self) -> None:
        if not self.init_physics_pending:
            return
        if not self.policy_requested or self.reference_qpos is None:
            return
        bridge = self.sim_env.unitree_bridge
        if not bridge.low_cmd_received:
            return
        self.init_physics_pending = False
        self.physics_requested = True
        self._set_physics_enabled(True)
        self.startup_phase = "aligning"
        print(
            "SONIC fresh controller command detected; running INIT physics "
            "under the full suspension harness.",
            flush=True,
        )

    def _publish_frozen_low_state(self) -> None:
        env = self.sim_env
        env.obs = env.prepare_obs()
        env.unitree_bridge.PublishLowState(env.obs)
        if env.unitree_bridge.joystick:
            env.unitree_bridge.PublishWirelessController()

    def _drain_controls(self) -> None:
        while self.control.poll(0):
            message = self.control.recv_json()
            action = message.get("action")
            if "startup_epoch" in message:
                try:
                    candidate_epoch = int(message["startup_epoch"])
                except (TypeError, ValueError):
                    candidate_epoch = -1
                if candidate_epoch >= 0:
                    self.startup_epoch = candidate_epoch
            if action == "physics":
                self.physics_requested = bool(message.get("enabled", True))
                if self.debug_mode == "reference_only":
                    self.physics_requested = False
                if not self.physics_requested:
                    self._set_physics_enabled(False)
                    if self.startup_phase == "running":
                        self.startup_phase = "paused"
                elif (
                    self.policy_requested
                    and not self.init_physics_pending
                    and not self.release_pending
                ):
                    self._set_physics_enabled(True)
                    if self.physics_enabled and self.startup_phase == 'paused':
                        self.startup_phase = 'running'
            elif action == "follow":
                self.camera_follow = bool(message.get("enabled", True))
                if self.camera is None:
                    continue
                self.camera.type = (
                    mujoco.mjtCamera.mjCAMERA_TRACKING
                    if self.camera_follow
                    else mujoco.mjtCamera.mjCAMERA_FREE
                )
                self.camera.lookat[:] = self.sim_env.mj_data.body("pelvis").xpos
            elif action == "camera_orbit":
                if self.camera is None:
                    continue
                self.camera.azimuth -= float(message.get("dx", 0.0)) * 0.25
                self.camera.elevation = float(
                    np.clip(
                        self.camera.elevation - float(message.get("dy", 0.0)) * 0.2,
                        -80.0,
                        10.0,
                    )
                )
            elif action == "camera_zoom":
                if self.camera is None:
                    continue
                self.camera.distance = float(
                    np.clip(
                        self.camera.distance + float(message.get("delta", 0.0)) * 0.25,
                        1.2,
                        8.0,
                    )
                )
            elif action == "restart":
                reference_qpos = message.get("reference_qpos", self.reference_qpos)
                self._reset_episode(reference_qpos)
            elif action == "reference_pose":
                self._reset_episode(message.get("reference_qpos"))
            elif action == "policy" and not bool(message.get("enabled", True)):
                self.policy_requested = False
                self.physics_requested = False
                self.release_pending = False
                if self.release_started_at is not None:
                    self.release_started_at = None
                    self._set_band_gain_scale(1.0)
                    band = getattr(self.sim_env, "elastic_band", None)
                    if band is not None:
                        band.enable = True
                self._set_physics_enabled(False)
                self.startup_phase = "holding_reference"
            elif action == "policy" and bool(message.get("enabled", True)):
                if self.debug_mode == "reference_only":
                    print(
                        "[SONIC debug] reference_only ignores policy enable; "
                        "physics and policy remain OFF.",
                        flush=True,
                    )
                    continue
                self.policy_requested = True
                self.physics_requested = True
                band = getattr(self.sim_env, "elastic_band", None)
                # `prepare` and `play` can both request policy execution. Once
                # key-9-equivalent release has happened, later requests must
                # not start a second suspension-release cycle.
                self.release_pending = bool(band is not None and band.enable)
                # INIT must continue integrating while the robot settles. If a
                # fresh command has not arrived yet, the pending gate above
                # still keeps stale commands from moving the reset robot.
                self._set_physics_enabled(
                    self.reference_qpos is not None and not self.init_physics_pending
                )
                self.startup_phase = (
                    'running' if not self.init_physics_pending
                    and not self.release_pending and self.release_started_at is None
                    and self.policy_ready_steps >= self.release_stable_steps
                    else 'releasing' if self.release_started_at is not None
                    else 'aligning'
                )
                self.release_not_before = (
                    time.monotonic() + self.release_min_seconds
                )

    def _reset_episode(self, reference_qpos) -> None:
        candidate = None
        if reference_qpos is not None:
            candidate = np.asarray(reference_qpos, dtype=np.float64)
            if candidate.shape != (36,) or not np.isfinite(candidate).all():
                print(
                    "SONIC rejected invalid reference reset pose; keeping physics frozen.",
                    flush=True,
                )
                self._set_physics_enabled(False)
                self.startup_phase = "unsafe_reference"
                return
            quat_norm = float(np.linalg.norm(candidate[3:7]))
            if not 0.95 <= quat_norm <= 1.05:
                print(
                    "SONIC rejected reference reset pose with invalid quaternion.",
                    flush=True,
                )
                self._set_physics_enabled(False)
                self.startup_phase = "unsafe_reference"
                return
            candidate = candidate.copy()
            candidate[3:7] /= quat_norm

        self.sim_env.reset()
        data = self.sim_env.mj_data
        if candidate is not None:
            qpos_indices = self.sim_env.body_joint_index + self.sim_env.qpos_offset - 1
            data.qpos[:7] = candidate[:7]
            data.qpos[qpos_indices] = candidate[7:]
            self.reference_qpos = candidate
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        if data.ctrl.size:
            data.ctrl[:] = 0.0
        mujoco.mj_forward(self.sim_env.mj_model, data)
        self.sim_env.unitree_bridge.reset()

        self._set_physics_enabled(False)
        self.physics_requested = False
        self.policy_requested = False
        # Keep the epoch attached to the control message that initiated this
        # reset. Actual-state telemetry must acknowledge the same seek/start
        # epoch before playback can be released.
        self.init_physics_pending = True
        self.fallen = False
        self.frame = 0
        self.release_pending = False
        self.release_started_at = None
        self.release_not_before = time.monotonic() + self.release_min_seconds
        self.policy_ready_steps = 0
        self.last_stepped_low_cmd_sequence = (
            self.sim_env.unitree_bridge.get_low_cmd_sequence()
        )
        self.last_policy_ready_sequence = self.last_stepped_low_cmd_sequence
        self.physics_step_count = 0
        self.consumed_low_cmd_count = 0
        self.dropped_low_cmd_count = 0
        self.startup_phase = (
            "holding_reference" if self.reference_qpos is not None else "waiting_reference"
        )
        if self.camera is not None:
            self.camera.lookat[:] = self.sim_env.mj_data.body("pelvis").xpos
        self._arm_band_at_current_pose()
        print(
            "SONIC simulation reset to ARDY frame 0; velocities, forces, and "
            "controller command state cleared."
            if candidate is not None
            else "SONIC simulation reset without a reference pose; physics remains frozen.",
            flush=True,
        )

    def _release_when_policy_ready(self) -> None:
        if not self.release_pending:
            return
        now = time.monotonic()
        if now < self.release_not_before:
            return
        bridge = self.sim_env.unitree_bridge
        command_sequence = bridge.get_low_cmd_sequence()
        if command_sequence == self.last_policy_ready_sequence:
            return
        self.last_policy_ready_sequence = command_sequence
        motor_count = int(bridge.num_body_motor)
        commands = bridge.low_cmd.motor_cmd[:motor_count]
        if len(commands) != motor_count:
            return
        if self.reference_qpos is None:
            return
        targets = np.asarray([command.q for command in commands], dtype=np.float64)
        command_default_rms = float(
            np.sqrt(np.mean(np.square(targets - SONIC_DEFAULT_ANGLES)))
        )
        # A non-default command proves that the controller has left INIT and
        # is running the unmodified policy. Do not unlock playback from an
        # INIT pose/velocity heuristic: that can deadlock or release too early.
        ready = command_default_rms > 0.005
        self.policy_ready_steps = self.policy_ready_steps + 1 if ready else 0
        if self.policy_ready_steps < self.release_stable_steps:
            return
        band = getattr(self.sim_env, "elastic_band", None)
        if band is not None:
            self._set_physics_enabled(self.physics_requested)
            if self.safety_harness_scale > 0.0:
                self._set_band_gain_scale(self.safety_harness_scale)
                band.enable = True
                self.release_started_at = None
            else:
                # Keep the original policy command untouched, but unload the
                # reset-time suspension over a short interval. A one-step
                # key-9 toggle can inject a large base impulse before contacts
                # and the policy have taken over.
                self._set_band_gain_scale(1.0)
                band.enable = True
                self.release_started_at = now
        self.release_pending = False
        self._set_physics_enabled(self.physics_requested)
        self.startup_phase = (
            "releasing"
            if self.release_started_at is not None
            else "running"
        )
        print(
            "SONIC policy command detected; starting the suspension release ramp."
            if self.safety_harness_scale == 0.0
            else "SONIC policy command detected; safety harness remains enabled "
            f"at scale={self.safety_harness_scale:.3f}.",
            flush=True,
        )

    def _advance_release_ramp(self) -> None:
        started_at = self.release_started_at
        if started_at is None:
            return
        band = getattr(self.sim_env, "elastic_band", None)
        if band is None:
            self.release_started_at = None
            self.startup_phase = "running"
            return
        if self.release_ramp_seconds <= 0.0:
            fraction = 1.0
        else:
            fraction = np.clip(
                (time.monotonic() - started_at) / self.release_ramp_seconds,
                0.0,
                1.0,
            )
        self._set_band_gain_scale(1.0 - float(fraction))
        if fraction < 1.0:
            return
        band.enable = False
        self.sim_env.mj_data.xfrc_applied[
            self.sim_env.band_attached_link
        ] = np.zeros(6)
        self.release_started_at = None
        self.startup_phase = "running"
        print("SONIC suspension release ramp completed; policy is fully free.", flush=True)

    def _set_physics_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled) and not self.fallen
        if enabled and not self.physics_enabled:
            self.physics_wall_anchor = time.monotonic()
            self.physics_sim_anchor = float(self.sim_env.mj_data.time)
        self.physics_enabled = enabled

    def _arm_band_at_current_pose(self) -> None:
        band = getattr(self.sim_env, "elastic_band", None)
        if band is None:
            return
        # Balance gravity at the current pose. A fixed [0, 0, 1] anchor injects
        # a large upward impulse when the ARDY root starts around 0.77 m.
        attached_position = np.asarray(
            self.sim_env.mj_data.xpos[self.sim_env.band_attached_link], dtype=float
        ).copy()
        total_mass = float(mujoco.mj_getTotalmass(self.sim_env.mj_model))
        gravity = abs(float(self.sim_env.mj_model.opt.gravity[2]))
        support_deflection = total_mass * gravity / max(float(band.kp_pos), 1.0)
        band.point = attached_position + np.array(
            [0.0, 0.0, support_deflection], dtype=float
        )
        band.length = 0.0
        self._set_band_gain_scale(1.0)
        band.enable = True

    def _set_band_gain_scale(self, scale: float) -> None:
        band = getattr(self.sim_env, "elastic_band", None)
        if band is None or self.band_nominal_gains is None:
            return
        band.kp_pos, band.kd_pos, band.kp_ang, band.kd_ang = (
            gain * scale for gain in self.band_nominal_gains
        )

    def _actual_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.sim_env.mj_data
        qpos_indices = self.sim_env.body_joint_index + self.sim_env.qpos_offset - 1
        qvel_indices = self.sim_env.body_joint_index + self.sim_env.qvel_offset - 1
        joint_qpos = np.asarray(data.qpos[qpos_indices], dtype=np.float32)
        joint_qvel = np.asarray(data.qvel[qvel_indices], dtype=np.float32)
        if joint_qpos.shape != (29,) or joint_qvel.shape != (29,):
            raise RuntimeError(
                f"Expected 29 G1 body joints, got {joint_qpos.shape}/{joint_qvel.shape}"
            )
        qpos = np.concatenate((data.qpos[:7], joint_qpos)).astype("<f4", copy=False)
        qvel = np.concatenate((data.qvel[:6], joint_qvel)).astype("<f4", copy=False)
        return qpos, qvel

    def _publish_if_due(self) -> None:
        now = time.monotonic()
        if now < self.next_telemetry:
            return
        self.next_telemetry = max(self.next_telemetry + self.telemetry_period, now)
        qpos, qvel = self._actual_arrays()
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            return
        header = {
            "type": "actual_state",
            "version": 1,
            "dtype": "f32",
            "endian": "little",
            "qpos_count": 36,
            "qvel_count": 35,
            "frame": self.frame,
            "timestamp": time.time(),
            "monotonic_timestamp": now,
            "sim_time": float(self.sim_env.mj_data.time),
            "source_time": float(self.sim_env.mj_data.time),
            "frame_float": float(self.frame),
            "epoch_id": max(0, int(self.startup_epoch)),
            "sequence_id": self.frame,
            "physics": self.physics_enabled,
            "fallen": self.fallen,
            "startup_phase": self.startup_phase,
            "startup_epoch": self.startup_epoch,
            "reference_pose_loaded": self.reference_qpos is not None,
            "alignment_stable_steps": self.policy_ready_steps,
            "alignment_required_steps": self.release_stable_steps,
            "elastic_band_enabled": bool(
                getattr(getattr(self.sim_env, "elastic_band", None), "enable", False)
            ),
            "debug_mode": self.debug_mode,
            "safety_harness_scale": self.safety_harness_scale,
            "physics_step_mode": self.physics_step_mode,
            "physics_step_count": self.physics_step_count,
            "consumed_low_cmd_count": self.consumed_low_cmd_count,
            "dropped_low_cmd_count": self.dropped_low_cmd_count,
            "qpos_layout": "root_xyz_root_quat_wxyz_29_mujoco_joints",
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        packet = (
            ACTUAL_MAGIC
            + struct.pack("<I", len(encoded))
            + encoded
            + qpos.tobytes()
            + qvel.tobytes()
        )
        try:
            self.telemetry.send(packet, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        self._log_diagnostic_if_due(qpos, qvel)
        self.frame += 1

    def _log_diagnostic_if_due(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> None:
        now = time.monotonic()
        if now < self.next_diagnostic:
            return
        self.next_diagnostic = now + 1.0
        w, x, y, z = (float(value) for value in qpos[3:7])
        norm = max(1e-12, float(np.linalg.norm(qpos[3:7])))
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        bridge = self.sim_env.unitree_bridge
        commands = bridge.low_cmd.motor_cmd[: int(bridge.num_body_motor)]
        targets = np.asarray([command.q for command in commands], dtype=float)
        gains = self.sim_env.effective_body_pd_gains()
        elapsed = max(1e-9, now - self.stats_wall_anchor)
        command_sequence = bridge.get_low_cmd_sequence()
        window_steps = self.physics_step_count - self.stats_step_anchor
        window_commands = command_sequence - self.stats_command_anchor
        command_change_sequence = bridge.get_low_cmd_change_sequence()
        window_command_changes = (
            command_change_sequence - self.stats_command_change_anchor
        )
        simulation_hz = window_steps / elapsed
        command_hz = window_commands / elapsed
        command_change_hz = window_command_changes / elapsed
        steps_per_command = window_steps / max(1, window_commands)
        self.stats_wall_anchor = now
        self.stats_sim_anchor = float(self.sim_env.mj_data.time)
        self.stats_step_anchor = self.physics_step_count
        self.stats_command_anchor = command_sequence
        self.stats_command_change_anchor = command_change_sequence
        local_gyro = self.sim_env.get_base_imu_velocity()[3:6]
        free_joint_gyro = np.asarray(self.sim_env.mj_data.qvel[3:6])
        gyro_frame_delta = float(np.linalg.norm(local_gyro - free_joint_gyro))
        target_range = (
            f"[{targets.min():+.3f},{targets.max():+.3f}]"
            if targets.size
            else "unavailable"
        )
        gain_range = (
            f"kp=[{gains[:, 0].min():.2f},{gains[:, 0].max():.2f}] "
            f"kd=[{gains[:, 1].min():.2f},{gains[:, 1].max():.2f}]"
            if gains.size
            else "kp/kd=unavailable"
        )
        waist_joint_id = int(self.sim_env.body_joint_index[14])
        waist_actuator_id = waist_joint_id - 1
        waist_qpos_index = waist_joint_id + self.sim_env.qpos_offset - 1
        waist_detail = (
            f"waist_pitch(target={targets[14]:+.3f},"
            f"actual={self.sim_env.mj_data.qpos[waist_qpos_index]:+.3f},"
            f"ctrl={self.sim_env.mj_data.ctrl[waist_actuator_id]:+.3f},"
            f"force={self.sim_env.mj_data.actuator_force[waist_actuator_id]:+.3f})"
        )
        print(
            "[MuJoCo diagnostic] effective_sim_gains "
            f"mode={self.debug_mode} physics={self.physics_enabled} "
            f"harness={self.safety_harness_scale:.3f} "
            f"sim_time={self.sim_env.mj_data.time:.3f}s "
            f"sim_hz={simulation_hz:.1f} lowcmd_hz={command_hz:.1f} "
            f"target_change_hz={command_change_hz:.1f} "
            f"steps/lowcmd={steps_per_command:.2f} dropped={self.dropped_low_cmd_count} "
            f"pelvis_gyro_local_minus_qvel={gyro_frame_delta:.4f}rad/s "
            f"root_xyz=[{qpos[0]:+.3f},{qpos[1]:+.3f},{qpos[2]:+.3f}] "
            f"root_vel=[{qvel[0]:+.3f},{qvel[1]:+.3f},{qvel[2]:+.3f}]m/s "
            f"rpy=[{np.degrees(roll):+.1f},{np.degrees(pitch):+.1f},{np.degrees(yaw):+.1f}]deg "
            f"actual_q=[{qpos[7:].min():+.3f},{qpos[7:].max():+.3f}]rad "
            f"target_q={target_range}rad {gain_range} {waist_detail}",
            flush=True,
        )

    def close(self) -> None:
        self.control.close(linger=0)
        self.telemetry.close(linger=0)
        self.context.term()
