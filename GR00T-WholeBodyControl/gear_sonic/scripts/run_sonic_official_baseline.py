#!/usr/bin/env python3
"""Run the repository's local SONIC reference path and record stability metrics.

This intentionally uses BaseSimulator and DDS directly.  It does not create a
web bridge, an ARDY publisher, or a ZMQ motion input.  Touch ``release_file``
after starting the C++ controller to apply the exact MuJoCo key-9 elastic-band
toggle.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig


@dataclass
class OfficialBaselineConfig(SimLoopConfig):
    release_file: str = "/tmp/sonic_official_baseline_release"
    """Touch this file to perform the simulator's key-9 release."""

    result_file: str = "/data3/hy/logs/sonic_official_baseline_result.json"
    """Machine-readable stability result."""

    duration_after_release: float = 20.0
    """Seconds to run after release before declaring the trial complete."""

    startup_timeout: float = 300.0
    """Maximum time to wait for the release signal."""

    report_period: float = 1.0
    """Console diagnostic interval in seconds."""


def quat_to_rpy_degrees(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz / max(float(np.linalg.norm(quat_wxyz)), 1e-12)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def body_joint_state(env) -> tuple[np.ndarray, np.ndarray]:
    q_indices = env.body_joint_index + env.qpos_offset - 1
    dq_indices = env.body_joint_index + env.qvel_offset - 1
    q = np.asarray(env.mj_data.qpos[q_indices], dtype=np.float64)
    dq = np.asarray(env.mj_data.qvel[dq_indices], dtype=np.float64)
    if q.shape != (29,) or dq.shape != (29,):
        raise RuntimeError(f"Expected 29 body joints, got {q.shape}/{dq.shape}")
    return q, dq


def main(config: OfficialBaselineConfig) -> None:
    config.enable_onscreen = False
    config.enable_offscreen = False
    config.enable_image_publish = False
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name
    wbc_config["AUTO_RESET_ON_FALL"] = False

    release_path = Path(config.release_file)
    result_path = Path(config.result_file)
    release_path.unlink(missing_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    simulator = BaseSimulator(config=wbc_config, env_name=config.env_name, onscreen=False)
    env = simulator.sim_env
    original_sim_step = env.sim_step
    wall_started = time.monotonic()
    released_at: float | None = None
    next_report = wall_started
    release_root_xy: np.ndarray | None = None
    minimum_height = float("inf")
    maximum_abs_rpy = np.zeros(3, dtype=np.float64)
    maximum_tracking_rms = 0.0
    maximum_joint_speed = 0.0
    fallen = False
    samples = 0

    print("[Official baseline] ARDY=OFF ZMQ motion=OFF web wrapper=OFF", flush=True)
    print(f"[Official baseline] waiting for key-9 signal: {release_path}", flush=True)

    def instrumented_step() -> None:
        nonlocal released_at, next_report, release_root_xy
        nonlocal minimum_height, maximum_abs_rpy, maximum_tracking_rms
        nonlocal maximum_joint_speed, fallen, samples

        now = time.monotonic()
        if released_at is None and release_path.exists():
            env.handle_keyboard_button("9")
            release_path.unlink(missing_ok=True)
            released_at = now
            release_root_xy = np.asarray(env.mj_data.qpos[:2], dtype=np.float64).copy()
            print("[Official baseline] applied exact key-9 elastic-band toggle", flush=True)

        original_sim_step()
        q, dq = body_joint_state(env)
        root = np.asarray(env.mj_data.qpos[:3], dtype=np.float64)
        rpy = quat_to_rpy_degrees(np.asarray(env.mj_data.qpos[3:7], dtype=np.float64))
        commands = env.unitree_bridge.low_cmd.motor_cmd[:29]
        targets = np.asarray([command.q for command in commands], dtype=np.float64)
        tracking_rms = float(np.degrees(np.sqrt(np.mean(np.square(targets - q)))))

        minimum_height = min(minimum_height, float(root[2]))
        maximum_abs_rpy = np.maximum(maximum_abs_rpy, np.abs(rpy))
        maximum_tracking_rms = max(maximum_tracking_rms, tracking_rms)
        maximum_joint_speed = max(maximum_joint_speed, float(np.max(np.abs(dq))))
        fallen = fallen or bool(env.fall) or float(root[2]) < 0.2
        samples += 1

        now = time.monotonic()
        if now >= next_report:
            next_report = now + config.report_period
            displacement = (
                0.0
                if release_root_xy is None
                else float(np.linalg.norm(root[:2] - release_root_xy))
            )
            print(
                "[Official baseline] "
                f"sim_time={env.mj_data.time:.3f}s "
                f"root_xyz=[{root[0]:+.3f},{root[1]:+.3f},{root[2]:+.3f}] "
                f"rpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]deg "
                f"command_tracking_rms={tracking_rms:.2f}deg "
                f"root_displacement={displacement:.3f}m fallen={fallen}",
                flush=True,
            )

        if released_at is None and now - wall_started >= config.startup_timeout:
            print("[Official baseline] release signal timeout", flush=True)
            simulator._running = False
        elif released_at is not None and now - released_at >= config.duration_after_release:
            simulator._running = False

    env.sim_step = instrumented_step
    try:
        simulator.start()
    finally:
        root = np.asarray(env.mj_data.qpos[:3], dtype=np.float64)
        rpy = quat_to_rpy_degrees(np.asarray(env.mj_data.qpos[3:7], dtype=np.float64))
        displacement = (
            None
            if release_root_xy is None
            else float(np.linalg.norm(root[:2] - release_root_xy))
        )
        result = {
            "path": "official_local_keyboard_reference",
            "released": released_at is not None,
            "duration_after_release_s": (
                0.0 if released_at is None else time.monotonic() - released_at
            ),
            "samples": samples,
            "final_root_xyz_m": root.tolist(),
            "final_root_rpy_deg": rpy.tolist(),
            "root_displacement_xy_m": displacement,
            "minimum_root_height_m": minimum_height,
            "maximum_abs_rpy_deg": maximum_abs_rpy.tolist(),
            "maximum_command_tracking_rms_deg": maximum_tracking_rms,
            "maximum_abs_joint_velocity_rad_s": maximum_joint_speed,
            "fallen": fallen,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        release_path.unlink(missing_ok=True)
        print(f"[Official baseline result] {json.dumps(result)}", flush=True)
        print(f"[Official baseline] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(OfficialBaselineConfig))
