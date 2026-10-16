"""Publish ARDY G1 reference motion to SONIC and the local web viewer."""

from __future__ import annotations

import json
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import zmq
from websockets.sync.client import connect as ws_connect
from .control_planning import live_seam_limits


WEB_MAGIC = b"ASB1"
ZMQ_HEADER_SIZE = 1280
DEBUG_MODES = {
    "reference_only",
    "sonic_official",
    "ardy_offline",
    "ardy_streaming",
}

# MujocoQposConverter writes qpos joints in this explicit order.  Keep names
# next to the reorder indices so an XML/order change fails loudly instead of
# silently driving a different joint.
MUJOCO_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/g1.py:
# G1_MUJOCO_TO_ISAACLAB_DOF. Tests compare this value with the upstream source.
MUJOCO_TO_ISAACLAB = np.asarray(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)
ISAACLAB_TO_MUJOCO = np.argsort(MUJOCO_TO_ISAACLAB)
SONIC_REFERENCE_JOINT_NAMES = tuple(
    MUJOCO_BODY_JOINT_NAMES[index] for index in MUJOCO_TO_ISAACLAB
)

if len(MUJOCO_BODY_JOINT_NAMES) != 29:
    raise RuntimeError("The ARDY/MuJoCo body-joint contract must contain 29 joints.")
if sorted(MUJOCO_TO_ISAACLAB.tolist()) != list(range(29)):
    raise RuntimeError("MUJOCO_TO_ISAACLAB must be a complete 29-joint permutation.")
if len(set(SONIC_REFERENCE_JOINT_NAMES)) != 29:
    raise RuntimeError("SONIC reference joint names must be unique.")


def joint_mapping_rows() -> list[dict[str, object]]:
    """Return the explicit ARDY -> SONIC -> MuJoCo reference contract."""
    return [
        {
            "index": sonic_index,
            "ardy_index": int(mujoco_index),
            "ardy_joint": MUJOCO_BODY_JOINT_NAMES[int(mujoco_index)],
            "sonic_joint": SONIC_REFERENCE_JOINT_NAMES[sonic_index],
            "mujoco_joint": MUJOCO_BODY_JOINT_NAMES[int(mujoco_index)],
            "actuator": MUJOCO_BODY_JOINT_NAMES[int(mujoco_index)].removesuffix(
                "_joint"
            ),
            "sign": 1.0,
            "scale": 1.0,
            "offset": 0.0,
        }
        for sonic_index, mujoco_index in enumerate(MUJOCO_TO_ISAACLAB)
    ]


def _wrapped_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(left - right), np.cos(left - right))


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    return float(
        Rotation.from_quat(quaternion, scalar_first=True).as_euler("xyz")[2]
    )


def motion_statistics(qpos: Any, fps: float) -> dict[str, object]:
    """Validate units and return named per-joint/reference diagnostics."""
    motion = validate_qpos(qpos)
    quaternions = normalize_quaternions_wxyz(motion[:, 3:7])
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0))
    )
    joints = motion[:, 7:]
    root_xy_speed = np.linalg.norm(np.diff(motion[:, :2], axis=0), axis=1) * fps
    joint_speed = np.abs(np.diff(np.unwrap(joints, axis=0), axis=0)) * fps
    root_yaw = np.unwrap(
        Rotation.from_quat(quaternions, scalar_first=True).as_euler("xyz")[:, 2]
    )
    root_yaw_speed = np.abs(np.diff(root_yaw)) * fps
    jumps = (
        np.max(np.abs(_wrapped_delta(joints[1:], joints[:-1])), axis=0)
        if len(joints) > 1
        else np.zeros(29)
    )
    per_joint = []
    for index, name in enumerate(MUJOCO_BODY_JOINT_NAMES):
        values = joints[:, index]
        per_joint.append(
            {
                "index": index,
                "name": name,
                "min_rad": float(values.min()),
                "max_rad": float(values.max()),
                "mean_rad": float(values.mean()),
                "first_rad": float(values[0]),
                "max_step_rad": float(jumps[index]),
            }
        )
    return {
        "frames": len(motion),
        "fps": float(fps),
        "duration_s": (len(motion) - 1) / float(fps),
        "root_start_xyz": motion[0, :3].tolist(),
        "root_end_xyz": motion[-1, :3].tolist(),
        "root_displacement_m": float(np.linalg.norm(motion[-1, :2] - motion[0, :2])),
        "root_xy_speed_mps": {
            "mean": float(np.mean(root_xy_speed)),
            "p95": float(np.percentile(root_xy_speed, 95)),
            "max": float(np.max(root_xy_speed)),
        },
        "joint_abs_speed_radps": {
            "p95": float(np.percentile(joint_speed, 95)),
            "p99": float(np.percentile(joint_speed, 99)),
            "max": float(np.max(joint_speed)),
        },
        "root_yaw_abs_speed_radps": {
            "p95": float(np.percentile(root_yaw_speed, 95)),
            "max": float(np.max(root_yaw_speed)),
        },
        "quaternion_norm_error_max": quaternion_norm_error,
        "quaternion_sign_flips": int(
            np.count_nonzero(np.sum(motion[:-1, 3:7] * motion[1:, 3:7], axis=1) < 0)
        ),
        "joint_unit": "radian",
        "joint_velocity_unit": "radian/second",
        "per_joint": per_joint,
    }


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy/CPU/CUDA tensors without importing torch here."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def normalize_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternions, dtype=np.float64).copy()
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape [T, 4], got {quat.shape}.")
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Root quaternion contains a near-zero norm.")
    quat /= norms
    for index in range(1, len(quat)):
        if float(np.dot(quat[index - 1], quat[index])) < 0.0:
            quat[index] *= -1.0
    return quat


def validate_qpos(qpos: Any) -> np.ndarray:
    qpos = _as_numpy(qpos)
    if qpos.ndim == 3:
        if qpos.shape[0] != 1:
            raise ValueError(f"Expected one motion, got batch shape {qpos.shape}.")
        qpos = qpos[0]
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos shape [T, 36], got {qpos.shape}.")
    if qpos.shape[0] < 2:
        raise ValueError("Motion must contain at least two frames.")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains NaN or Inf.")
    normalize_quaternions_wxyz(qpos[:, 3:7])
    return np.ascontiguousarray(qpos, dtype=np.float64)


def resample_mujoco_qpos(
    qpos: Any,
    source_fps: float,
    target_fps: float = 50.0,
    time_stretch: float = 1.0,
) -> np.ndarray:
    """Resample qpos with one shared time scale for root, rotation, and joints."""
    return resample_mujoco_qpos_suffix(
        qpos,
        source_fps,
        target_fps,
        time_stretch,
        target_start_index=0,
    )


def resample_mujoco_qpos_suffix(
    qpos: Any,
    source_fps: float,
    target_fps: float = 50.0,
    time_stretch: float = 1.0,
    *,
    target_start_index: int = 0,
) -> np.ndarray:
    """Resample only ``[target_start_index, end]`` on the global time grid."""
    source = validate_qpos(qpos)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive.")
    if not np.isfinite(time_stretch) or time_stretch < 1.0:
        raise ValueError("time_stretch must be finite and at least 1.0.")

    if target_start_index < 0:
        raise ValueError("target_start_index must be non-negative")
    if np.isclose(source_fps, target_fps) and np.isclose(time_stretch, 1.0):
        # Avoid changing the representation of an already correctly sampled
        # trajectory. Quaternion q and -q encode the same rotation, but keeping
        # the producer's signs makes the 50 Hz path byte-for-byte predictable.
        output = np.ascontiguousarray(source[target_start_index:], dtype=np.float32)
        norms = np.linalg.norm(output[:, 3:7], axis=1, keepdims=True)
        output[:, 3:7] /= norms
        return output

    full_source_time = (
        np.arange(len(source), dtype=np.float64)
        / float(source_fps)
        * float(time_stretch)
    )
    duration = full_source_time[-1]
    target_count = int(np.floor(duration * target_fps + 1e-9)) + 1
    if target_start_index >= target_count:
        return np.empty((0, 36), dtype=np.float32)
    target_time = (
        np.arange(target_start_index, target_count, dtype=np.float64)
        / float(target_fps)
    )
    # Decimal FPS products can put the mathematically exact last sample a few
    # ulps past Slerp's inclusive endpoint.
    target_time = np.minimum(target_time, duration)
    output = np.empty((len(target_time), 36), dtype=np.float64)

    # Retain one source sample before the requested suffix so interpolation at
    # the boundary is identical to a full-history conversion.
    source_start = max(0, int(np.searchsorted(full_source_time, target_time[0])) - 1)
    source_time = full_source_time[source_start:]
    source = source[source_start:]

    for column in range(3):
        output[:, column] = np.interp(target_time, source_time, source[:, column])

    quat_wxyz = normalize_quaternions_wxyz(source[:, 3:7])
    rotations = Rotation.from_quat(quat_wxyz, scalar_first=True)
    output[:, 3:7] = Slerp(source_time, rotations)(target_time).as_quat(
        scalar_first=True
    )

    unwrapped_joints = np.unwrap(source[:, 7:], axis=0)
    for column in range(unwrapped_joints.shape[1]):
        output[:, 7 + column] = np.interp(
            target_time, source_time, unwrapped_joints[:, column]
        )
    return np.ascontiguousarray(output, dtype=np.float32)


def recommend_time_stretch(
    qpos: Any,
    source_fps: float,
    *,
    mean_root_speed_limit: float = 0.65,
    p95_root_speed_limit: float = 1.0,
    p99_joint_speed_limit: float = 5.5,
    p95_yaw_speed_limit: float = 1.30,
    maximum: float = 4.0,
) -> tuple[float, dict[str, object]]:
    """Choose a conservative SONIC playback scale from official-motion limits."""
    if source_fps <= 0:
        raise ValueError("source_fps must be positive.")
    limits = (
        mean_root_speed_limit,
        p95_root_speed_limit,
        p99_joint_speed_limit,
        p95_yaw_speed_limit,
        maximum,
    )
    if not all(np.isfinite(value) and value > 0 for value in limits):
        raise ValueError("Time-stretch speed limits must be finite and positive.")

    stats = motion_statistics(qpos, source_fps)
    required = max(
        1.0,
        stats["root_xy_speed_mps"]["mean"] / mean_root_speed_limit,
        stats["root_xy_speed_mps"]["p95"] / p95_root_speed_limit,
        stats["joint_abs_speed_radps"]["p99"] / p99_joint_speed_limit,
        stats["root_yaw_abs_speed_radps"]["p95"] / p95_yaw_speed_limit,
    )
    # Quantization prevents tiny floating-point/profile changes from changing
    # the resampled prefix when a streaming trajectory is extended.
    stretch = min(maximum, np.ceil(required * 20.0) / 20.0)
    return float(stretch), stats


def compute_sonic_joint_data(
    qpos: Any, fps: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MuJoCo joint order to the IsaacLab order consumed by SONIC."""
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    qpos_array = validate_qpos(qpos).astype(np.float32, copy=False)
    positions = np.ascontiguousarray(
        qpos_array[:, 7:][:, MUJOCO_TO_ISAACLAB], dtype=np.float32
    )
    # SONIC's official reference CSVs store the velocity at frame t as the
    # forward difference to frame t + 1. Matching that convention matters for
    # the policy's multi-frame velocity observations; np.gradient() uses a
    # central difference internally and produces materially different values.
    velocities = np.empty_like(positions, dtype=np.float64)
    velocities[:-1] = np.diff(positions.astype(np.float64), axis=0) * fps
    velocities[-1] = velocities[-2]
    return positions, np.ascontiguousarray(velocities, dtype=np.float32)


def pack_web_motion(
    qpos: Any,
    *,
    fps: float,
    name: str,
    current_frame: int,
    playing: bool,
    debug_mode: str = "ardy_streaming",
    time_stretch: float = 1.0,
    epoch_id: int = 0,
    source_time: float | None = None,
    frame_float: float | None = None,
    sequence_id: int | None = None,
    prompt_segments: list[dict[str, object]] | None = None,
    frame_origin: int = 0,
) -> bytes:
    data = validate_qpos(qpos).astype("<f4", copy=False)
    header = {
        "type": "motion",
        "version": 1,
        "dtype": "f32",
        "endian": "little",
        "shape": [int(data.shape[0]), 36],
        "fps": float(fps),
        "name": str(name),
        "current_frame": int(current_frame),
        "frame_origin": int(frame_origin),
        "playing": bool(playing),
        "debug_mode": str(debug_mode),
        "time_stretch": float(time_stretch),
        # Explicit clock contract. ``source_time`` is seconds on the ARDY
        # source timeline; monotonic_timestamp is producer host time.
        "epoch_id": int(epoch_id),
        "source_time": float(
            current_frame / fps if source_time is None else source_time
        ),
        "frame_float": float(
            current_frame if frame_float is None else frame_float
        ),
        "monotonic_timestamp": float(time.monotonic()),
        "sequence_id": int(current_frame if sequence_id is None else sequence_id),
        "layout": "mujoco_qpos_root_xyz_root_quat_wxyz_29joints",
    }
    if prompt_segments:
        header["prompt_segments"] = prompt_segments
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return WEB_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + data.tobytes()


def pack_web_control(action: str, **values: object) -> str:
    if "startup_epoch" in values and "epoch_id" not in values:
        values["epoch_id"] = values["startup_epoch"]
    return json.dumps(
        {"type": "control", "version": 1, "action": action, **values},
        separators=(",", ":"),
    )


def _dtype_name(value: np.ndarray) -> str:
    names = {
        np.dtype("float32"): "f32",
        np.dtype("float64"): "f64",
        np.dtype("int32"): "i32",
        np.dtype("int64"): "i64",
        np.dtype("uint8"): "u8",
        np.dtype("bool"): "bool",
    }
    try:
        return names[value.dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported ZMQ dtype: {value.dtype}") from exc


def pack_zmq_message(
    fields_by_name: dict[str, np.ndarray], *, topic: str, version: int = 1
) -> bytes:
    fields: list[dict[str, object]] = []
    payloads: list[bytes] = []
    count = 1
    for name, value in fields_by_name.items():
        array = np.ascontiguousarray(value)
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"))
        fields.append(
            {"name": name, "dtype": _dtype_name(array), "shape": list(array.shape)}
        )
        payloads.append(array.tobytes(order="C"))
        if array.ndim >= 1:
            count = max(count, int(array.shape[0]))
    header = {"v": int(version), "endian": "le", "count": count, "fields": fields}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > ZMQ_HEADER_SIZE:
        raise ValueError(f"ZMQ header too large: {len(encoded)} bytes.")
    return topic.encode("utf-8") + encoded.ljust(ZMQ_HEADER_SIZE, b"\0") + b"".join(payloads)


class WebProducer:
    """Small reconnecting queue so inference never waits on a browser."""

    def __init__(self, url: str):
        self.url = url
        self._queue: queue.Queue[bytes | str | None] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._connection_lock = threading.Lock()
        self._connected = False
        self._latest_motion: bytes | None = None
        self._latest_control: str | None = None
        self._thread = threading.Thread(target=self._run, name="sonic-web", daemon=True)
        self._thread.start()

    def send(self, payload: bytes | str) -> None:
        with self._connection_lock:
            if isinstance(payload, bytes):
                self._latest_motion = payload
            else:
                self._latest_control = payload
            connected = self._connected
        # A new connection replays the latest motion/control snapshots. Queue
        # only while connected so the same restart is not replayed twice.
        if not connected:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(payload)

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        retry_delay = 0.5
        while not self._stop.is_set():
            try:
                with ws_connect(
                    self.url, open_timeout=5, max_size=None, proxy=None
                ) as websocket:
                    retry_delay = 0.5
                    with self._connection_lock:
                        latest_motion = self._latest_motion
                        latest_control = self._latest_control
                        self._connected = True
                    if latest_motion is not None:
                        websocket.send(latest_motion)
                    if latest_control is not None:
                        websocket.send(latest_control)
                    while not self._stop.is_set():
                        try:
                            payload = self._queue.get(timeout=1.0)
                        except queue.Empty:
                            pong = websocket.ping()
                            if not pong.wait(timeout=2.0):
                                raise TimeoutError("web bridge ping timed out")
                            continue
                        if payload is None:
                            return
                        websocket.send(payload)
            except Exception as exc:
                with self._connection_lock:
                    self._connected = False
                print(
                    f"[ARDY->SONIC web] disconnected: {exc}; "
                    f"retrying in {retry_delay:.1f}s"
                )
                self._stop.wait(retry_delay)
                retry_delay = min(retry_delay * 2.0, 5.0)


class ZmqPublisher:
    """Official SONIC packed-message publisher for pose and manager commands."""

    def __init__(self, bind: str, pose_topic: str = "pose"):
        self.pose_topic = pose_topic
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 4)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(bind)
        print(f"[ARDY->SONIC ZMQ] bound {bind}, pose topic={pose_topic}")

    def send_pose(
        self,
        *,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        body_pos: np.ndarray,
        body_quat_wxyz: np.ndarray,
        frame_index: int | np.ndarray,
        catch_up: bool = False,
        stream_complete: bool = False,
    ) -> None:
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        body_pos = np.asarray(body_pos, dtype=np.float32)
        body_quat_wxyz = np.asarray(body_quat_wxyz, dtype=np.float32)
        if joint_pos.ndim == 1:
            joint_pos = joint_pos[None, :]
            joint_vel = joint_vel[None, :]
        if body_pos.ndim == 1:
            body_pos = body_pos[None, :]
        if body_quat_wxyz.ndim == 1:
            body_quat_wxyz = body_quat_wxyz[None, :]
        frame_indices = np.atleast_1d(np.asarray(frame_index, dtype=np.int64))
        frame_count = joint_pos.shape[0]
        if (
            joint_pos.shape != (frame_count, 29)
            or joint_vel.shape != (frame_count, 29)
            or body_pos.ndim not in (2, 3)
            or body_pos.shape[0] != frame_count
            or body_pos.shape[-1] != 3
            or body_quat_wxyz.ndim not in (2, 3)
            or body_quat_wxyz.shape[0] != frame_count
            or body_quat_wxyz.shape[-1] != 4
            or frame_indices.shape != (frame_count,)
        ):
            raise ValueError(
                "SONIC pose fields have inconsistent batch shapes: "
                f"joint_pos={joint_pos.shape}, joint_vel={joint_vel.shape}, "
                f"body_pos={body_pos.shape}, body_quat={body_quat_wxyz.shape}, "
                f"frame_index={frame_indices.shape}"
            )
        fields = {
            "joint_pos": np.ascontiguousarray(joint_pos),
            "joint_vel": np.ascontiguousarray(joint_vel),
            "body_pos": np.ascontiguousarray(body_pos),
            "body_quat": np.ascontiguousarray(body_quat_wxyz),
            "frame_index": frame_indices,
            "catch_up": np.asarray([catch_up], dtype=np.bool_),
            "stream_complete": np.asarray([stream_complete], dtype=np.bool_),
            "timestamp_monotonic": np.asarray([time.monotonic()], dtype=np.float64),
        }
        self.socket.send(
            pack_zmq_message(fields, topic=self.pose_topic), flags=zmq.NOBLOCK
        )

    def send_command(
        self,
        *,
        start: bool,
        stop: bool,
        planner: bool = False,
        play: bool = False,
        restart: bool = False,
    ) -> None:
        fields = {
            "start": np.asarray([start], dtype=np.bool_),
            "stop": np.asarray([stop], dtype=np.bool_),
            "planner": np.asarray([planner], dtype=np.bool_),
            "play": np.asarray([play], dtype=np.bool_),
            "restart": np.asarray([restart], dtype=np.bool_),
        }
        self.socket.send(pack_zmq_message(fields, topic="command"), flags=zmq.NOBLOCK)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


@dataclass
class MotionState:
    qpos: np.ndarray | None = None
    canonical_qpos: np.ndarray | None = None
    joint_pos: np.ndarray | None = None
    joint_vel: np.ndarray | None = None
    frame: int = 0
    playing: bool = False
    name: str = "ARDY Live"
    prompt_segments: tuple[dict[str, object], ...] = ()
    epoch: int = 0
    revision: int = 0
    controller_ready_not_before: float = 0.0
    controller_restart_pending: bool = False
    chunk_id: int = 0
    updated_at: float = 0.0
    time_stretch: float = 1.0
    stream_complete: bool = False
    frame_origin: int = 0
    source_next: int = 0
    source_tail: np.ndarray | None = None
    consumer_ack_epoch: int = -1
    consumer_received_until: int = 0
    replacement_start: int | None = None
    consumer_clock_epoch: int = -1
    consumer_clock_observed_at: float = 0.0



class SonicOutputBridge:
    """Single rank-zero adapter shared by all ARDY demo callbacks."""

    def __init__(
        self,
        *,
        web_url: str | None = "ws://127.0.0.1:8080/ws/producer",
        zmq_bind: str | None = "tcp://127.0.0.1:5556",
        zmq_topic: str = "pose",
        target_fps: float = 50.0,
        debug_mode: str = "ardy_streaming",
        time_stretch: float | str = 1.0,
        catch_up: bool = True,
    ):
        if debug_mode not in DEBUG_MODES:
            raise ValueError(
                f"Unknown ARDY/SONIC debug mode {debug_mode!r}; "
                f"expected one of {sorted(DEBUG_MODES)}."
            )
        self.target_fps = float(target_fps)
        self.debug_mode = debug_mode
        self.catch_up = bool(catch_up)
        if isinstance(time_stretch, str) and time_stretch.strip().lower() == "auto":
            self._time_stretch_setting: float | str = "auto"
        else:
            try:
                fixed_time_stretch = float(time_stretch)
            except (TypeError, ValueError) as exc:
                raise ValueError("time_stretch must be 'auto' or a number >= 1") from exc
            if not np.isfinite(fixed_time_stretch) or fixed_time_stretch < 1.0:
                raise ValueError("time_stretch must be 'auto' or a number >= 1")
            self._time_stretch_setting = fixed_time_stretch
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state = MotionState()
        # A bounded Native cache cannot rewind itself to an evicted frame.
        # Its owner reloads the accepted trajectory from the existing archive.
        self.source_control = None
        self.stream_error = None
        self.web = WebProducer(web_url) if web_url else None
        self._zmq_bind = zmq_bind
        self._zmq_topic = zmq_topic
        self._controller_restart_seconds = float(
            os.environ.get("ARDY_SONIC_CONTROLLER_RESTART_SECONDS", "3.2")
        )
        self._zmq_commands: queue.Queue[tuple[bool, bool, bool, bool]] = (
            queue.Queue(maxsize=8)
        )
        self.zmq: ZmqPublisher | None = None
        self._thread = threading.Thread(
            target=self._stream_loop, name="sonic-reference-50hz", daemon=True
        )
        self._thread.start()
        print(
            f"[ARDY->SONIC contract] mode={self.debug_mode}, "
            "ARDY qpos=36 (root xyz + quaternion wxyz + 29 radian joints), "
            "SONIC reference=29 joints in IsaacLab order, target=50 Hz, "
            f"time_stretch={self._time_stretch_setting}"
        )
        for row in joint_mapping_rows():
            print(
                "[ARDY->SONIC mapping] "
                f"{row['index']:02d} ardy[{row['ardy_index']:02d}]={row['ardy_joint']} "
                f"-> sonic={row['sonic_joint']} -> mujoco={row['mujoco_joint']} "
                f"-> actuator={row['actuator']} sign=+1 scale=1 offset=0"
            )

    @classmethod
    def from_env(cls) -> "SonicOutputBridge | None":
        enabled = os.environ.get("ARDY_SONIC_ENABLE", "0") == "1"
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if not enabled or rank != 0:
            if enabled:
                print(f"[ARDY->SONIC] disabled on worker rank {rank}")
            return None
        debug_mode = os.environ.get(
            "ARDY_SONIC_DEBUG_MODE", "ardy_streaming"
        ).strip()
        zmq_bind = os.environ.get(
            "ARDY_SONIC_ZMQ_BIND", "tcp://127.0.0.1:5556"
        ) or None
        if debug_mode == "reference_only":
            zmq_bind = None
        return cls(
            web_url=os.environ.get(
                "ARDY_SONIC_WEB_URL", "ws://127.0.0.1:8080/ws/producer"
            )
            or None,
            zmq_bind=zmq_bind,
            zmq_topic=os.environ.get("ARDY_SONIC_ZMQ_TOPIC", "pose"),
            target_fps=float(os.environ.get("ARDY_SONIC_FPS", "50")),
            debug_mode=debug_mode,
            time_stretch=os.environ.get("ARDY_SONIC_TIME_STRETCH", "auto"),
        )

    def _select_time_stretch(
        self, qpos: np.ndarray, source_fps: float
    ) -> tuple[float, dict[str, object]]:
        if self._time_stretch_setting != "auto":
            return float(self._time_stretch_setting), motion_statistics(qpos, source_fps)
        return recommend_time_stretch(
            qpos,
            source_fps,
            mean_root_speed_limit=float(
                os.environ.get("ARDY_SONIC_MEAN_ROOT_SPEED_LIMIT_MPS", "0.65")
            ),
            p95_root_speed_limit=float(
                os.environ.get("ARDY_SONIC_P95_ROOT_SPEED_LIMIT_MPS", "1.0")
            ),
            p99_joint_speed_limit=float(
                os.environ.get("ARDY_SONIC_P99_JOINT_SPEED_LIMIT_RADPS", "5.5")
            ),
            p95_yaw_speed_limit=float(
                os.environ.get("ARDY_SONIC_P95_YAW_SPEED_LIMIT_RADPS", "1.30")
            ),
            maximum=float(os.environ.get("ARDY_SONIC_MAX_TIME_STRETCH", "4.0")),
        )

    @staticmethod
    def _coherent_pose_window(qpos, joint_pos, joint_vel, *, context):
        """Return one frame-aligned pose packet, repairing stale derived arrays.

        qpos is the source of truth.  A restart/append can replace qpos while a
        derived velocity array from the previous window is still visible to a
        publisher snapshot.  Recompute both derived fields rather than killing
        the 50 Hz stream thread on a transient shape mismatch.
        """
        qpos = np.asarray(qpos)
        joint_pos = np.asarray(joint_pos)
        joint_vel = np.asarray(joint_vel)
        count = len(qpos)
        if (joint_pos.ndim != 2 or joint_vel.ndim != 2
                or joint_pos.shape[0] != count or joint_vel.shape[0] != count):
            print(
                "[ARDY->SONIC] repaired stale derived pose arrays "
                f"context={context} qpos={qpos.shape} "
                f"joint_pos={joint_pos.shape} joint_vel={joint_vel.shape}",
                flush=True,
            )
            joint_pos, joint_vel = compute_sonic_joint_data(qpos, 50.0)
        return qpos, np.ascontiguousarray(joint_pos), np.ascontiguousarray(joint_vel)

    def replace_motion(
        self,
        qpos: Any,
        *,
        source_fps: float,
        source_frame: int = 0,
        name: str = "ARDY Live",
        prompt_segments: list[dict[str, object]] | None = None,
        stream_complete: bool = False,
    ) -> None:
        source_qpos = validate_qpos(qpos)
        with self._lock:
            existing_qpos = self._state.qpos
            existing_canonical_qpos = self._state.canonical_qpos
            existing_joint_pos = self._state.joint_pos
            existing_joint_vel = self._state.joint_vel
            existing_stretch = self._state.time_stretch
        if existing_qpos is None:
            time_stretch, source_stats = self._select_time_stretch(
                source_qpos, source_fps
            )
            print(
                "[ARDY->SONIC retime] "
                f"selected={time_stretch:.2f}x setting={self._time_stretch_setting} "
                f"root_mean={source_stats['root_xy_speed_mps']['mean']:.3f}m/s "
                f"root_p95={source_stats['root_xy_speed_mps']['p95']:.3f}m/s "
                f"joint_p99={source_stats['joint_abs_speed_radps']['p99']:.3f}rad/s "
                f"yaw_p95={source_stats['root_yaw_abs_speed_radps']['p95']:.3f}rad/s"
            )
        else:
            # Streaming extensions must keep the epoch's initial scale so the
            # already-consumed prefix remains byte-for-byte stable.
            time_stretch = existing_stretch
        conversion_started = time.perf_counter()
        target_count = int(
            np.floor(
                (len(source_qpos) - 1)
                / float(source_fps)
                * float(time_stretch)
                * self.target_fps
                + 1e-9
            )
        ) + 1
        suffix_start = 0
        incremental = (
            self.debug_mode == "ardy_streaming"
            and existing_qpos is not None
            and target_count > len(existing_qpos)
        )
        if incremental:
            suffix_start = max(0, len(existing_qpos) - 2)
            converted_suffix = resample_mujoco_qpos_suffix(
                source_qpos,
                source_fps,
                self.target_fps,
                time_stretch=time_stretch,
                target_start_index=suffix_start,
            )
            canonical_prefix = (
                existing_canonical_qpos
                if existing_canonical_qpos is not None
                else existing_qpos
            )
            canonical_qpos50 = np.concatenate(
                (canonical_prefix[:suffix_start], converted_suffix), axis=0
            )
            if np.isclose(source_fps, self.target_fps) and np.isclose(
                time_stretch, 1.0
            ):
                # At 1:1 sampling the incoming array is already the canonical
                # trajectory; retain mutable-tail diagnostics without another
                # conversion pass.
                canonical_qpos50 = np.ascontiguousarray(
                    source_qpos, dtype=np.float32
                )
            qpos50 = np.concatenate(
                (existing_qpos[:suffix_start], converted_suffix), axis=0
            )
        else:
            qpos50 = resample_mujoco_qpos(
                source_qpos,
                source_fps,
                self.target_fps,
                time_stretch=time_stretch,
            )
            canonical_qpos50 = qpos50
        conversion_ms = (time.perf_counter() - conversion_started) * 1000.0
        frame = int(
            round(source_frame * self.target_fps / source_fps * time_stretch)
        )
        frame = max(0, min(frame, len(qpos50) - 1))
        frame_scale = self.target_fps / float(source_fps) * float(time_stretch)
        target_prompt_segments: list[dict[str, object]] = []
        for segment in prompt_segments or []:
            try:
                text = str(segment["text"])
                source_start = max(0, int(segment["start_frame"]))
                source_end = max(source_start, int(segment["end_frame"]))
            except (KeyError, TypeError, ValueError):
                continue
            target_start = int(round(source_start * frame_scale))
            target_end = int(round((source_end + 1) * frame_scale)) - 1
            target_start = max(0, min(target_start, len(qpos50) - 1))
            target_end = max(target_start, min(target_end, len(qpos50) - 1))
            target_prompt_segments.append(
                {
                    "text": text,
                    "start_frame": target_start,
                    "end_frame": target_end,
                }
            )
        now = time.monotonic()
        with self._lock:
            previous_qpos = self._state.qpos
            previous_canonical_qpos = self._state.canonical_qpos
            self._state.stream_complete = bool(stream_complete)
            previous_frame = self._state.frame
            if self.debug_mode == "ardy_offline" and previous_qpos is not None:
                print(
                    "[ARDY->SONIC offline] ignored rolling reference update: "
                    f"frozen={len(previous_qpos)} incoming={len(qpos50)} frames"
                )
                return
            if (
                self.debug_mode == "ardy_streaming"
                and previous_qpos is not None
                and len(qpos50) <= len(previous_qpos)
            ):
                # Restart is an explicit control operation that clears qpos.
                # A rolling update can therefore never replace or shrink the
                # trajectory that SONIC has already consumed.
                self._state.canonical_qpos = canonical_qpos50
                print(
                    "[ARDY->SONIC streaming] ignored non-growing update: "
                    f"published={len(previous_qpos)} incoming={len(qpos50)} frames"
                )
                return

            prefix_joint_delta = float("inf")
            prefix_root_delta = float("inf")
            prefix_compare_count = 0
            comparison_qpos = (
                previous_canonical_qpos
                if previous_canonical_qpos is not None
                else previous_qpos
            )
            if comparison_qpos is not None and len(qpos50) >= len(comparison_qpos):
                # ARDY replans a short overlap at the old live edge. That tail
                # is intentionally mutable; continuity is established from the
                # older frozen prefix, then _stabilize_extension keeps every
                # already-published frame and joins only the genuinely new tail.
                mutable_tail_frames = int(
                    np.ceil(
                        float(
                            os.environ.get(
                                "ARDY_SONIC_REPLAN_MUTABLE_TAIL_SECONDS", "0.5"
                            )
                        )
                        * self.target_fps
                    )
                )
                prefix_compare_count = max(
                    2, len(comparison_qpos) - mutable_tail_frames
                )
                prefix_joint_delta = float(
                    np.max(
                        np.abs(
                            _wrapped_delta(
                                qpos50[:prefix_compare_count, 7:],
                                comparison_qpos[:prefix_compare_count, 7:],
                            )
                        )
                    )
                )
                prefix_root_delta = float(
                    np.max(
                        np.linalg.norm(
                            qpos50[:prefix_compare_count, :3]
                            - comparison_qpos[:prefix_compare_count, :3],
                            axis=1,
                        )
                    )
                )
            has_extension = (
                previous_qpos is not None
                and len(qpos50) > len(previous_qpos)
            )
            frame_is_forward = (
                previous_qpos is not None
                and frame >= max(0, previous_frame - 10)
            )
            prefix_is_compatible = (
                prefix_joint_delta <= float(
                    os.environ.get("ARDY_SONIC_PREFIX_JOINT_TOLERANCE_RAD", "0.05")
                )
                and prefix_root_delta <= float(
                    os.environ.get("ARDY_SONIC_PREFIX_ROOT_TOLERANCE_M", "0.02")
                )
            )
            # In streaming mode, only restart() may create a new motion epoch.
            # Prefix comparison remains diagnostic because ARDY is allowed to
            # rewrite its unpublished planning tail. SONIC's published prefix
            # stays immutable regardless of how far that replan reaches back.
            is_continuation = has_extension and (
                self.debug_mode == "ardy_streaming"
                or (frame_is_forward and prefix_is_compatible)
            )
            if is_continuation:
                qpos50 = self._stabilize_extension(previous_qpos, qpos50)
            if (
                incremental
                and suffix_start > 0
                and existing_joint_pos is not None
                and existing_joint_vel is not None
            ):
                suffix_joint_pos, suffix_joint_vel = compute_sonic_joint_data(
                    qpos50[suffix_start:], self.target_fps
                )
                joint_pos = np.concatenate(
                    (existing_joint_pos[:suffix_start], suffix_joint_pos), axis=0
                )
                joint_vel = np.concatenate(
                    (existing_joint_vel[:suffix_start], suffix_joint_vel), axis=0
                )
            else:
                joint_pos, joint_vel = compute_sonic_joint_data(
                    qpos50, self.target_fps
                )
            playing = self._state.playing
            self._state.qpos = qpos50
            self._state.canonical_qpos = canonical_qpos50
            self._state.joint_pos = joint_pos
            self._state.joint_vel = joint_vel
            self._state.frame = frame
            self._state.name = name
            self._state.prompt_segments = tuple(target_prompt_segments)
            self._state.revision += 1
            self._state.chunk_id += 1
            chunk_id = self._state.chunk_id
            previous_update_at = self._state.updated_at
            self._state.updated_at = now
            self._state.time_stretch = time_stretch
            if not is_continuation:
                self._state.epoch += 1
            epoch = self._state.epoch
        if self.web:
            self.web.send(
                pack_web_motion(
                    qpos50,
                    fps=self.target_fps,
                    name=name,
                    current_frame=frame,
                    playing=playing,
                    debug_mode=self.debug_mode,
                    time_stretch=time_stretch,
                    epoch_id=epoch,
                    source_time=frame / self.target_fps,
                    frame_float=float(frame),
                    sequence_id=self._state.revision,
                    prompt_segments=target_prompt_segments,
                )
            )
            self.web.send(
                pack_web_control(
                    "sync",
                    frame=frame,
                    fps=self.target_fps,
                    playing=playing,
                    debug_mode=self.debug_mode,
                    time_stretch=time_stretch,
                    startup_epoch=epoch,
                )
            )
        stats = motion_statistics(qpos50, self.target_fps)
        if not is_continuation:
            print(
                "[ARDY->SONIC motion] "
                f"epoch={epoch} chunk={chunk_id} frames={stats['frames']} "
                f"duration={stats['duration_s']:.3f}s unit=rad root_frame=MuJoCo/Z-up/+X-forward "
                f"root_start={stats['root_start_xyz']} root_end={stats['root_end_xyz']} "
                f"root_xy_displacement={stats['root_displacement_m']:.4f}m "
                f"quat=wxyz norm_error={stats['quaternion_norm_error_max']:.3e} "
                f"sign_flips={stats['quaternion_sign_flips']}"
            )
            for item in stats["per_joint"]:
                print(
                    "[ARDY->SONIC joint stats] "
                    f"{item['index']:02d} {item['name']} min={item['min_rad']:+.4f} "
                    f"max={item['max_rad']:+.4f} mean={item['mean_rad']:+.4f} "
                    f"first={item['first_rad']:+.4f} max_step={item['max_step_rad']:.4f} rad"
                )
        elif previous_qpos is not None:
            first_new = len(previous_qpos)
            joint_jump = float(
                np.max(
                    np.abs(
                        _wrapped_delta(
                            qpos50[first_new, 7:], previous_qpos[-1, 7:]
                        )
                    )
                )
            )
            root_jump = float(
                np.linalg.norm(qpos50[first_new, :3] - previous_qpos[-1, :3])
            )
            yaw_jump = abs(
                float(
                    np.arctan2(
                        np.sin(
                            _yaw_wxyz(qpos50[first_new, 3:7])
                            - _yaw_wxyz(previous_qpos[-1, 3:7])
                        ),
                        np.cos(
                            _yaw_wxyz(qpos50[first_new, 3:7])
                            - _yaw_wxyz(previous_qpos[-1, 3:7])
                        ),
                    )
                )
            )
            print(
                "[ARDY->SONIC chunk boundary] "
                f"chunk={chunk_id} old_length={len(previous_qpos)} new_length={len(qpos50)} "
                f"joint_jump={joint_jump:.5f}rad root_jump={root_jump:.5f}m "
                f"root_yaw_jump={yaw_jump:.5f}rad "
                f"prefix_joint_delta={prefix_joint_delta:.3e}rad "
                f"prefix_root_delta={prefix_root_delta:.3e}m "
                f"frozen_prefix_frames={prefix_compare_count} "
                f"timestamp_gap={max(0.0, now - previous_update_at):.3f}s "
                f"conversion_ms={conversion_ms:.2f} history_frames={len(source_qpos)} "
                f"suffix_target_frames={len(qpos50) - suffix_start}"
            )
        print(
            f"[ARDY->SONIC] reference "
            f"{'extended' if is_continuation else 'replaced'}: "
            f"{len(qpos50)} frames @ {self.target_fps:g} Hz, "
            f"time_stretch={time_stretch:.2f}x"
        )

    def acknowledge_source_reference(self, *, epoch, received_until):
        """Record matched-epoch consumer buffer coverage, never playback progress."""
        with self._lock:
            state = self._state
            if (epoch != state.epoch or not state.source_next or state.qpos is None
                    or received_until <= 0
                    or received_until > state.frame_origin + len(state.qpos)):
                return
            if state.consumer_ack_epoch != epoch:
                state.consumer_ack_epoch = epoch
                state.consumer_received_until = 0
            state.consumer_received_until = max(state.consumer_received_until, received_until)

    def observe_source_consumer(self, *, epoch, frame, observed_at, received_until):
        """The receiver owns this clock; no inference or render lock is needed."""
        with self._lock:
            state = self._state
            age = time.monotonic()-observed_at
            if (epoch != state.epoch or not state.source_next or state.qpos is None
                    or not 0 <= age < .15 or frame < state.frame_origin
                    or frame >= state.frame_origin+len(state.qpos)
                    or (state.consumer_clock_epoch == epoch
                        and observed_at < state.consumer_clock_observed_at)):
                return False
            state.consumer_clock_epoch = epoch
            state.consumer_clock_observed_at = observed_at
            state.frame = max(state.frame, int(frame))
            if received_until <= state.frame_origin+len(state.qpos):
                state.consumer_ack_epoch = epoch
                state.consumer_received_until = max(state.consumer_received_until, int(received_until))
            return True

    def unacknowledged_source_reference(self, epoch, published_until):
        """Copy only immutable, sent-but-unacknowledged Native reference frames.

        Repeating global indices is safe for the merger with catch_up=False.
        Never resend a shorter/older tail that could truncate a newer plan.
        """
        with self._lock:
            state = self._state
            if epoch != state.epoch or not state.source_next or state.qpos is None:
                return None
            start = (state.consumer_received_until if state.consumer_ack_epoch == epoch
                     else state.frame_origin)
            end = min(published_until, state.frame_origin + len(state.qpos))
            if start >= end:
                return None
            if start < state.frame_origin:
                raise RuntimeError('unacknowledged reference evicted before consumer receipt')
            lo, hi = start-state.frame_origin, end-state.frame_origin
            qpos_window, joint_pos_window, joint_vel_window = self._coherent_pose_window(
                state.qpos[lo:hi].copy(), state.joint_pos[lo:hi].copy(),
                state.joint_vel[lo:hi].copy(), context="retry"
            )
            return dict(joint_pos=joint_pos_window,
                        joint_vel=joint_vel_window,
                        body_pos=qpos_window[:, :3],
                        body_quat_wxyz=qpos_window[:, 3:7],
                        frame_index=np.arange(start, end, dtype=np.int64),
                        catch_up=False, stream_complete=state.stream_complete)

    def append_source_window(self, qpos, *, source_start, name, stream_complete=False):
        """Bounded append for MM2, no rewriting of published reference frames.

        25->50Hz, fixed 1x. Absolute indices survive local buffer eviction.
        The original ARDY replace_motion path remains available unchanged.
        """
        source = validate_qpos(qpos)
        if source.shape != (52, 36) or self.target_fps != 50 or self._time_stretch_setting != 1.0:
            raise ValueError('MM2 append requires 52 frames, 25->50FPS and time_stretch=1')
        with self._lock:
            state = self._state
            if source_start != state.source_next:
                raise ValueError(f'non-contiguous source append: {source_start} != {state.source_next}')
            if state.source_tail is None:
                converted = resample_mujoco_qpos(source, 25, 50, time_stretch=1.0)
            else:
                converted = resample_mujoco_qpos(np.concatenate((state.source_tail, source)), 25, 50, time_stretch=1.0)[1:]
            joints, velocities = compute_sonic_joint_data(converted, 50)
            if state.qpos is None:
                state.qpos, state.joint_pos, state.joint_vel = converted, joints, velocities
            else:
                # The previous tail used a provisional backward difference
                # because its next sample was not available yet. Match the
                # official forward-difference observation once that sample
                # arrives, but never rewrite an already consumed frame.
                seam = state.frame_origin + len(state.qpos) - 1
                if seam > state.frame:
                    state.joint_vel[-1] = (joints[0] - state.joint_pos[-1]) * self.target_fps
                    state.replacement_start = (seam if state.replacement_start is None
                                               else min(seam, state.replacement_start))
                    if state.consumer_ack_epoch == state.epoch:
                        state.consumer_received_until = min(state.consumer_received_until, seam)
                state.qpos = np.concatenate((state.qpos, converted))
                state.joint_pos = np.concatenate((state.joint_pos, joints))
                state.joint_vel = np.concatenate((state.joint_vel, velocities))
            # Retain 2s of consumed reference. Never remove the cursor or any
            # future reference; generation backpressure bounds that future.
            drop = max(0, min(state.frame-100-state.frame_origin, len(state.qpos)-2))
            if drop:
                state.qpos = state.qpos[drop:].copy()
                state.joint_pos = state.joint_pos[drop:].copy()
                state.joint_vel = state.joint_vel[drop:].copy()
                state.frame_origin += drop
            state.canonical_qpos = None
            # Carry the accepted quaternion sign and unwrapped joint branch;
            # each independently resampled raw window may choose +/-q anew.
            state.source_tail = converted[-1:].astype(np.float64).copy()
            state.source_next += 52
            state.name = name
            state.time_stretch = 1.0
            state.stream_complete = stream_complete
            state.revision += 1
            state.chunk_id += 1
            state.updated_at = time.monotonic()
            packet = pack_web_motion(state.qpos, fps=50, name=name,
                current_frame=state.frame, playing=state.playing, debug_mode=self.debug_mode,
                epoch_id=state.epoch, sequence_id=state.revision, frame_origin=state.frame_origin)
        if self.web:
            self.web.send(packet)

    def restore_source_playback(self, qpos, *, source_frame, name, playing=False,
                                stream_complete=False):
        """Seek/replay a saved Native trajectory using ARDY's single timeline.

        Reload the requested suffix even when its old reference was evicted.
        Generation state is owned by the RPC session and is not reset here.
        """
        source = validate_qpos(qpos)
        if self.target_fps != 50 or self._time_stretch_setting != 1.0:
            raise ValueError('Native replay requires 25->50FPS and time_stretch=1')
        frame = max(0, min(int(source_frame), len(source)-1)) * 2
        converted = resample_mujoco_qpos(source, 25, 50, time_stretch=1.0)
        # The UI can select the last loaded sample. Web motion validation
        # requires two frames, so retain its neighbour without moving the
        # selected cursor. Derivatives come from the complete saved motion.
        origin = min(frame, len(converted)-2)
        retained = converted[origin:].copy()
        full_joints, full_velocities = compute_sonic_joint_data(converted, 50)
        joints, velocities = full_joints[origin:].copy(), full_velocities[origin:].copy()
        with self._lock:
            state = self._state
            state.qpos, state.joint_pos, state.joint_vel = retained, joints, velocities
            state.canonical_qpos = None
            state.frame_origin, state.frame = origin, frame
            state.source_next = len(source)
            state.source_tail = converted[-1:].astype(np.float64).copy()
            state.replacement_start = None
            state.epoch += 1
            # Earlier indices are deliberately skipped by this seek; retry
            # only the requested suffix, not the retained preceding neighbour.
            state.consumer_ack_epoch = state.epoch
            state.consumer_received_until = frame
            state.revision += 1
            state.controller_restart_pending = True
            state.controller_ready_not_before = 0.0
            state.playing = bool(playing)
            state.name = name
            state.time_stretch = 1.0
            state.stream_complete = bool(stream_complete)
            state.updated_at = time.monotonic()
            epoch = state.epoch
            packet = pack_web_motion(retained, fps=50, name=name,
                current_frame=frame, playing=playing, debug_mode=self.debug_mode,
                epoch_id=epoch, sequence_id=state.revision, frame_origin=origin)
        self._send_manager_command(start=False, play=False)
        if self.web:
            self.web.send(packet)
            self.web.send(pack_web_control('seek', frame=frame, fps=50,
                debug_mode=self.debug_mode, time_stretch=1.0, startup_epoch=epoch))

    def replace_source_tail(self, qpos, *, source_start, name, guard_frames=8, commit=None):
        """Replace only unconsumed reference, preserving epoch and cursor.

        Caller must restore the matching generator history before continuing.
        The guard is checked again under the same lock as the actual splice.
        """
        source = validate_qpos(qpos)
        if source.shape[1] != 36 or len(source) < 2 or self.target_fps != 50:
            raise ValueError('invalid replacement source')
        with self._lock:
            state = self._state
            start = int(source_start) * 2
            age = time.monotonic()-state.consumer_clock_observed_at
            if state.playing and (state.consumer_clock_epoch != state.epoch or not 0 <= age < .15):
                raise ValueError('replacement needs fresh matched-epoch consumer telemetry')
            # Conservatively account for motion since the producer timestamp.
            consumer_frame = state.frame + (int(np.ceil(age*50)) if state.playing else 0)
            if state.qpos is None or start <= consumer_frame + guard_frames:
                raise ValueError('replacement boundary already consumed or inside guard')
            lo = start - state.frame_origin
            if lo < 1 or start >= state.frame_origin + len(state.qpos):
                raise ValueError('replacement boundary outside retained reference')
            # Anchor the 25->50 interpolation to the immutable previous sample.
            previous = state.qpos[lo - 2:lo - 1]
            incoming = resample_mujoco_qpos(np.concatenate((previous, source)), 25, 50,
                                           time_stretch=1.0)[2:]
            old = state.qpos[:lo].copy()
            root_jump = np.linalg.norm(incoming[0, :3] - old[-1, :3])
            joint_jump = np.max(np.abs(_wrapped_delta(incoming[0, 7:], old[-1, 7:])))
            yaw_jump = abs(np.arctan2(np.sin(_yaw_wxyz(incoming[0,3:7])-_yaw_wxyz(old[-1,3:7])),
                                     np.cos(_yaw_wxyz(incoming[0,3:7])-_yaw_wxyz(old[-1,3:7]))))
            if root_jump > .25 or joint_jump > .75 or yaw_jump > np.deg2rad(45):
                raise ValueError(f'unsafe replacement seam: root={root_jump:.3f}, joint={joint_jump:.3f}, yaw={yaw_jump:.3f}')
            continuity = live_seam_limits(old, incoming)
            if commit is not None:
                commit()
            prefix_velocities = state.joint_vel[:lo - 1].copy()
            state.qpos = np.concatenate((old, incoming))
            state.joint_pos, state.joint_vel = compute_sonic_joint_data(state.qpos, 50)
            state.joint_vel[:lo - 1] = prefix_velocities
            state.source_next = int(source_start) + len(source)
            state.source_tail = state.qpos[-1:].copy()
            state.canonical_qpos = None
            state.name = name
            state.revision += 1
            state.chunk_id += 1
            state.consumer_ack_epoch = state.epoch
            # The first new position also changes the preceding unconsumed
            # frame's forward-difference velocity. Send that frame as well.
            publish_start = start - 1
            state.consumer_received_until = publish_start
            state.replacement_start = (publish_start if state.replacement_start is None
                                       else min(publish_start, state.replacement_start))
            state.updated_at = time.monotonic()
            packet = pack_web_motion(state.qpos, fps=50, name=name, current_frame=state.frame,
                playing=state.playing, debug_mode=self.debug_mode, epoch_id=state.epoch,
                sequence_id=state.revision, frame_origin=state.frame_origin)
        if self.web:
            self.web.send(packet)
        return dict(source_start=source_start, target_start=start, root_jump=float(root_jump),
                    joint_jump=float(joint_jump), yaw_jump=float(yaw_jump),
                    consumer_frame=consumer_frame, consumer_age_seconds=age if state.playing else None,
                    remaining_lead_seconds=(start-consumer_frame)/50, **continuity)

    def _stabilize_extension(
        self, previous: np.ndarray, incoming: np.ndarray
    ) -> np.ndarray:
        """Blend only a dangerously discontinuous new tail; preserve old frames."""
        boundary = len(previous)
        if boundary >= len(incoming):
            return incoming
        output = incoming.copy()
        output[:boundary] = previous
        joint_jump = float(
            np.max(
                np.abs(
                    _wrapped_delta(incoming[boundary, 7:], previous[-1, 7:])
                )
            )
        )
        root_jump = float(
            np.linalg.norm(incoming[boundary, :3] - previous[-1, :3])
        )
        yaw_jump = abs(
            float(
                np.arctan2(
                    np.sin(
                        _yaw_wxyz(incoming[boundary, 3:7])
                        - _yaw_wxyz(previous[-1, 3:7])
                    ),
                    np.cos(
                        _yaw_wxyz(incoming[boundary, 3:7])
                        - _yaw_wxyz(previous[-1, 3:7])
                    ),
                )
            )
        )
        joint_limit = float(
            os.environ.get("ARDY_SONIC_BOUNDARY_JOINT_LIMIT_RAD", "0.75")
        )
        root_limit = float(
            os.environ.get("ARDY_SONIC_BOUNDARY_ROOT_LIMIT_M", "0.25")
        )
        yaw_limit = np.deg2rad(
            float(os.environ.get("ARDY_SONIC_BOUNDARY_YAW_LIMIT_DEG", "45"))
        )
        if joint_jump <= joint_limit and root_jump <= root_limit and yaw_jump <= yaw_limit:
            return output

        blend_count = min(
            int(os.environ.get("ARDY_SONIC_BOUNDARY_BLEND_FRAMES", "10")),
            len(output) - boundary,
        )
        if blend_count <= 0:
            return output
        previous_joint_step = (
            _wrapped_delta(previous[-1, 7:], previous[-2, 7:])
            if len(previous) > 1
            else np.zeros(29)
        )
        expected_joints = previous[-1, 7:] + previous_joint_step
        joint_offset = _wrapped_delta(expected_joints, output[boundary, 7:])
        previous_root_step = (
            previous[-1, :3] - previous[-2, :3]
            if len(previous) > 1
            else np.zeros(3)
        )
        root_offset = previous[-1, :3] + previous_root_step - output[boundary, :3]
        correction = (
            Rotation.from_quat(previous[-1, 3:7], scalar_first=True)
            * Rotation.from_quat(output[boundary, 3:7], scalar_first=True).inv()
        )
        correction_slerp = Slerp(
            [0.0, 1.0],
            Rotation.concatenate([correction, Rotation.identity()]),
        )
        for local_index in range(blend_count):
            alpha = (local_index + 1) / blend_count
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            frame_index = boundary + local_index
            output[frame_index, :3] += root_offset * (1.0 - smooth)
            output[frame_index, 7:] += joint_offset * (1.0 - smooth)
            raw_rotation = Rotation.from_quat(
                output[frame_index, 3:7], scalar_first=True
            )
            output[frame_index, 3:7] = (
                correction_slerp([smooth])[0] * raw_rotation
            ).as_quat(scalar_first=True)
        output[:, 3:7] = normalize_quaternions_wxyz(output[:, 3:7])
        print(
            "[ARDY->SONIC chunk boundary] applied safety blend: "
            f"frames={blend_count} joint_jump={joint_jump:.4f}rad "
            f"root_jump={root_jump:.4f}m yaw_jump={yaw_jump:.4f}rad"
        )
        return output

    def play(self, *, source_frame: int, source_fps: float) -> None:
        self.set_playing(True, source_frame=source_frame, source_fps=source_fps)

    def viewer_playback_control(self, action: str) -> None:
        """Control an already exported trajectory without discarding it."""
        if action not in {"play", "pause", "restart"}:
            raise ValueError(f"Unsupported playback control: {action}")
        with self._lock:
            native_rewind = self._state.source_next > 0 and action == 'restart'
        if native_rewind and self.source_control is not None:
            if self.source_control('restart'):
                return
        with self._lock:
            state = self._state
            if state.qpos is None:
                return
            # The current streaming tail is temporary, not a replay boundary.
            rewind = action == "restart" or (
                action == "play" and state.frame >= state.frame_origin+len(state.qpos)-1
                and (self.debug_mode != "ardy_streaming" or state.stream_complete)
            )
            if rewind:
                if state.frame_origin > 0:
                    # Never turn a global cursor into a negative array slice.
                    # Keep the valid pose paused if its archive owner is gone.
                    state.playing = False
                    print('[ARDY->SONIC control] replay requires the source archive owner', flush=True)
                    return
                state.frame = 0
                state.epoch += 1
                state.controller_restart_pending = True
            state.playing = action == "play"
            frame, epoch, stretch = state.frame, state.epoch, state.time_stretch
        print(f"[ARDY->SONIC control] action={action} frame={frame} epoch={epoch} rewind={rewind}", flush=True)
        # Play is sent by the stream loop after the existing readiness gate.
        # Restart/pause are announced immediately so the browser and simulator
        # use the same new epoch; producer controls are not relayed back here.
        if self.web and (rewind or action == "pause"):
            self.web.send(pack_web_control(
                "restart" if rewind else "pause", frame=frame, fps=self.target_fps,
                debug_mode=self.debug_mode, time_stretch=stretch, startup_epoch=epoch,
            ))

    def pause(self, *, source_frame: int, source_fps: float) -> None:
        self.set_playing(False, source_frame=source_frame, source_fps=source_fps)

    def set_playing(
        self, playing: bool, *, source_frame: int, source_fps: float
    ) -> None:
        frame = self._source_to_target_frame(source_frame, source_fps)
        with self._lock:
            self._state.frame = frame
            self._state.playing = bool(playing)
            epoch = self._state.epoch
        if not playing:
            self._send_manager_command(start=False, play=False)
        # Play is announced by the stream thread only when SONIC is ready to
        # start. This keeps MuJoCo frozen while a restarted controller runs its
        # three-second INIT ramp. Pause remains immediate.
        if self.web and not playing:
            self.web.send(
                pack_web_control(
                    "pause",
                    frame=frame,
                    fps=self.target_fps,
                    debug_mode=self.debug_mode,
                    time_stretch=self._state.time_stretch,
                    startup_epoch=epoch,
                )
            )

    def seek(self, *, source_frame: int, source_fps: float) -> None:
        frame = self._source_to_target_frame(source_frame, source_fps)
        with self._lock:
            changed = frame != self._state.frame
            self._state.frame = frame
            if changed:
                self._state.epoch += 1
                self._state.revision += 1
            epoch = self._state.epoch
        if self.web:
            self.web.send(
                pack_web_control(
                    "seek",
                    frame=frame,
                    fps=self.target_fps,
                    debug_mode=self.debug_mode,
                    time_stretch=self._state.time_stretch,
                    startup_epoch=epoch,
                )
            )

    def seek_source(self, source_frame: int, source_fps: float) -> None:
        self.seek(source_frame=source_frame, source_fps=source_fps)

    def sync_source(
        self, source_frame: int, source_fps: float, tolerance_frames: int = 5
    ) -> None:
        target = self._source_to_target_frame(source_frame, source_fps)
        with self._lock:
            if abs(self._state.frame - target) <= tolerance_frames:
                return
            previous = self._state.frame
            self._state.frame = target
            playing = self._state.playing
            epoch = self._state.epoch
        print(
            f"[ARDY->SONIC] clock correction: frame {previous} -> {target}"
        )
        if self.web:
            self.web.send(
                pack_web_control(
                    "sync",
                    frame=target,
                    fps=self.target_fps,
                    playing=playing,
                    debug_mode=self.debug_mode,
                    startup_epoch=epoch,
                )
            )

    def restart(self) -> None:
        with self._lock:
            self._state.frame_origin = 0
            self._state.source_next = 0
            self._state.source_tail = None
            self._state.replacement_start = None
            self._state.consumer_ack_epoch = -1
            self._state.consumer_received_until = 0
            # GenerationMixin calls restart before producing the replacement.
            # Drop the previous trajectory so the stream thread cannot publish
            # a stale epoch while inference is still running.
            self._state.qpos = None
            self._state.canonical_qpos = None
            self._state.joint_pos = None
            self._state.joint_vel = None
            self._state.frame = 0
            self._state.playing = False
            self._state.epoch += 1
            self._state.revision += 1
            self._state.controller_restart_pending = True
            self._state.time_stretch = 1.0
            self._state.stream_complete = False
            epoch = self._state.epoch
        if self.web:
            self.web.send(
                pack_web_control(
                    "restart",
                    frame=0,
                    fps=self.target_fps,
                    debug_mode=self.debug_mode,
                    time_stretch=1.0,
                    startup_epoch=epoch,
                )
            )

    def reset(self) -> None:
        self.restart()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self.web:
            self.web.close()

    def _send_manager_command(
        self,
        *,
        start: bool,
        stop: bool = False,
        play: bool = False,
        restart: bool = False,
    ) -> None:
        if self._zmq_bind is None:
            return
        try:
            self._zmq_commands.put_nowait((start, stop, play, restart))
        except queue.Full:
            try:
                self._zmq_commands.get_nowait()
            except queue.Empty:
                pass
            self._zmq_commands.put_nowait((start, stop, play, restart))

    def _source_to_target_frame(self, source_frame: int, source_fps: float) -> int:
        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        with self._lock:
            count = 0 if self._state.qpos is None else self._state.frame_origin + len(self._state.qpos)
            time_stretch = self._state.time_stretch
        frame = int(
            round(source_frame * self.target_fps / source_fps * time_stretch)
        )
        return max(0, min(frame, max(0, count - 1)))

    def source_playback_fps(self, source_fps: float) -> float:
        """Return the ARDY viewer rate matching the retimed SONIC reference."""
        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        with self._lock:
            time_stretch = self._state.time_stretch
        return float(source_fps) / time_stretch

    def _stream_loop(self) -> None:
        # A transport or packet failure must not permanently remove the only
        # reference publisher. Reconnect paused and re-enter the readiness gate;
        # never bypass alignment or automatically enable physics after failure.
        while not self._stop.is_set():
            self._run_stream_loop()
            if not self._stop.is_set():
                self._stop.wait(1.0)

    def _run_stream_loop(self) -> None:
        control_socket: zmq.Socket | None = None
        if self._zmq_bind is not None:
            self.zmq = ZmqPublisher(self._zmq_bind, self._zmq_topic)
            control_socket = self.zmq.context.socket(zmq.PULL)
            control_socket.setsockopt(zmq.RCVHWM, 8)
            control_socket.setsockopt(zmq.LINGER, 0)
            control_socket.bind(
                os.environ.get(
                    "ARDY_SONIC_CONTROL_BIND", "tcp://127.0.0.1:5559"
                )
            )
        period = 1.0 / self.target_fps
        next_tick = time.monotonic()
        last_epoch = -1
        last_revision = -1
        published_until = 0
        last_pose_send = 0.0
        start_not_before = 0.0
        epoch_ready = False
        motor_started = False
        manager_playing = False
        previous_playing = False
        simulation_ready_epoch = -1
        warmup_seconds = float(
            os.environ.get("ARDY_SONIC_REFERENCE_WARMUP_SECONDS", "0.1")
        )
        mode_switch_seconds = float(
            os.environ.get("ARDY_SONIC_MODE_SWITCH_SECONDS", "0.2")
        )
        pending_epoch = -1
        publish_not_before = 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                self._stop.wait(max(0.0, next_tick - now))
                if self._stop.is_set():
                    break
                now = time.monotonic()
                next_tick = max(next_tick + period, now)

                if control_socket is not None:
                    while control_socket.poll(0):
                        command = control_socket.recv_json()
                        if command.get("action") == "policy":
                            enabled = bool(command.get("enabled", True))
                            # SONIC's stop flag is a process-level emergency stop.
                            # Policy OFF pauses physics through the simulator control
                            # channel, so keep the controller alive for later resume.
                            with self._lock:
                                current_epoch = self._state.epoch
                                play = (
                                    enabled
                                    and self._state.playing
                                    and simulation_ready_epoch == current_epoch
                                )
                            self._queue_manager_command(
                                start=enabled,
                                stop=False,
                                play=play,
                                restart=False,
                            )
                            manager_playing = play
                        elif command.get("action") in {"play", "pause", "restart"}:
                            action = command['action']
                            if self.source_control is None or not self.source_control(action):
                                self.viewer_playback_control(action)
                        elif command.get("action") == "simulation_ready":
                            try:
                                ready_epoch = int(command.get("startup_epoch", -1))
                            except (TypeError, ValueError):
                                ready_epoch = -1
                            with self._lock:
                                current_epoch = self._state.epoch
                            if ready_epoch == current_epoch:
                                simulation_ready_epoch = ready_epoch
                                print(
                                    "[ARDY->SONIC safety] MuJoCo reported stable "
                                    f"for epoch {ready_epoch}; playback unlocked."
                                )

                while True:
                    try:
                        start, stop, play, restart = (
                            self._zmq_commands.get_nowait()
                        )
                    except queue.Empty:
                        break
                    self._queue_manager_command(
                        start=start,
                        stop=stop,
                        play=play,
                        restart=restart,
                    )
                    manager_playing = play

                with self._lock:
                    state = self._state
                    if (
                        state.qpos is None
                        or state.joint_pos is None
                        or state.joint_vel is None
                    ):
                        continue
                    frame = max(state.frame_origin,
                                min(state.frame, state.frame_origin + len(state.qpos) - 1))
                    playing = state.playing
                    epoch = state.epoch
                    revision = state.revision
                    frame_count = state.frame_origin + len(state.qpos)
                    frame_origin = state.frame_origin
                    controller_ready_not_before = (
                        state.controller_ready_not_before
                    )
                    controller_restart_pending = (
                        state.controller_restart_pending
                    )

                # A user pause can cancel MuJoCo alignment after `prepare`.
                # Resuming that unfinished epoch must prepare again; otherwise
                # we wait forever for readiness from physics we just disabled.
                # An already ready episode resumes without resetting its pose.
                if (playing and not previous_playing and epoch_ready
                        and simulation_ready_epoch != epoch):
                    epoch_ready = False
                previous_playing = playing

                if epoch != last_epoch:
                    if self.zmq and pending_epoch != epoch:
                        epoch_ready = False
                        manager_playing = False
                        simulation_ready_epoch = -1
                        pending_epoch = epoch
                        publish_not_before = now + mode_switch_seconds
                        # Enter streamed mode and pause its cursor. Explicit
                        # restarts are sent here, after PUB/SUB has had time to
                        # connect; sending from restart() can lose the command
                        # before SONIC's subscriber is ready.
                        self._queue_manager_command(
                            start=False,
                            stop=False,
                            play=False,
                            restart=controller_restart_pending,
                        )
                        if controller_restart_pending:
                            controller_ready_not_before = (
                                now + self._controller_restart_seconds
                            )
                            with self._lock:
                                if self._state.epoch == epoch:
                                    self._state.controller_restart_pending = False
                                    self._state.controller_ready_not_before = (
                                        controller_ready_not_before
                                    )
                        continue
                    if self.zmq and now < publish_not_before:
                        continue

                    epoch_ready = self.zmq is None
                    start_not_before = max(
                        now + warmup_seconds,
                        controller_ready_not_before,
                    )
                    if self.zmq:
                        with self._lock:
                            # Restart, append or tail replacement can commit
                            # between the header snapshot and this copy. Never
                            # combine old frame indices with the new arrays.
                            if (state is not self._state or state.epoch != epoch
                                    or state.revision != revision
                                    or state.frame_origin != frame_origin
                                    or state.frame_origin + len(state.qpos) != frame_count):
                                continue
                            local = frame - state.frame_origin
                            end = frame_count - state.frame_origin
                            qpos_window = state.qpos[local:end].copy()
                            joint_pos_window = state.joint_pos[local:end].copy()
                            joint_vel_window = state.joint_vel[local:end].copy()
                        qpos_window, joint_pos_window, joint_vel_window = self._coherent_pose_window(
                            qpos_window, joint_pos_window, joint_vel_window,
                            context="epoch"
                        )
                        self.zmq.send_pose(
                            joint_pos=joint_pos_window,
                            joint_vel=joint_vel_window,
                            body_pos=qpos_window[:, :3],
                            body_quat_wxyz=qpos_window[:, 3:7],
                            frame_index=np.arange(
                                frame, frame_count, dtype=np.int64
                            ),
                            catch_up=self.catch_up,
                            stream_complete=self.debug_mode == "ardy_offline" or state.stream_complete,
                        )
                        # Preserve the paused state without resetting the motion
                        # that was just decoded.
                        self._queue_manager_command(
                            start=False,
                            stop=False,
                            play=False,
                            restart=False,
                        )
                    print(
                        f"[ARDY->SONIC] epoch {epoch}: published once, "
                        f"frames [{frame}, {frame_count})"
                    )
                    published_until = frame_count
                    last_pose_send = now
                    last_epoch = epoch
                    last_revision = revision
                    pending_epoch = -1
                    self.stream_error = None
                elif (revision != last_revision and simulation_ready_epoch == epoch
                      and (not state.source_next or state.consumer_ack_epoch == epoch)):
                    with self._lock:
                        if (state is not self._state or state.epoch != epoch
                                or state.revision != revision
                                or state.frame_origin != frame_origin
                                or state.frame_origin + len(state.qpos) != frame_count):
                            continue
                        replacement_start = state.replacement_start
                    if self.zmq and (frame_count > published_until or replacement_start is not None):
                        # Keep SONIC's streamed window anchored at the real
                        # frame zero during INIT/alignment. Rolling appends can
                        # otherwise replace local frame zero with a much later
                        # global window before playback has even been unlocked.
                        # A future edit can happen before the initial prefix
                        # has been acknowledged. Fill every unsent frame before
                        # the replacement; never introduce a publication gap.
                        append_start = published_until if replacement_start is None else min(published_until,replacement_start)
                        with self._lock:
                            if (state is not self._state or state.epoch != epoch
                                    or state.revision != revision
                                    or state.frame_origin != frame_origin
                                    or state.frame_origin + len(state.qpos) != frame_count):
                                continue
                            state.replacement_start = None
                            local = append_start - state.frame_origin
                            if local < 0:
                                raise RuntimeError('consumer publication fell behind retained reference')
                            end = frame_count - state.frame_origin
                            qpos_window = state.qpos[local:end].copy()
                            joint_pos_window = state.joint_pos[local:end].copy()
                            joint_vel_window = state.joint_vel[local:end].copy()
                        qpos_window, joint_pos_window, joint_vel_window = self._coherent_pose_window(
                            qpos_window, joint_pos_window, joint_vel_window,
                            context="append"
                        )
                        self.zmq.send_pose(
                            joint_pos=joint_pos_window,
                            joint_vel=joint_vel_window,
                            body_pos=qpos_window[:, :3],
                            body_quat_wxyz=qpos_window[:, 3:7],
                            frame_index=np.arange(
                                append_start, frame_count, dtype=np.int64
                            ),
                            catch_up=self.catch_up,
                            stream_complete=state.stream_complete,
                        )
                        print(
                            f"[ARDY->SONIC] epoch {epoch}: appended once, "
                            f"frames [{append_start}, {frame_count})"
                        )
                        published_until = frame_count
                        last_pose_send = now
                    last_revision = revision
                # PUB/conflation does not guarantee delivery of a one-shot
                # delta. Retry identical global frames until the matched
                # consumer reports their buffer coverage. No model re-commit,
                # epoch reset, cursor seek, or qpos modification is involved.
                if self.zmq and epoch == last_epoch and now-last_pose_send >= 0.5:
                    retry = self.unacknowledged_source_reference(epoch, published_until)
                    if retry is not None:
                        self.zmq.send_pose(**retry)
                        last_pose_send = now
                        indices = retry['frame_index']
                        print(f'[ARDY->SONIC] epoch {epoch}: retry unacknowledged frames '
                              f'[{indices[0]}, {indices[-1]+1})')
                if not epoch_ready and playing and now >= start_not_before:
                    self._queue_manager_command(
                        start=True,
                        stop=False,
                        play=False,
                        restart=False,
                    )
                    motor_started = True
                    manager_playing = False
                    epoch_ready = True
                    with self._lock:
                        if self._state.epoch == epoch:
                            self._state.controller_ready_not_before = 0.0
                    if self.web and self.debug_mode != "reference_only":
                        self.web.send(
                            pack_web_control(
                                "prepare",
                                frame=frame,
                                fps=self.target_fps,
                                debug_mode=self.debug_mode,
                                startup_epoch=epoch,
                            )
                        )
                elif epoch_ready and playing != manager_playing:
                    # No timeout fallback: failure to prove frame-zero
                    # stability must leave both cursors paused indefinitely.
                    if (
                        playing
                        and self.zmq is not None
                        and simulation_ready_epoch != epoch
                    ):
                        continue
                    self._queue_manager_command(
                        start=playing and not motor_started,
                        stop=False,
                        play=playing,
                        restart=False,
                    )
                    motor_started = motor_started or playing
                    manager_playing = playing
                    if self.web and playing:
                        self.web.send(
                            pack_web_control(
                                "play",
                                frame=frame,
                                fps=self.target_fps,
                                debug_mode=self.debug_mode,
                                startup_epoch=epoch,
                            )
                        )
                if not epoch_ready or not playing:
                    continue

                with self._lock:
                    if state.epoch != epoch or not state.playing:
                        continue
                    # ARDY publishes a rolling horizon. Reaching the current
                    # buffer head is not the end of the motion: hold the last
                    # available frame and keep SONIC in play mode so the next
                    # replan append resumes without a pause/catch-up reset.
                    state.frame = min(state.frame + 1, frame_count - 1)
        except Exception as exc:
            self.stream_error = f'{type(exc).__name__}: {exc}'
            with self._lock:
                self._state.playing = False
                epoch, frame = self._state.epoch, self._state.frame
            self._queue_manager_command(start=False, stop=False, play=False, restart=False)
            if self.web:
                self.web.send(pack_web_control('pause', frame=frame, fps=self.target_fps,
                    debug_mode=self.debug_mode, startup_epoch=epoch))
            print(f"[ARDY->SONIC ZMQ] paused after error; reconnecting: {self.stream_error}", flush=True)
        finally:
            if control_socket is not None:
                control_socket.close(linger=0)
            if self.zmq is not None:
                self.zmq.close()
                self.zmq = None

    def _queue_manager_command(
        self, *, start: bool, stop: bool, play: bool, restart: bool
    ) -> None:
        if self.zmq is None:
            return
        try:
            self.zmq.send_command(
                start=start,
                stop=stop,
                planner=False,
                play=play,
                restart=restart,
            )
        except zmq.Again:
            pass
