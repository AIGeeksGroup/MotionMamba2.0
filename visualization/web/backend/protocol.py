"""Versioned binary packet validation shared by the web bridge and tests."""

from __future__ import annotations

import json
import struct
import time
from typing import Any

import msgpack
import numpy as np


MOTION_MAGIC = b"ASB1"
ACTUAL_MAGIC = b"AST1"
MAX_HEADER_BYTES = 16 * 1024
MAX_MOTION_BYTES = 64 * 1024 * 1024
REFERENCE_TOPIC = b"g1_debug"


def decode_packet(packet: bytes, expected_magic: bytes) -> tuple[dict[str, Any], memoryview]:
    if len(packet) < 8 or packet[:4] != expected_magic:
        raise ValueError("invalid binary packet magic or length")
    header_length = struct.unpack_from("<I", packet, 4)[0]
    if header_length <= 0 or header_length > MAX_HEADER_BYTES:
        raise ValueError("invalid binary header length")
    payload_offset = 8 + header_length
    if payload_offset > len(packet):
        raise ValueError("truncated binary header")
    try:
        header = json.loads(packet[8:payload_offset].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid binary JSON header") from exc
    if header.get("version") != 1 or header.get("dtype") != "f32":
        raise ValueError("unsupported binary protocol version or dtype")
    if header.get("endian") != "little":
        raise ValueError("only little-endian packets are supported")
    return header, memoryview(packet)[payload_offset:]


def validate_motion_packet(packet: bytes) -> dict[str, Any]:
    if len(packet) > MAX_MOTION_BYTES:
        raise ValueError("motion packet exceeds the configured cache limit")
    header, payload = decode_packet(packet, MOTION_MAGIC)
    shape = header.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[1] != 36
        or not isinstance(shape[0], int)
        or shape[0] < 2
    ):
        raise ValueError("motion shape must be [T, 36] with T >= 2")
    if len(payload) != shape[0] * shape[1] * 4:
        raise ValueError("motion payload length does not match shape")
    values = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("motion payload contains NaN or Inf")
    prompt_segments = header.get("prompt_segments", [])
    if not isinstance(prompt_segments, list):
        raise ValueError("motion prompt_segments must be a list")
    previous_start = -1
    for segment in prompt_segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("invalid motion prompt segment")
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < previous_start
            or start < 0
            or end < start
            or end >= shape[0]
        ):
            raise ValueError("invalid motion prompt segment range")
        previous_start = start
    return header


def decode_safe_initial_qpos(packet: bytes, *, frame_index: int = 0) -> np.ndarray:
    """Return a normalized, bounded reference pose for MuJoCo alignment."""
    header, payload = decode_packet(packet, MOTION_MAGIC)
    validate_motion_packet(packet)
    frame_count = int(header["shape"][0])
    origin = header.get('frame_origin', 0)
    if type(origin) is not int or origin < 0:
        raise ValueError('invalid frame_origin')
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or not origin <= frame_index < origin + frame_count
    ):
        raise ValueError("reference frame index is outside the motion range")
    qpos = np.frombuffer(
        payload,
        dtype="<f4",
        count=36,
        offset=(frame_index-origin) * 36 * 4,
    ).astype(np.float64)
    quaternion_norm = float(np.linalg.norm(qpos[3:7]))
    if quaternion_norm < 1e-8:
        raise ValueError("initial root quaternion is degenerate")
    qpos[3:7] /= quaternion_norm
    if not 0.45 <= float(qpos[2]) <= 1.10:
        raise ValueError("initial root height is outside the safe range")
    w, x, y, z = qpos[3:7]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    if max(abs(float(roll)), abs(float(pitch))) > np.deg2rad(30.0):
        raise ValueError("initial root tilt is outside the safe range")
    if float(np.max(np.abs(qpos[7:]))) > 3.2:
        raise ValueError("initial joint magnitude is outside the safe range")
    return qpos


def validate_actual_packet(packet: bytes) -> dict[str, Any]:
    header, payload = decode_packet(packet, ACTUAL_MAGIC)
    qpos_count = int(header.get("qpos_count", -1))
    qvel_count = int(header.get("qvel_count", -1))
    if qpos_count != 36 or qvel_count != 35:
        raise ValueError("actual state must contain qpos[36] and qvel[35]")
    if len(payload) != (qpos_count + qvel_count) * 4:
        raise ValueError("actual-state payload length does not match header")
    if not np.isfinite(np.frombuffer(payload, dtype="<f4")).all():
        raise ValueError("actual state contains NaN or Inf")
    for name in ("sim_time", "timestamp", "monotonic_timestamp"):
        if name in header and (
            not isinstance(header[name], (int, float))
            or not np.isfinite(float(header[name]))
        ):
            raise ValueError(f"invalid actual-state time field: {name}")
    if "epoch_id" in header and (
        isinstance(header["epoch_id"], bool)
        or not isinstance(header["epoch_id"], int)
        or header["epoch_id"] < 0
    ):
        raise ValueError("invalid actual-state epoch_id")
    return header


def decode_reference_telemetry_packet(
    packet: bytes, topic: bytes = REFERENCE_TOPIC
) -> dict[str, Any]:
    if not packet.startswith(topic) or len(packet) <= len(topic):
        raise ValueError("invalid SONIC reference telemetry topic")
    try:
        decoded = msgpack.unpackb(packet[len(topic) :], raw=False, strict_map_key=False)
    except (ValueError, msgpack.ExtraData, msgpack.FormatError, msgpack.StackError) as exc:
        raise ValueError("invalid SONIC reference telemetry payload") from exc
    if not isinstance(decoded, dict):
        raise ValueError("SONIC reference telemetry must be a map")

    integer_fields = (
        "stream_window_start",
        "current_frame_local",
        "current_frame_global",
        "buffer_length",
        "steady_timestamp_ns",
    )
    for name in integer_fields:
        value = decoded.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid SONIC reference telemetry field: {name}")

    fps = decoded.get("reference_fps")
    if not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid SONIC reference telemetry field: reference_fps")

    arrays = {
        "reference_joint_pos": (decoded.get("body_q_target"), 29),
        "reference_root_pos": (decoded.get("base_trans_target"), 3),
        "reference_root_quat": (decoded.get("base_quat_target"), 4),
    }
    normalized_arrays: dict[str, list[float]] = {}
    for name, (value, expected_size) in arrays.items():
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid SONIC reference telemetry field: {name}") from exc
        if array.shape != (expected_size,) or not np.isfinite(array).all():
            raise ValueError(f"invalid SONIC reference telemetry field: {name}")
        normalized_arrays[name] = array.tolist()

    return {
        "type": "reference_telemetry",
        "version": 1,
        "current_frame": decoded["current_frame_global"],
        "current_frame_local": decoded["current_frame_local"],
        "current_frame_global": decoded["current_frame_global"],
        "stream_window_start": decoded["stream_window_start"],
        "buffer_length": decoded["buffer_length"],
        "reference_fps": float(fps),
        "steady_timestamp_ns": decoded["steady_timestamp_ns"],
        "epoch_id": int(decoded.get("startup_epoch", 0)),
        "source_time": decoded["current_frame_global"] / float(fps),
        "sim_time": decoded["current_frame_global"] / float(fps),
        "frame_float": float(decoded["current_frame_global"]),
        "monotonic_timestamp": decoded["steady_timestamp_ns"] / 1e9,
        "sequence_id": decoded["current_frame_global"],
        **normalized_arrays,
    }


def encode_actual_packet(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    frame: int,
    timestamp: float,
    sim_time: float,
    physics: bool,
    epoch_id: int = 0,
    monotonic_timestamp: float | None = None,
) -> bytes:
    qpos = np.ascontiguousarray(qpos, dtype="<f4")
    qvel = np.ascontiguousarray(qvel, dtype="<f4")
    if qpos.shape != (36,) or qvel.shape != (35,):
        raise ValueError(f"expected qpos[36], qvel[35], got {qpos.shape}, {qvel.shape}")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("actual state contains NaN or Inf")
    header = {
        "type": "actual_state",
        "version": 1,
        "dtype": "f32",
        "endian": "little",
        "qpos_count": 36,
        "qvel_count": 35,
        "frame": int(frame),
        "timestamp": float(timestamp),
        "sim_time": float(sim_time),
        "source_time": float(sim_time),
        "frame_float": float(frame),
        "monotonic_timestamp": float(
            time.monotonic() if monotonic_timestamp is None else monotonic_timestamp
        ),
        "epoch_id": int(epoch_id),
        "sequence_id": int(frame),
        "physics": bool(physics),
        "qpos_layout": "root_xyz_root_quat_wxyz_29_mujoco_joints",
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return ACTUAL_MAGIC + struct.pack("<I", len(encoded)) + encoded + qpos.tobytes() + qvel.tobytes()
