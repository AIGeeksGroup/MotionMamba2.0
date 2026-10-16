"""Local SONIC web service: ARDY relay, telemetry fan-out, and camera proxy."""

from __future__ import annotations

import asyncio
from collections import deque, OrderedDict
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import zmq
import zmq.asyncio

from protocol import (
    REFERENCE_TOPIC,
    decode_safe_initial_qpos,
    decode_reference_telemetry_packet,
    validate_actual_packet,
    validate_motion_packet,
)


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
NODE_MODULES = ROOT / "node_modules"
ROBOT_DESCRIPTION_CANDIDATES = (
    ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description",
    ROOT.parents[1] / "GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description",
)
ROBOT_DESCRIPTION = next(
    (candidate for candidate in ROBOT_DESCRIPTION_CANDIDATES if candidate.is_dir()),
    ROBOT_DESCRIPTION_CANDIDATES[0],
)
TELEMETRY_ENDPOINT = os.environ.get("SONIC_TELEMETRY_ENDPOINT", "tcp://127.0.0.1:5560")
REFERENCE_TELEMETRY_ENDPOINT = os.environ.get(
    "SONIC_REFERENCE_TELEMETRY_ENDPOINT", "tcp://127.0.0.1:5557"
)
CONTROL_ENDPOINT = os.environ.get("SONIC_CONTROL_ENDPOINT", "tcp://127.0.0.1:5558")
BRIDGE_CONTROL_ENDPOINT = os.environ.get(
    "ARDY_BRIDGE_CONTROL_ENDPOINT", "tcp://127.0.0.1:5559"
)


@dataclass
class RelayState:
    consumers: set[WebSocket] = field(default_factory=set)
    consumer_queues: dict[WebSocket, ConsumerOutbox] = field(default_factory=dict)
    consumer_tasks: dict[WebSocket, asyncio.Task] = field(default_factory=dict)
    latest_motion: bytes | None = None
    latest_motion_header: dict | None = None
    latest_initial_qpos: object | None = None
    latest_control: str | None = None
    latest_actual: bytes | None = None
    latest_reference_telemetry: str | None = None
    motion_at: float = 0.0
    actual_at: float = 0.0
    reference_telemetry_at: float = 0.0
    producer_count: int = 0
    invalid_packets: int = 0
    websocket_dropped_frames: int = 0
    startup_epoch_pending: int | None = None
    simulation_ready_epoch: int | None = None
    reference_reset_pending: bool = False
    telemetry_task: asyncio.Task | None = None
    reference_telemetry_task: asyncio.Task | None = None
    zmq_context: zmq.asyncio.Context | None = None
    control_socket: zmq.asyncio.Socket | None = None
    bridge_control_socket: zmq.asyncio.Socket | None = None


class ConsumerOutbox:
    """Ordered events plus independent latest actual/reference state slots.

    Telemetry may coalesce with its own stream, never with motion or controls.
    An unresponsive consumer reconnects for fresh snapshots instead of losing
    an epoch transition or building an unbounded backlog.
    """
    def __init__(self, max_events=64):
        self.events = deque()
        self.latest = OrderedDict()
        self.ready = asyncio.Event()
        self.max_events = max_events
        self.overflow = False

    def put(self, payload, stream=None):
        replaced = False
        if stream is None:
            if len(self.events) >= self.max_events:
                self.overflow = True
            else:
                self.events.append(payload)
        else:
            replaced = stream in self.latest
            self.latest[stream] = payload
        self.ready.set()
        return replaced

    async def get(self):
        while True:
            if self.overflow:
                raise RuntimeError('consumer event backlog requires resync')
            if self.events:
                return self.events.popleft()
            if self.latest:
                return self.latest.popitem(last=False)[1]
            self.ready.clear()
            await self.ready.wait()


state = RelayState()
app = FastAPI(title="ARDY SONIC Web", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
app.mount("/vendor", StaticFiles(directory=NODE_MODULES), name="vendor")
app.mount("/robot", StaticFiles(directory=ROBOT_DESCRIPTION), name="robot")


@app.middleware("http")
async def cache_versioned_assets(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/vendor/", "/robot/")):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


async def broadcast(payload: bytes | str) -> None:
    stream = None
    if isinstance(payload, bytes) and payload.startswith(b'AST1'):
        stream = 'actual'
    elif isinstance(payload, str) and json.loads(payload).get('type') == 'reference_telemetry':
        stream = 'reference'
    for websocket in tuple(state.consumers):
        queue = state.consumer_queues.get(websocket)
        if queue is None:
            continue
        if queue.put(payload, stream):
            state.websocket_dropped_frames += 1


def clear_reference_telemetry() -> None:
    state.latest_reference_telemetry = None
    state.reference_telemetry_at = 0.0
    state.reference_reset_pending = True


def cache_reference_telemetry(telemetry: dict) -> str | None:
    if state.reference_reset_pending:
        if (
            telemetry.get("current_frame_global") != 0
            or telemetry.get("current_frame_local") != 0
            or telemetry.get("stream_window_start") != 0
        ):
            return None
        state.reference_reset_pending = False
    # C++ reports epoch=0. Once frame-zero reset has been observed, tag this
    # stream with the bridge epoch so browsers do not reject all live reference
    # packets as older than the motion/actual stream.
    epoch = (state.latest_motion_header or {}).get('epoch_id')
    if isinstance(epoch, int) and not telemetry.get('epoch_id'):
        telemetry = dict(telemetry, epoch_id=epoch)
    payload = json.dumps(telemetry, separators=(",", ":"))
    state.latest_reference_telemetry = payload
    state.reference_telemetry_at = time.monotonic()
    return payload


async def signal_simulation_ready(actual_header: dict) -> bool:
    pending = state.startup_epoch_pending
    try:
        actual_epoch = int(actual_header.get("startup_epoch", -1))
    except (TypeError, ValueError):
        return False
    if (
        pending is None
        or actual_epoch != pending
        or actual_header.get("startup_phase") != "running"
        or bool(actual_header.get("fallen", False))
        or state.simulation_ready_epoch == pending
        or state.bridge_control_socket is None
    ):
        return False
    try:
        await state.bridge_control_socket.send_json(
            {"action": "simulation_ready", "startup_epoch": pending},
            flags=zmq.NOBLOCK,
        )
    except zmq.Again:
        return False
    state.simulation_ready_epoch = pending
    return True


async def _consumer_sender(websocket: WebSocket, queue: ConsumerOutbox) -> None:
    try:
        while True:
            payload = await queue.get()
            sender = (
                websocket.send_bytes(payload)
                if isinstance(payload, bytes)
                else websocket.send_text(payload)
            )
            await asyncio.wait_for(sender, timeout=1.0)
    except asyncio.CancelledError:
        raise
    except Exception:
        state.consumers.discard(websocket)
        # Closing triggers the browser's reconnect/snapshot recovery. Merely
        # removing it from consumers left an apparently connected frozen page.
        try:
            await websocket.close(code=1013, reason='slow consumer; reconnect for current state')
        except Exception:
            pass


async def telemetry_loop() -> None:
    assert state.zmq_context is not None
    socket = state.zmq_context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(TELEMETRY_ENDPOINT)
    try:
        while True:
            packet = await socket.recv()
            try:
                actual_header = validate_actual_packet(packet)
            except ValueError:
                state.invalid_packets += 1
                continue
            state.latest_actual = packet
            state.actual_at = time.monotonic()
            await signal_simulation_ready(actual_header)
            await broadcast(packet)
    finally:
        socket.close(linger=0)


async def reference_telemetry_loop() -> None:
    assert state.zmq_context is not None
    socket = state.zmq_context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, REFERENCE_TOPIC)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(REFERENCE_TELEMETRY_ENDPOINT)
    try:
        while True:
            packet = await socket.recv()
            try:
                telemetry = decode_reference_telemetry_packet(packet)
            except ValueError:
                state.invalid_packets += 1
                continue
            payload = cache_reference_telemetry(telemetry)
            if payload is None:
                continue
            await broadcast(payload)
    finally:
        socket.close(linger=0)


@app.on_event("startup")
async def startup() -> None:
    state.zmq_context = zmq.asyncio.Context()
    state.control_socket = state.zmq_context.socket(zmq.PUSH)
    state.control_socket.setsockopt(zmq.SNDHWM, 8)
    state.control_socket.setsockopt(zmq.LINGER, 0)
    state.control_socket.connect(CONTROL_ENDPOINT)
    state.bridge_control_socket = state.zmq_context.socket(zmq.PUSH)
    state.bridge_control_socket.setsockopt(zmq.SNDHWM, 8)
    state.bridge_control_socket.setsockopt(zmq.LINGER, 0)
    state.bridge_control_socket.connect(BRIDGE_CONTROL_ENDPOINT)
    state.telemetry_task = asyncio.create_task(telemetry_loop())
    state.reference_telemetry_task = asyncio.create_task(reference_telemetry_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    if state.telemetry_task is not None:
        state.telemetry_task.cancel()
        try:
            await state.telemetry_task
        except asyncio.CancelledError:
            pass
    if state.reference_telemetry_task is not None:
        state.reference_telemetry_task.cancel()
        try:
            await state.reference_telemetry_task
        except asyncio.CancelledError:
            pass
    if state.control_socket is not None:
        state.control_socket.close(linger=0)
    if state.bridge_control_socket is not None:
        state.bridge_control_socket.close(linger=0)
    if state.zmq_context is not None:
        state.zmq_context.term()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    now = time.monotonic()
    return JSONResponse(
        {
            "ok": True,
            "producer_connected": state.producer_count > 0,
            "consumers": len(state.consumers),
            "motion_cached": state.latest_motion is not None,
            "motion_age_s": None if not state.motion_at else round(now - state.motion_at, 3),
            "telemetry_connected": bool(state.actual_at and now - state.actual_at < 1.0),
            "telemetry_age_s": None if not state.actual_at else round(now - state.actual_at, 3),
            "reference_telemetry_connected": bool(
                state.reference_telemetry_at and now - state.reference_telemetry_at < 1.0
            ),
            "reference_telemetry_age_s": (
                None
                if not state.reference_telemetry_at
                else round(now - state.reference_telemetry_at, 3)
            ),
            "invalid_packets": state.invalid_packets,
            "websocket_dropped_frames": state.websocket_dropped_frames,
            "motion": state.latest_motion_header,
            "startup_epoch_pending": state.startup_epoch_pending,
            "simulation_ready_epoch": state.simulation_ready_epoch,
        }
    )


async def send_control(message: dict, *, relay_bridge: bool = True) -> None:
    if message.get("type") == "viewer_control" and message.get("action") in {"play", "pause", "restart"}:
        # The producer owns playback epochs and the controller-ready gate.
        # Do not start physics directly or reset MuJoCo with a stale epoch.
        if state.bridge_control_socket is not None:
            try:
                await state.bridge_control_socket.send_json(message, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
        return
    if state.control_socket is None:
        return
    try:
        await state.control_socket.send_json(message, flags=zmq.NOBLOCK)
    except zmq.Again:
        pass
    if (
        relay_bridge
        and state.bridge_control_socket is not None
        and message.get("action") in {"policy", "restart"}
    ):
        try:
            await state.bridge_control_socket.send_json(message, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass


def _initial_qpos_payload() -> list[float] | None:
    """Return the validated selected reference pose in JSON-safe form."""
    if state.latest_initial_qpos is None:
        return None
    try:
        return [float(value) for value in state.latest_initial_qpos]
    except (TypeError, ValueError):
        return None


async def align_simulation_to_motion_frame(frame: object, epoch: object) -> bool:
    """Freeze MuJoCo and load a validated motion frame as the next start pose."""
    if state.latest_motion is None:
        return False
    try:
        frame_index = int(frame)
        if isinstance(frame, bool) or frame_index != frame:
            return False
        reference_qpos = decode_safe_initial_qpos(
            state.latest_motion, frame_index=frame_index
        )
        startup_epoch = int(epoch)
    except (TypeError, ValueError):
        return False

    state.latest_initial_qpos = reference_qpos
    state.startup_epoch_pending = startup_epoch
    state.simulation_ready_epoch = None
    await send_control(
        {
            "action": "physics",
            "enabled": False,
            "startup_epoch": startup_epoch,
        },
        relay_bridge=False,
    )
    await send_control(
        {
            "action": "reference_pose",
            "startup_epoch": startup_epoch,
            "reference_qpos": [float(value) for value in reference_qpos],
        },
        relay_bridge=False,
    )
    return True


@app.websocket("/ws/producer")
async def producer(websocket: WebSocket) -> None:
    await websocket.accept()
    state.producer_count += 1
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                packet = message["bytes"]
                previous_epoch = (
                    None
                    if state.latest_motion_header is None
                    else state.latest_motion_header.get("epoch_id")
                )
                try:
                    header = validate_motion_packet(packet)
                except ValueError:
                    state.invalid_packets += 1
                    continue
                try:
                    initial_qpos = decode_safe_initial_qpos(
                        packet, frame_index=int(header.get("current_frame", 0))
                    )
                except ValueError:
                    state.invalid_packets += 1
                    continue
                state.latest_motion = packet
                state.latest_motion_header = header
                state.latest_initial_qpos = initial_qpos
                state.motion_at = time.monotonic()
                motion_epoch = header.get("epoch_id")
                if (
                    isinstance(motion_epoch, int)
                    and not isinstance(motion_epoch, bool)
                    and motion_epoch != previous_epoch
                ):
                    # A new epoch should be visible at its validated selected
                    # pose even while playback is paused. Physics and policy still
                    # remain behind the existing prepare/simulation-ready gate.
                    state.startup_epoch_pending = motion_epoch
                    state.simulation_ready_epoch = None
                    await send_control(
                        {
                            "action": "reference_pose",
                            "startup_epoch": motion_epoch,
                            "reference_qpos": [float(value) for value in initial_qpos],
                        },
                        relay_bridge=False,
                    )
                    await send_control(
                        {
                            "action": "physics",
                            "enabled": False,
                            "startup_epoch": motion_epoch,
                        },
                        relay_bridge=False,
                    )
                await broadcast(packet)
            elif message.get("text") is not None:
                raw = message["text"]
                try:
                    control = json.loads(raw)
                except json.JSONDecodeError:
                    state.invalid_packets += 1
                    continue
                if control.get("type") != "control" or control.get("version") != 1:
                    state.invalid_packets += 1
                    continue
                state.latest_control = raw
                reference_only = control.get("debug_mode") == "reference_only"
                if "startup_epoch" in control or "epoch_id" in control:
                    try:
                        candidate_epoch = int(
                            control.get("startup_epoch", control.get("epoch_id"))
                        )
                    except (TypeError, ValueError):
                        candidate_epoch = None
                    if candidate_epoch is not None and control.get("action") in {
                        "prepare",
                        "play",
                        "restart",
                        "reset",
                    }:
                        state.startup_epoch_pending = candidate_epoch
                        state.simulation_ready_epoch = None
                if control.get("action") in {"restart", "reset"}:
                    try:
                        state.startup_epoch_pending = int(
                            control.get("startup_epoch", control.get("epoch_id", 0))
                        )
                    except (TypeError, ValueError):
                        state.startup_epoch_pending = None
                    state.simulation_ready_epoch = None
                    clear_reference_telemetry()
                    # The producer is the ARDY bridge and has already sent the
                    # matching SONIC manager command. Only reset MuJoCo here;
                    # routing back to port 5559 would duplicate the restart.
                    if not reference_only:
                        restart_message = {
                            "action": "restart",
                            "startup_epoch": control.get(
                                "startup_epoch", control.get("epoch_id", 0)
                            ),
                        }
                        reference_qpos = _initial_qpos_payload()
                        if reference_qpos is not None:
                            restart_message["reference_qpos"] = reference_qpos
                        await send_control(
                            restart_message,
                            relay_bridge=False,
                        )
                elif control.get("action") == "prepare":
                    if not reference_only:
                        # MuJoCo must reset onto the validated ARDY frame-zero
                        # pose before policy/physics are enabled for this epoch.
                        reference_qpos = _initial_qpos_payload()
                        if reference_qpos is not None:
                            await send_control(
                                {
                                    "action": "restart",
                                    "startup_epoch": control.get(
                                        "startup_epoch", control.get("epoch_id", 0)
                                    ),
                                    "reference_qpos": reference_qpos,
                                },
                                relay_bridge=False,
                            )
                        await send_control(
                            {
                                "action": "physics",
                                "enabled": True,
                                "startup_epoch": control.get(
                                    "startup_epoch", control.get("epoch_id", 0)
                                ),
                            }
                        )
                        await send_control(
                            {"action": "policy", "enabled": True},
                            relay_bridge=False,
                        )
                elif control.get("action") == "play":
                    if not reference_only:
                        await send_control({"action": "physics", "enabled": True})
                        await send_control(
                            {"action": "policy", "enabled": True},
                            relay_bridge=False,
                        )
                elif control.get("action") == "pause":
                    if not reference_only:
                        await send_control({"action": "physics", "enabled": False})
                elif control.get("action") == "seek":
                    if not reference_only:
                        await align_simulation_to_motion_frame(
                            control.get("frame"),
                            control.get(
                                "startup_epoch",
                                control.get(
                                    "epoch_id",
                                    (state.latest_motion_header or {}).get(
                                        "epoch_id", 0
                                    ),
                                ),
                            ),
                        )
                await broadcast(raw)
    except WebSocketDisconnect:
        pass
    finally:
        state.producer_count = max(0, state.producer_count - 1)


@app.websocket("/ws/consumer")
async def consumer(websocket: WebSocket) -> None:
    await websocket.accept()
    # Send snapshots directly before registering the live latest-value queue;
    # this preserves the established refresh ordering without allowing a live
    # telemetry burst to evict the initial motion/control packets.
    try:
        if state.latest_motion is not None:
            await websocket.send_bytes(state.latest_motion)
        if state.latest_control is not None:
            await websocket.send_text(state.latest_control)
        if state.latest_reference_telemetry is not None:
            await websocket.send_text(state.latest_reference_telemetry)
        if state.latest_actual is not None:
            await websocket.send_bytes(state.latest_actual)
        queue = ConsumerOutbox()
        state.consumers.add(websocket)
        state.consumer_queues[websocket] = queue
        sender_task = asyncio.create_task(_consumer_sender(websocket, queue))
        state.consumer_tasks[websocket] = sender_task
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if message.get("type") == "viewer_control":
                await send_control(message)
    except WebSocketDisconnect:
        pass
    finally:
        state.consumers.discard(websocket)
        state.consumer_queues.pop(websocket, None)
        sender_task = state.consumer_tasks.pop(websocket, None)
        if sender_task is not None:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass
