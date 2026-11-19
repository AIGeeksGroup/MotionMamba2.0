#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import tyro

from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.scripts.sonic_web_io import SonicWebIO
from gear_sonic.scripts.g1_contract_audit import audit_g1_model


@dataclass
class HeadlessWebConfig(SimLoopConfig):
    debug_mode: Literal[
        "reference_only", "sonic_official", "ardy_offline", "ardy_streaming"
    ] = "ardy_streaming"
    """Select an isolated reference/control diagnostic path."""

    release_file: str = "/tmp/sonic_release"
    """创建此文件后，自动释放机器人悬挂弹力带。"""

    telemetry_port: int = 5560
    """Publish actual MuJoCo qpos/qvel to the local web bridge."""

    control_port: int = 5558
    """Receive browser camera, physics, and restart controls."""


def configure_third_person_camera(simulator: BaseSimulator) -> mujoco.MjvCamera:
    """关闭默认第一人称相机，并创建跟随 pelvis 的第三人称相机。"""

    third_person_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(third_person_camera)

    # 使用跟踪相机，持续跟随 G1 pelvis。
    third_person_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    third_person_camera.trackbodyid = (
        simulator.sim_env.mj_model.body("pelvis").id
    )

    # 默认斜后方第三人称视角。
    third_person_camera.distance = 3.0
    third_person_camera.azimuth = 135.0
    third_person_camera.elevation = -20.0

    # 设置初始观察中心。
    third_person_camera.lookat[:] = (
        simulator.sim_env.mj_data.body("pelvis").xpos
    )

    # BaseSimulator 初始化时已经创建了默认 ego_view Renderer。
    # 必须先关闭旧 Renderer，再使用第三人称配置重新创建。
    for renderer in list(simulator.sim_env.renderers.values()):
        try:
            renderer.close()
        except Exception as exc:
            print(f"关闭旧 Renderer 时出现警告：{exc}")

    # 当前模型的离屏 framebuffer 宽度为 640，
    # 因此先使用 640×360，避免 width 1280 > framebuffer width 640。
    simulator.sim_env.camera_configs = {
        "third_person": {
            "height": 360,
            "width": 640,
            "params": third_person_camera,
        }
    }

    # init_renderers() 会按照新的 camera_configs 重建 Renderer。
    simulator.sim_env.init_renderers()
    return third_person_camera


def watch_release_file(
    simulator: BaseSimulator,
    release_path: Path,
    stop_event: threading.Event,
) -> None:
    """后台监听网页发送的机器人释放指令。

    此线程只修改 elastic_band.enable，不进行任何 MuJoCo 渲染。
    Renderer 的创建和 render() 都留在主线程。
    """

    released = False

    while not stop_event.is_set():
        if not released and release_path.exists():
            band = getattr(simulator.sim_env, "elastic_band", None)

            if band is not None:
                band.enable = False
                released = True
                print("已释放机器人弹力带。", flush=True)
            else:
                released = True
                print("当前场景没有 elastic_band。", flush=True)

            try:
                release_path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"删除释放指令文件失败：{exc}", flush=True)

        stop_event.wait(0.1)


def close_renderers(simulator: BaseSimulator) -> None:
    """在创建 EGL Context 的主线程中显式关闭所有 Renderer。"""

    renderers = getattr(simulator.sim_env, "renderers", {})

    for camera_name, renderer in list(renderers.items()):
        try:
            renderer.close()
        except Exception as exc:
            print(
                f"关闭相机 {camera_name} 的 Renderer 时出现警告：{exc}",
                flush=True,
            )

    renderers.clear()


def main(config: HeadlessWebConfig) -> None:
    # 浏览器直接从 Actual State 做 WebGL 正向运动学，不再创建 EGL/MJPEG 链路。
    config.enable_onscreen = False
    config.enable_offscreen = False
    config.enable_image_publish = False

    release_path = Path(config.release_file)
    release_path.parent.mkdir(parents=True, exist_ok=True)
    release_path.unlink(missing_ok=True)

    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name
    # Preserve the failed state for telemetry and diagnostics. Automatic
    # mj_resetData loops hide the first bad command and look like visual
    # twitching in the browser.
    wbc_config["AUTO_RESET_ON_FALL"] = False

    simulator: BaseSimulator | None = None
    release_thread: threading.Thread | None = None
    stop_event = threading.Event()
    web_io: SonicWebIO | None = None

    try:
        simulator = BaseSimulator(
            config=wbc_config,
            env_name=config.env_name,
            onscreen=False,
            offscreen=False,
            enable_image_publish=False,
        )
        robot_scene = Path(__file__).resolve().parents[2] / wbc_config["ROBOT_SCENE"]
        audit_g1_model(simulator.sim_env, robot_scene)

        web_io = SonicWebIO(
            simulator,
            None,
            telemetry_port=config.telemetry_port,
            control_port=config.control_port,
            debug_mode=config.debug_mode,
        )
        original_sim_step = simulator.sim_env.sim_step
        simulator.sim_env.sim_step = lambda: web_io.controlled_sim_step(
            original_sim_step
        )

        # 只让辅助线程监听释放文件。
        # 不在辅助线程中调用 renderer.render()。
        release_thread = threading.Thread(
            target=watch_release_file,
            args=(simulator, release_path, stop_event),
            name="sonic-release-file-watcher",
            daemon=True,
        )
        release_thread.start()

        print("=" * 60, flush=True)
        print("SONIC MuJoCo 已在无窗口模式启动", flush=True)
        print(f"Debug mode: {config.debug_mode}", flush=True)
        print(f"Actual State telemetry：{config.telemetry_port}", flush=True)
        print(f"等待释放指令文件：{release_path}", flush=True)
        print("浏览器使用本地 WebGL renderer；MuJoCo 不创建图像流", flush=True)
        print("=" * 60, flush=True)

        # 关键修复：
        #
        # 错误写法：
        # simulator.start_as_thread()
        #
        # 正确写法：
        # 在创建 EGL Renderer 的当前主线程中运行仿真和 render()。
        simulator.start()

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止无窗口 MuJoCo……", flush=True)

    finally:
        stop_event.set()

        if simulator is not None:
            try:
                simulator.close()
            except Exception as exc:
                print(f"停止 Simulator 时出现警告：{exc}", flush=True)

        if web_io is not None:
            web_io.close()

        if release_thread is not None and release_thread.is_alive():
            release_thread.join(timeout=2)

        release_path.unlink(missing_ok=True)
        print("SONIC MuJoCo 已停止。", flush=True)


if __name__ == "__main__":
    main(tyro.cli(HeadlessWebConfig))
