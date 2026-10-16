#!/usr/bin/env python3

import argparse
from pathlib import Path
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>ARDY + SONIC Third-Person Simulation</title>
    <style>
        body { margin: 0; background: #111; color: white; font-family: Arial, sans-serif; text-align: center; }
        header { padding: 12px; background: #1d1d1d; }
        img { width: min(100vw, 1280px); height: auto; background: black; }
        button { padding: 10px 24px; margin: 10px; font-size: 16px; cursor: pointer; }
        #status { margin: 8px; }
    </style>
</head>
<body>
<header>
    <strong>SONIC · MuJoCo 第三人称无窗口仿真</strong>
    <button onclick="releaseRobot()">释放机器人</button>
    <span id="status">等待 third_person 图像……</span>
</header>
<img src="/video.mjpg">
<script>
async function releaseRobot() {
    const response = await fetch("/release", {method: "POST"});
    const result = await response.json();
    document.getElementById("status").textContent = result.message;
}
</script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--camera-name", default="third_person")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=7870)
    parser.add_argument("--release-file", default="/tmp/sonic_release")
    return parser.parse_args()


args = parse_args()
app = Flask(__name__)
jpeg_lock = threading.Lock()
latest_jpeg = None
latest_camera_name = "unknown"


def camera_worker():
    global latest_jpeg, latest_camera_name

    client = ComposedCameraClientSensor(
        server_ip=args.camera_host,
        port=args.camera_port,
    )
    print(f"正在连接相机流：{args.camera_host}:{args.camera_port}")
    print(f"目标相机名称：{args.camera_name}")

    try:
        while True:
            sample = client.read(blocking=False)
            if not sample or not sample.get("images"):
                time.sleep(0.02)
                continue

            images = sample["images"]
            if args.camera_name and args.camera_name in images:
                name = args.camera_name
            else:
                valid_names = [
                    key for key, value in images.items() if value is not None
                ]
                if not valid_names:
                    time.sleep(0.02)
                    continue
                name = sorted(valid_names)[0]

            image = images.get(name)
            if image is None:
                time.sleep(0.02)
                continue

            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            if ok:
                with jpeg_lock:
                    latest_jpeg = encoded.tobytes()
                    latest_camera_name = name
            time.sleep(0.005)
    finally:
        client.close()


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/release", methods=["POST"])
def release():
    Path(args.release_file).touch()
    return jsonify({"ok": True, "message": "已发送释放指令"})


@app.route("/video.mjpg")
def video():
    def generate():
        while True:
            with jpeg_lock:
                frame = latest_jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
            time.sleep(1.0 / 30.0)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with jpeg_lock:
        ready = latest_jpeg is not None
        name = latest_camera_name
    return jsonify({"ready": ready, "camera": name})


if __name__ == "__main__":
    thread = threading.Thread(target=camera_worker, daemon=True)
    thread.start()
    app.run(
        host=args.web_host,
        port=args.web_port,
        threaded=True,
        use_reloader=False,
    )