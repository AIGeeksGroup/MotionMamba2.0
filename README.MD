# Motion Mamba 2.0 — G1 Viewer + SONIC simulation

This `simulation` branch packages the working G1 demo: the motion generator,
the modified ARDY/Viser timeline, keyboard and mouse planning, and the synchronized
SONIC/MuJoCo execution page. It includes the deployed acceleration code and the
modified Viser TypeScript **and compiled client**. You do not need the original
server's training directories, datasets or launcher files.

The released model is named **moiton mamba 2.0**.
It is the original Native G1 checkpoint at global step **117,000**: foundation
112k followed by 5k future/interaction training. It has not been pruned or converted.

## 1. Machine requirements

Use Linux x86-64 (reference deployment: Ubuntu), an NVIDIA driver supporting
CUDA 12.4, CUDA Toolkit 12.x for the C++ build, Python 3.11 for generation/viewing,
Python 3.10 for the matched SONIC/CycloneDDS environment, Node.js 20+ and npm.
The reference uses four RTX 4090 GPUs:

| Process | Default physical GPU | Typical reference use |
| --- | --- | --- |
| LLM2Vec text encoder | 0 | about 15 GB VRAM |
| Motion generator and warmed CUDA graphs | 1 | about 19 GB VRAM |
| Viewer backend | 2 | less than 1 GB VRAM |
| SONIC policy | 3 | about 1–2 GB VRAM |

The Viewer and SONIC pages render in your browser. Enable browser hardware
acceleration. A headless server does not need an X display. Allow at least 50 GB
free disk for model assets, environments, downloaded wheels and initial caches.
Different GPUs may have different speed and floating-point results; the generator
validates its own eager/optimized equivalence before admitting acceleration.

Install the build tools (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs tmux build-essential cmake clang pkg-config \
  libeigen3-dev libyaml-cpp-dev libboost-all-dev libzmq3-dev \
  libmsgpack-dev libasound2-dev libssl-dev zlib1g-dev libgtest-dev \
  libgl1 libglfw3 libglib2.0-0 python3.11 python3.11-venv \
  python3.10 python3.10-venv nodejs npm curl
```

If your distribution's Node.js is older than 20, install Node.js 20+ first.
`nvcc --version` and `nvidia-smi` must work. Install the `just` command used by
SONIC's deployment script (e.g. `cargo install --locked just` using Rust's Cargo).

## 2. Clone and install

Both the GitHub repository and the Motion Mamba HF repository are private.
Your accounts must have access; use your own credentials. No access token is
embedded in this branch or any launch command.

```bash
gh auth login
git clone --branch simulation --single-branch \
  https://github.com/AIGeeksGroup/MotionMamba2.0.git
cd MotionMamba2.0
bash launchers/install.sh
```

If the Python executables have different names, select them for installation:
`PYTHON=/path/to/python3.11 SIM_PYTHON=/path/to/python3.10 bash launchers/install.sh`.
Both interpreters must include `venv`/`ensurepip` support.

The installer creates `.venv` for generation/viewing and a separate
`GR00T-WholeBodyControl/.venv_sim` for MuJoCo. It installs the current deployment's
Python package versions. ARDY's editable source is installed with `--no-deps`;
the deployment requirements in `simulation/requirements.txt` take precedence
over ARDY's general-purpose dependency ranges. In addition,
the two `simulation/requirements*.lock.txt` files also pin the tested transitive
dependencies, preventing unrelated package upgrades from changing the deployment.

Torch is **2.6.0 + CUDA 12.4**. The Mamba-3 kernels require **Triton 3.5.1**, which
is deliberately installed after Torch with `--no-deps`. Do not run a general
dependency upgrade that silently replaces it with Torch's default Triton 3.2.
The Mamba-3 Python/Triton sources are included under `vendor/state-spaces-mamba`;
this inference path does not need a separately compiled Mamba-1 extension.

## 3. Download model and runtime assets

The text encoder uses gated Meta Llama weights. Accept the upstream
[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
terms for your HF account and obtain read access to the required NVIDIA assets.
Authenticate before downloading:

```bash
HF_ENDPOINT=https://huggingface.co .venv/bin/hf auth login
.venv/bin/python simulation/download_assets.py
```

This downloads the following pinned assets:

- `Royalman1/Motion-mamba-2.0` → `checkpoints/moiton mamba 2.0.pt`.
- NVIDIA ARDY G1's official tokenizer, model configuration and normalization.
- Meta Llama 3 8B plus the two McGill LLM2Vec adapters used by ARDY.
- NVIDIA GEAR-SONIC encoder/decoder policy and planner ONNX files.

The downloader records revisions in `runtime/ASSET_DOWNLOAD.json` and checks
the released checkpoint and SONIC asset hashes. The original checkpoint is
4,570,178,548 bytes and its SHA-256 is:

```text
2e828449addff3aa079387f319589cf4ac9b202c48577278a6d274f28b14f679
```

The authoritative hash is also in the HF `checkpoint_manifest.json` and
`simulation/download_assets.py`. To download only this weight manually:

```bash
HF_ENDPOINT=https://huggingface.co .venv/bin/hf download Royalman1/Motion-mamba-2.0 \
  "moiton mamba 2.0.pt" --local-dir checkpoints
```

If supplying another checkpoint, run `download_assets.py --skip-checkpoint`
to fetch the remaining assets. These assets are still required even when the
selected checkpoint is not the released 117k model.

## 4. Install C++ inference dependencies and build SONIC

Install the **TensorRT 10.13.3.9 CUDA 12 Linux x86-64 tar package** from
[NVIDIA TensorRT downloads](https://developer.nvidia.com/tensorrt/download).
Extract it so the following files exist:

```text
vendor/TensorRT/include/NvInfer.h
vendor/TensorRT/lib/libnvinfer.so
vendor/TensorRT/lib/libnvonnxparser.so
```

Install the ONNX Runtime **C++** distribution (the pip wheel alone lacks the
required C++ headers):

```bash
curl -L --fail -o /tmp/onnxruntime.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
mkdir -p vendor/onnxruntime
tar -xzf /tmp/onnxruntime.tgz --strip-components=1 -C vendor/onnxruntime
bash launchers/build_sonic.sh
```

The C++ executable is rebuilt from the included SONIC source, including its
streamed-motion merger modifications. ROS2 is disabled for this simulation.
Unitree SDK headers and libraries used by this build are included with their
licenses. If TensorRT/ONNX Runtime are installed elsewhere, set `TENSORRT_ROOT`
and `onnxruntime_ROOT` when building; placing them at the paths above avoids
needing extra environment configuration later.

## 5. Select the checkpoint in one file

Edit the repository root's **`simulation.env`**:

```bash
MM2_CHECKPOINT="./checkpoints/moiton mamba 2.0.pt"
```

Absolute paths also work. Quote paths containing spaces. On this four-GPU setup
the other values already have sensible defaults. GPU placement, when needed,
is edited in the same file:

```bash
MM2_GPU_TEXT=0
MM2_GPU_GENERATOR=1
MM2_GPU_VIEWER=2
MM2_GPU_SIM=3
MM2_INFERENCE_BACKEND=optimized
MM2_SESSION_WINDOWS=0
```

For subsequent model changes, **change only `MM2_CHECKPOINT`**, stop and start
the services below. Model architecture is read from checkpoint metadata.
Supported inputs are compatible Native/Minimal G1 training `.pt` checkpoints,
saved ARDY G1 Transformer checkpoints, and official ARDY G1 release directories
containing `config.yaml` and weights. A Core/HumanML3D checkpoint is not a G1
model. A checkpoint using different normalization or an unsupported architecture
is rejected explicitly rather than played with incorrect statistics.

To select the official ARDY release downloaded above:

```bash
MM2_CHECKPOINT="./checkpoints/official/ARDY-G1-RP-25FPS-Horizon52"
```

## 6. Start and stop

From any directory:

```bash
bash /path/to/MotionMamba2.0/launchers/start_ardy_sonic_all.sh
```

Or from this checkout:

```bash
bash launchers/start_ardy_sonic_all.sh
```

The old invocation with `--generator motion_mamba --debug-mode ardy_streaming
--time-stretch 1.0` is also accepted, but is unnecessary. The checkpoint selects
the backend. There is no separate `--restart` step.

Open **Viewer: http://127.0.0.1:2333** and **SONIC: http://127.0.0.1:8080**.
For a remote machine, forward both browser ports:

```bash
ssh -L 2333:127.0.0.1:2333 -L 8080:127.0.0.1:8080 user@server
```

Stop this checkout's six services with:

```bash
bash launchers/stop_ardy_sonic_all.sh
```

Sessions use a checkout-specific tmux prefix; stopping one checkout does not
kill another checkout or unrelated training. Startup refuses ports owned by
other processes and refuses a different already-running generator checkpoint.
After editing the checkpoint, stop before starting again.

When running multiple copies on one host, give each copy distinct TCP/ZMQ ports
and set a distinct `SONIC_DDS_DOMAIN_ID` (0–232, default 0). Different ports alone
do not isolate the policy-to-physics DDS channel. A normal single-copy setup
needs no DDS override.

## 7. What is accelerated and what startup does

The default is optimized inference: fused Mamba-3 operations, prepared attention,
shared-KV CUDA graphs, masked future-control graphs and fixed forward launch
configurations. Native/Minimal startup runs paired eager/optimized validation
including output/state/RNG equality, sparse controls, FIFO eviction, stopping,
restart, retry and prefix replanning. It also checks offline/incremental parity.
The final graph warmup covers future horizons before opening the generator port.
An equivalence failure stops startup; the launcher does not silently run a slow
or numerically different fallback while claiming acceleration is on.

**First startup takes several minutes**, with additional time for SONIC's local
TensorRT engine build. Wait for the final Viewer/SONIC URLs. Do not confuse
initial compilation with per-window generation speed. Cached gates are tied to
the actual checkpoint, source files, GPU model and Torch version; a newly
selected checkpoint is checked again. CUDA graphs are recreated for each
process. TensorRT engines are built locally, not copied between devices.

The reference deployment generates a warmed 52-frame / 25 FPS window in roughly
0.22–0.26 seconds. That is generation time, not an unconditional end-to-end input
latency guarantee. Text encoding, control admission, buffering, the browser and
physical execution also contribute. The Viewer includes batched mesh/quaternion
updates, stable constraint geometry and 2D timeline markers, bounded passive GUI
updates, 50 Hz pose playback deadlines and disabled costly dynamic shadows.
SONIC runs synchronized 50 Hz commands with four 5 ms physics steps per command.

## 8. Controls and verification

The Viewer keeps the original ARDY planning timeline. Generate/preload windows,
review them, seek back to frame 0 and Play. Select prompts from the ten-entry
Prompt List, enter arbitrary text, and use Target Velocity / Target Heading or
2D Root targets for keyboard/mouse control. The SONIC page displays the physical
execution of the same reference clock. A control update can only replace an
unconsumed accepted boundary; it cannot retroactively change consumed physics.

For a warmed Native/Minimal generator, run the bounded RPC smoke test:

```bash
.venv/bin/python simulation/verify_service.py
```

It uses a separate generator session, exercises six windows and future-target
replans, and saves `runtime/SERVICE_ACCEPTANCE.json`. It does not send its test
motion to physical execution. Logs are under `runtime/logs/`, with per-model
load/gate evidence under `runtime/simulation_previews/`.

If startup fails, inspect the specific component log (`text.log`, `generator.log`,
`web.log`, `sim.log`, `policy.log`, `viewer.log`). Verify GPU capacity, all downloads,
TensorRT libraries and free ports before restarting. There is no forced proxy;
set `HTTPS_PROXY`/`HTTP_PROXY` for downloads only if your network requires one.

## Included source and licensing

`simulation/runtime/` is the model's frozen implementation;
`simulation/live/` contains the deployed controls/replanning/acceleration overlay.
`ardy/` and `vendor/viser/` contain the actual modified viewer and its compiled UI.
`GR00T-WholeBodyControl/` and `ardy_sonic_web/` contain physical execution and web
transport. Third-party licenses remain alongside their sources. Downloaded model
assets retain their upstream terms; access to this private repo does not grant
access to gated upstream models.
