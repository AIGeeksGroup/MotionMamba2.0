# Deployment verification

The release checkpoint is **moiton mamba 2.0**, the original full Native G1
117,000-step checkpoint (112k foundation + 5k interaction). SHA-256:
`2e828449addff3aa079387f319589cf4ac9b202c48577278a6d274f28b14f679`.

Verification performed on the reference Linux / RTX 4090 deployment:

| Check | Observed result |
| --- | --- |
| Packaged C++ SONIC source | Rebuilt successfully with TensorRT 10.13.3.9 and ONNX Runtime C++ 1.16.3 |
| Packaged MuJoCo scene | Loaded: nq=50, nv=49, nu=43 |
| Eager vs optimized generation | Admission passed for two seeds through eight windows, including FIFO eviction and future-control/replanning cases |
| Offline/incremental, interleaved sessions, rewind/retry | Zero motion error; no double state commit |
| Deployed vs packaged generator | Exact normalized-motion and qpos equality, two seeds × eight windows, including a text switch |
| Deployed vs packaged text encoder | Exact embedding equality for walking, waving and turning prompts |
| Live generator RPC | Six windows plus five future replans passed; warmed windows approximately 0.23 seconds |
| Playback/control regressions | 18 tests passed |
| Headless browser | Keyboard velocity/heading controls, five 2D timeline markers, rendered geometry and play/pause passed |

The browser check used software WebGL and is a functional check, not a browser
FPS benchmark. The warmed window time is generator compute time; it excludes
text encoding, admission/buffering, network, rendering and physical execution.
Cross-device arithmetic and real-time performance are not guaranteed by these
same-device checks. Each supported Native/Minimal checkpoint must pass its own
startup equivalence gate.

The installer and complete six-service startup are also exercised during release
preparation. Publication status and any still-pending installation or full-chain
checks must be reported separately; a local commit alone is not a successful
remote publication.

Re-run `simulation/verify_service.py` after startup for bounded generation and
future-control acceptance. Check the component logs and generated admission
records under `runtime/` if startup fails. Downloaded assets, environments,
hardware-specific engines, credentials and generated logs are not Git assets.
