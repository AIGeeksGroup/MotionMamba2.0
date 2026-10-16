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
| Deployed vs fresh-clone generator in a new environment | Exact normalized-motion and qpos equality, two seeds × eight windows, including a text switch |
| Deployed vs fresh-clone text encoder | Exact embedding equality for walking, waving and turning prompts |
| Live generator RPC | Six windows plus five future replans passed; warmed windows 0.219–0.234 seconds, median 0.227 seconds |
| Playback/control regressions | 18 tests passed |
| Headless browser | Keyboard velocity/heading controls, five 2D timeline markers, rendered geometry and play/pause passed |
| Six-service physical execution in a new environment | Viewer and SONIC loaded together; reference clock advanced 6.242 s and physics clock 6.252 s during the same bounded play/pause check; epoch matched, no fall or invalid packets |
| Asset downloader | Pinned official/Meta/LLM2Vec/SONIC assets resolved and exported successfully; existing pinned weights reused, missing metadata fetched from the official Hub |
| Published HF checkpoint | Full 4,570,178,548-byte download verified against the original SHA-256; commit fa25cde54468a8a2e72adddd05b5389e79af58ee |

The browser check used software WebGL and is a functional check, not a browser
FPS benchmark. The warmed window time is generator compute time; it excludes
text encoding, admission/buffering, network, rendering and physical execution.
Cross-device arithmetic and real-time performance are not guaranteed by these
same-device checks. Each supported Native/Minimal checkpoint must pass its own
startup equivalence gate.

Complete six-service startup was exercised on isolated ports and DDS domain 63.
This found and corrected two packaging defects: local adapter directories were
being mistaken for full text models, and SONIC's startup reference CSVs were
missing. SONIC's three TensorRT engines were rebuilt locally from the pinned ONNX
files. The final check used a fresh clone of the published GitHub branch,
independently installed Python 3.11/3.10 virtual environments, and a C++ executable
rebuilt from that clone. Downloaded model assets and installed TensorRT/ONNX Runtime
distributions were reused. The original production services remained running.
This verifies the packaged code and dependencies on the reference host, rather
than a newly installed operating system or every GPU/browser combination.

The installer uses the tested transitive dependency lock files. Both the stock
PyTorch CUDA 12.4 distribution and the PyPI 2.6.0 wheel report Torch 2.6.0+cu124;
the independent installation used the latter, followed by Triton 3.5.1. The exact
service-output comparison above checks the resulting arithmetic, including text
conditioning, against the deployed service.

Re-run `simulation/verify_service.py` after startup for bounded generation and
future-control acceptance. Check the component logs and generated admission
records under `runtime/` if startup fails. Downloaded assets, environments,
hardware-specific engines, credentials and generated logs are not Git assets.
