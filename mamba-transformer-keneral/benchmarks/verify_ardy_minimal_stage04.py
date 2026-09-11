"""Real-ARDY Minimal isolation and diffusion-memory acceptance gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR))

from train_ardy_dit_distill import _synthetic_teacher_batch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.ardy_root).resolve()))

    from ardy.model import load_model

    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    if not torch.cuda.is_available():
        raise RuntimeError("real ARDY gate requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(73)
    torch.cuda.reset_peak_memory_stats()
    model = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )
    client = install_ardy_sequence_kernel(
        model,
        ArdySequenceKernelConfig(
            mode="minimal",
            attention_backend="sdpa",
            window_size_chunks=2,
            num_sink_tokens=2,
        ),
    )
    raw = model.denoiser.model
    module_names = [type(module).__name__.lower() for module in raw.modules()]
    no_mamba_module = not any("mamba" in name for name in module_names)
    no_mamba_import = "mamba_ssm" not in sys.modules
    if not no_mamba_module or not no_mamba_import:
        raise RuntimeError("Minimal instantiated or imported Mamba")
    parameter_devices = {parameter.device.type for parameter in raw.root_model.parameters()}
    parameter_devices |= {parameter.device.type for parameter in raw.body_model.parameters()}
    if parameter_devices != {"cuda"}:
        raise RuntimeError(f"replacement blocks are not fully on CUDA: {parameter_devices}")

    raw_call = _synthetic_teacher_batch(model, device)
    call = {
        "x": raw_call["x"],
        "history_len": raw_call["history_len"],
        "generation_len": raw_call["generation_len"],
        "future_len": raw_call["future_len"],
        "history_mask": raw_call["history_mask"],
        "generation_mask": raw_call["generation_mask"],
        "future_mask": raw_call["future_mask"],
        "history_token_mask": raw_call["history_token_mask"],
        "generation_token_mask": raw_call["generation_token_mask"],
        "future_token_mask": raw_call["future_token_mask"],
        "text_feat": raw_call["text_feat"],
        "text_pad_mask": raw_call["text_feat_pad_mask"],
        "first_heading_angle": raw_call["first_heading_angle"],
        "motion_mask": raw_call["motion_mask"],
        "observed_motion": raw_call["observed_motion"],
        "num_denoising_steps": torch.tensor(
            [model.diffusion.num_base_steps], device=device
        ),
        "cfg_weight": 2.0,
    }
    started = time.perf_counter()
    for timestep in (2, 1):
        call["t"] = torch.full((1,), timestep, dtype=torch.long, device=device)
        call["x"] = model.denoising_step(**call)
        assert client.controller.trace.accepted_commits == 0
        assert raw.root_model.accepted_commits == 0
        assert raw.body_model.accepted_commits == 0
        assert all(
            state.attention_cache.num_chunks == 0
            for state in raw.root_model._sequence_state.layers
        )

    call["t"] = torch.zeros(1, dtype=torch.long, device=device)
    call["x"] = model.denoising_step(**call)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    trace = client.controller.trace
    if trace.accepted_commits != 1:
        raise RuntimeError(f"expected exactly one controller commit, got {trace}")
    if raw.root_model.accepted_commits != 1 or raw.body_model.accepted_commits != 1:
        raise RuntimeError("root/body did not each commit exactly once")
    if not torch.isfinite(call["x"]).all():
        raise RuntimeError("real Minimal denoising produced non-finite output")
    root_cache = raw.root_model._sequence_state.layers[0].attention_cache
    body_cache = raw.body_model._sequence_state.layers[0].attention_cache
    if root_cache.chunk_ids != (0,) or body_cache.chunk_ids != (0,):
        raise RuntimeError("accepted chunk id was not committed to both stages")

    report = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "mode": "minimal",
        "cfg_type": type(model.denoiser).__name__,
        "no_mamba_module": no_mamba_module,
        "mamba_ssm_not_imported": no_mamba_import,
        "parameter_devices": sorted(parameter_devices),
        "noisy_steps": 2,
        "noisy_commits": 0,
        "accepted_final_commits": trace.accepted_commits,
        "root_commits": raw.root_model.accepted_commits,
        "body_commits": raw.body_model.accepted_commits,
        "root_cache_ids": list(root_cache.chunk_ids),
        "body_cache_ids": list(body_cache.chunk_ids),
        "sink_tokens": root_cache.num_sink_tokens,
        "finite_output": True,
        "three_step_latency_ms": elapsed_ms,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
