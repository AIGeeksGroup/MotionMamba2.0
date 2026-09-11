"""Capture only last-chunk calls from real long-horizon ARDY generation."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "examples"))

from build_ardy_teacher_cache import _cpu_clone, _max_mean_error, _to_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--text-feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--capture-diffusion-indices", type=int, nargs="+", default=[0, 5, 9])
    parser.add_argument("--cfg-weight", type=float, nargs=2, default=[2.0, 2.0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-calls", type=int, default=12)
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _load_text_features(path: str, limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"][:limit]
    samples = []
    for record in records:
        sample = torch.load(
            manifest_path.parent / record["path"],
            map_location="cpu",
            weights_only=False,
        )
        samples.append(
            {
                "prompt": sample["prompt"],
                "prompt_id": sample["prompt_id"],
                "text_feat": sample["text_feat"],
                "text_feat_pad_mask": sample["text_feat_pad_mask"],
            }
        )
    if not samples:
        raise ValueError("text feature manifest has no records")
    return manifest, samples


class _SparseTeacherCapture:
    def __init__(
        self,
        raw: torch.nn.Module,
        diffusion_steps: int,
        target_chunk: int,
        diffusion_indices: set[int],
    ) -> None:
        self.raw = raw
        self.diffusion_steps = diffusion_steps
        self.target_chunk = target_chunk
        self.diffusion_indices = diffusion_indices
        self.calls: list[dict[str, Any]] = []
        self.total_calls = 0
        self._current: dict[str, Any] | None = None
        self._outer_names = list(inspect.signature(raw.forward).parameters)
        self._handles = [
            raw.register_forward_pre_hook(self._raw_pre, with_kwargs=True),
            raw.register_forward_hook(self._raw_post, with_kwargs=True),
            raw.root_model.register_forward_hook(self._block_hook("root")),
            raw.body_model.register_forward_hook(self._block_hook("body")),
        ]

    def _raw_pre(self, _module, args, kwargs) -> None:
        if self._current is not None:
            raise RuntimeError("nested sparse teacher capture is unsupported")
        index = self.total_calls
        self.total_calls += 1
        chunk = index // self.diffusion_steps
        diffusion_index = index % self.diffusion_steps
        if chunk != self.target_chunk or diffusion_index not in self.diffusion_indices:
            return
        outer = dict(zip(self._outer_names, args))
        outer.update(kwargs)
        self._current = {
            "chunk_index": chunk,
            "diffusion_call_index": diffusion_index,
            "outer_inputs": _cpu_clone(outer),
        }

    def _block_hook(self, name: str):
        def hook(_module, inputs, output) -> None:
            if self._current is None:
                return
            if name in self._current:
                raise RuntimeError(f"duplicate captured {name} call")
            self._current[name] = {
                "inputs": _cpu_clone(inputs),
                "teacher_output": _cpu_clone(output),
            }

        return hook

    def _raw_post(self, _module, _args, _kwargs, output) -> None:
        if self._current is None:
            return
        if not {"root", "body"} <= self._current.keys():
            raise RuntimeError("captured call is missing root/body output")
        self._current["teacher_output"] = _cpu_clone(output)
        self._current["diffusion_timestep"] = _cpu_clone(
            self._current["outer_inputs"]["timesteps"]
        )
        self.calls.append(self._current)
        self._current = None

    def close(self) -> tuple[list[dict[str, Any]], int]:
        for handle in self._handles:
            handle.remove()
        if self._current is not None:
            raise RuntimeError("incomplete sparse teacher capture")
        return self.calls, self.total_calls


def _verify(
    raw: torch.nn.Module,
    paths: list[Path],
    device: torch.device,
    limit: int,
) -> dict[str, float | int]:
    maxima = []
    means = []
    with torch.inference_mode():
        for path in paths:
            sample = torch.load(path, map_location="cpu", weights_only=False)
            for call in sample["calls"]:
                actual = raw(**_to_device(call["outer_inputs"], device))
                maximum, mean = _max_mean_error(
                    actual, call["teacher_output"].to(device)
                )
                maxima.append(maximum)
                means.append(mean)
                if len(maxima) >= limit:
                    return {
                        "num_calls": len(maxima),
                        "max_absolute_error": max(maxima),
                        "mean_absolute_error": sum(means) / len(means),
                    }
    return {
        "num_calls": len(maxima),
        "max_absolute_error": max(maxima, default=0.0),
        "mean_absolute_error": sum(means) / max(len(means), 1),
    }


def main() -> int:
    args = parse_args()
    if any(horizon <= 4 for horizon in args.horizons):
        raise ValueError("long-horizon cache requires every horizon > W=4")
    if any(
        index < 0 or index >= args.diffusion_steps
        for index in args.capture_diffusion_indices
    ):
        raise ValueError("capture diffusion index outside schedule")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    source_manifest, prompts = _load_text_features(
        args.text_feature_manifest, args.max_prompts
    )
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy.motion_rep.tools import length_to_mask
    from ardy.tools import seed_everything

    teacher = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    ).eval()
    raw = teacher.denoiser.model
    started = time.perf_counter()
    paths: list[Path] = []
    records = []
    for horizon in sorted(set(args.horizons)):
        horizon_dir = output_root / f"horizon_{horizon:03d}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        num_frames = int(horizon * teacher.gen_horizon_len)
        for prompt in prompts:
            seed_everything(args.seed)
            capture = _SparseTeacherCapture(
                raw,
                args.diffusion_steps,
                horizon - 1,
                set(args.capture_diffusion_indices),
            )
            with torch.inference_mode():
                accepted_motion = teacher(
                    [prompt["prompt"]],
                    num_frames,
                    num_denoising_steps=args.diffusion_steps,
                    pad_mask=length_to_mask(
                        torch.tensor([num_frames], device=device, dtype=torch.long)
                    ),
                    first_heading_angle=torch.zeros(1, device=device),
                    motion_mask=None,
                    observed_motion=None,
                    cfg_weight=tuple(args.cfg_weight),
                    text_feat=prompt["text_feat"].to(device),
                    text_pad_mask=prompt["text_feat_pad_mask"].to(device),
                    progress_bar=lambda values: values,
                )
            calls, total_calls = capture.close()
            if total_calls != horizon * args.diffusion_steps:
                raise RuntimeError(
                    f"expected {horizon * args.diffusion_steps} teacher calls, got {total_calls}"
                )
            if len(calls) != len(set(args.capture_diffusion_indices)):
                raise RuntimeError("sparse capture missed a requested target call")
            if not torch.isfinite(accepted_motion).all():
                raise RuntimeError("long-horizon teacher motion is non-finite")
            sample = {
                "format_version": 1,
                "prompt": prompt["prompt"],
                "prompt_id": prompt["prompt_id"],
                "seed": args.seed,
                "num_chunks": horizon,
                "target_chunk": horizon - 1,
                "chunk_frames": int(teacher.gen_horizon_len),
                "fps": float(teacher.motion_rep.fps),
                "text_feat": prompt["text_feat"],
                "text_feat_pad_mask": prompt["text_feat_pad_mask"],
                "accepted_motion": accepted_motion.detach().cpu(),
                "calls": calls,
            }
            path = horizon_dir / f"prompt_{prompt['prompt_id']:05d}_seed_{args.seed:010d}.pt"
            torch.save(sample, path)
            paths.append(path)
            records.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "prompt_id": prompt["prompt_id"],
                    "seed": args.seed,
                    "horizon": horizon,
                    "target_chunk": horizon - 1,
                    "captured_calls": len(calls),
                }
            )
            print(
                json.dumps(
                    {
                        "horizon": horizon,
                        "prompt_id": prompt["prompt_id"],
                        "frames": num_frames,
                        "captured_calls": len(calls),
                    }
                ),
                flush=True,
            )

    replay = _verify(raw, paths, device, args.verify_calls)
    manifest = {
        "format_version": 1,
        "phase": "stage06s_long_horizon_sparse_capture",
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "source_text_feature_manifest": str(Path(args.text_feature_manifest).resolve()),
        "source_text_encoder": source_manifest.get("text_encoder"),
        "horizons": sorted(set(args.horizons)),
        "num_prompts": len(prompts),
        "seed": args.seed,
        "diffusion_steps": args.diffusion_steps,
        "capture_diffusion_indices": sorted(set(args.capture_diffusion_indices)),
        "cfg_weight": args.cfg_weight,
        "records": records,
        "replay_verification": replay,
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if replay["max_absolute_error"] > 1e-5:
        raise RuntimeError("long-horizon sparse cache replay failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
