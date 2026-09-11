"""Capture real ARDY inference tensors for DiT distillation.

The cache is produced by the released ARDY autoregressive diffusion path.  It
does not synthesize text features or denoiser inputs: prompts are encoded by
the same two-stage LLM2Vec encoder as ARDY and forward hooks record the exact
root/body calls made by classifier-free-guided inference.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--cfg-weight", type=float, nargs=2, default=[2.0, 2.0])
    parser.add_argument("--text-encoder-base-mirror", required=True)
    parser.add_argument("--text-encoder-mntp-adapter", required=True)
    parser.add_argument("--text-encoder-supervised-adapter", required=True)
    parser.add_argument("--text-encoder-device", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-samples", type=int, default=4)
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _read_prompts(path: Path, limit: int | None) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    return value


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _encode_prompts(args: argparse.Namespace, prompts: list[str]) -> list[dict[str, Any]]:
    ardy_root = Path(args.ardy_root).resolve()
    sys.path.insert(0, str(ardy_root))
    from ardy.model.llm2vec import LLM2Vec
    from peft import PeftModel

    if args.text_encoder_device.startswith("cuda") and torch.cuda.device_count() != 1:
        raise RuntimeError(
            "ARDY LLM2Vec ignores an explicit CUDA index when multiple GPUs are "
            "visible and replicates the 8B encoder across every GPU. Set "
            "CUDA_VISIBLE_DEVICES to exactly one physical GPU and use "
            "--text-encoder-device cuda:0 --device cuda:0."
        )

    # Reproduce ARDY's two-stage LLM2Vec load explicitly.  Older Transformers
    # versions do not auto-detect a local adapter directory as a model source.
    # The public ARDY Space preloads this byte-compatible ungated base mirror.
    encoder = LLM2Vec.from_pretrained(
        base_model_name_or_path=str(Path(args.text_encoder_base_mirror).resolve()),
        peft_model_name_or_path=str(
            Path(args.text_encoder_mntp_adapter).resolve()
        ),
        merge_peft=True,
        torch_dtype=torch.bfloat16,
    )
    encoder.model = PeftModel.from_pretrained(
        encoder.model,
        str(Path(args.text_encoder_supervised_adapter).resolve()),
    )
    # prepare_for_tokenization keys off this canonical identity to insert the
    # exact Llama-3 Instruct user header used by the released ARDY encoder.
    encoder.model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    encoder = encoder.to(args.text_encoder_device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoded: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prompt in prompts:
            feature = encoder.encode(
                [prompt],
                batch_size=1,
                show_progress_bar=False,
                device=args.text_encoder_device,
            )
            feature = torch.as_tensor(feature).detach().cpu()[:, None, :]
            length = 1
            mask = torch.arange(feature.shape[1])[None, :] < length
            if feature.shape[-1] != 4096 or not torch.isfinite(feature).all():
                raise RuntimeError("ARDY text encoder returned invalid features")
            encoded.append(
                {"prompt": prompt, "text_feat": feature, "text_feat_pad_mask": mask}
            )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return encoded


class _TeacherCapture:
    def __init__(self, raw: torch.nn.Module, diffusion_steps: int) -> None:
        self.raw = raw
        self.diffusion_steps = diffusion_steps
        self.calls: list[dict[str, Any]] = []
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
            raise RuntimeError("nested raw denoiser capture is unsupported")
        outer = dict(zip(self._outer_names, args))
        outer.update(kwargs)
        index = len(self.calls)
        self._current = {
            "chunk_index": index // self.diffusion_steps,
            "diffusion_call_index": index % self.diffusion_steps,
            "outer_inputs": _cpu_clone(outer),
        }

    def _block_hook(self, name: str):
        def hook(_module, inputs, output) -> None:
            if self._current is None or name in self._current:
                raise RuntimeError(f"unexpected {name} teacher hook ordering")
            self._current[name] = {
                "inputs": _cpu_clone(inputs),
                "teacher_output": _cpu_clone(output),
            }

        return hook

    def _raw_post(self, _module, _args, _kwargs, output) -> None:
        if self._current is None or not {"root", "body"} <= self._current.keys():
            raise RuntimeError("teacher root/body hooks did not both execute")
        self._current["teacher_output"] = _cpu_clone(output)
        timestep = self._current["outer_inputs"]["timesteps"]
        self._current["diffusion_timestep"] = _cpu_clone(timestep)
        self.calls.append(self._current)
        self._current = None

    def close(self) -> list[dict[str, Any]]:
        for handle in self._handles:
            handle.remove()
        if self._current is not None:
            raise RuntimeError("incomplete teacher call at capture close")
        return self.calls


def _max_mean_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = (actual.float() - expected.float()).abs()
    return float(error.max()), float(error.mean())


def _replay_verify(
    raw: torch.nn.Module,
    sample_paths: list[Path],
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    maxima: list[float] = []
    means: list[float] = []
    verified = 0
    with torch.inference_mode():
        for path in sample_paths:
            sample = torch.load(path, map_location="cpu", weights_only=False)
            for call in sample["calls"]:
                outer = _to_device(call["outer_inputs"], device)
                actual = raw(**outer)
                expected = call["teacher_output"].to(device)
                maximum, mean = _max_mean_error(actual, expected)
                maxima.append(maximum)
                means.append(mean)
                verified += 1
                if verified >= max_samples:
                    return {
                        "num_calls": verified,
                        "max_absolute_error": max(maxima),
                        "mean_absolute_error": sum(means) / len(means),
                    }
    return {
        "num_calls": verified,
        "max_absolute_error": max(maxima, default=0.0),
        "mean_absolute_error": sum(means) / max(len(means), 1),
    }


def main() -> int:
    args = parse_args()
    if args.num_chunks < 1 or args.diffusion_steps < 1:
        raise ValueError("num chunks and diffusion steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    prompt_path = Path(args.prompt_file).resolve()
    prompts = _read_prompts(prompt_path, args.max_prompts)
    output_root = Path(args.output_dir).resolve()
    split_dir = output_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoded = _encode_prompts(args, prompts)

    from ardy.model import load_model
    from ardy.motion_rep.tools import length_to_mask
    from ardy.tools import seed_everything

    teacher = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )
    teacher.eval()
    raw = teacher.denoiser.model
    fps = float(teacher.motion_rep.fps)
    num_frames = int(args.num_chunks * teacher.gen_horizon_len)
    sample_paths: list[Path] = []
    records: list[dict[str, Any]] = []

    for prompt_id, encoded_prompt in enumerate(encoded):
        for seed in args.seeds:
            seed_everything(seed)
            text_feat = encoded_prompt["text_feat"].to(device)
            text_mask = encoded_prompt["text_feat_pad_mask"].to(device)
            capture = _TeacherCapture(raw, args.diffusion_steps)
            with torch.inference_mode():
                accepted_motion = teacher(
                    [encoded_prompt["prompt"]],
                    num_frames,
                    num_denoising_steps=args.diffusion_steps,
                    pad_mask=length_to_mask(
                        torch.tensor([num_frames], device=device, dtype=torch.long)
                    ),
                    first_heading_angle=torch.zeros(1, device=device),
                    motion_mask=None,
                    observed_motion=None,
                    cfg_weight=tuple(args.cfg_weight),
                    text_feat=text_feat,
                    text_pad_mask=text_mask,
                    progress_bar=lambda values: values,
                )
            calls = capture.close()
            expected_calls = args.num_chunks * args.diffusion_steps
            if len(calls) != expected_calls:
                raise RuntimeError(
                    f"expected {expected_calls} teacher calls, captured {len(calls)}"
                )
            if not torch.isfinite(accepted_motion).all():
                raise RuntimeError("teacher generation produced non-finite motion")
            sample = {
                "format_version": 1,
                "prompt": encoded_prompt["prompt"],
                "prompt_id": prompt_id,
                "seed": seed,
                "split": args.split,
                "text_feat": encoded_prompt["text_feat"],
                "text_feat_pad_mask": encoded_prompt["text_feat_pad_mask"],
                "accepted_motion": accepted_motion.detach().cpu(),
                "num_chunks": args.num_chunks,
                "chunk_frames": int(teacher.gen_horizon_len),
                "fps": fps,
                "calls": calls,
            }
            filename = f"prompt_{prompt_id:05d}_seed_{seed:010d}.pt"
            sample_path = split_dir / filename
            torch.save(sample, sample_path)
            sample_paths.append(sample_path)
            records.append(
                {
                    "path": str(sample_path.relative_to(output_root)),
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "num_calls": len(calls),
                }
            )

    replay = _replay_verify(raw, sample_paths, device, args.verify_samples)
    manifest = {
        "format_version": 1,
        "split": args.split,
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "text_encoder": {
            "base_mirror": str(Path(args.text_encoder_base_mirror).resolve()),
            "base_mirror_repo": "unsloth/llama-3-8b-Instruct",
            "mntp_adapter": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "supervised_adapter": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        },
        "prompt_file": str(prompt_path),
        "num_prompts": len(prompts),
        "seeds": args.seeds,
        "num_chunks": args.num_chunks,
        "diffusion_steps": args.diffusion_steps,
        "cfg_weight": args.cfg_weight,
        "records": records,
        "replay_verification": replay,
        "git_commit": _git_commit(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = output_root / f"manifest_{args.split}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if replay["max_absolute_error"] > 1e-5:
        raise RuntimeError("teacher cache replay parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
