"""Stage-08 train-fit/dev-eval Attention--Mamba complementarity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from analyze_ardy_stage07_complementarity import (  # noqa: E402
    ScalarGate, _evaluate, _fit_affine, _fit_gate, _fit_scalar,
)
from train_ardy_mamba_cached import _make_blocks  # noqa: E402
from train_ardy_minimal_cached import (  # noqa: E402
    TrajectoryDataset, _commit_history, _move, _reset,
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--minimal-checkpoint", required=True)
    parser.add_argument("--mamba-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-records", default=None)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _flat(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().flatten()


def _collect(raw, mamba_root, mamba_body, manifest: str, device: torch.device):
    dataset = TrajectoryDataset(manifest, None)
    print(json.dumps({"event": "collect_start", "manifest": manifest,
                      "trajectories": len(dataset)}), flush=True)
    selected = {0, dataset.manifest["diffusion_steps"] // 2,
                dataset.manifest["diffusion_steps"] - 1}
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for trajectory, sample in enumerate(dataset.samples):
            steps = dataset.manifest["diffusion_steps"]
            for call_index, call in enumerate(sample["calls"]):
                diffusion = call_index % steps
                if diffusion not in selected:
                    continue
                chunk = call_index // steps
                _reset(raw)
                _commit_history(raw, sample, chunk)
                root_inputs = _move(call["root"]["inputs"], device)
                body_inputs = _move(call["body"]["inputs"], device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ar = raw.root_model(*root_inputs)
                    ab = raw.body_model(*body_inputs)
                    mr = mamba_root(*root_inputs)
                    mb = mamba_body(*body_inputs)
                for stage, a, m, target in (
                    ("root", ar, mr, call["root"]["teacher_output"]),
                    ("body", ab, mb, call["body"]["teacher_output"]),
                ):
                    records.append({
                        "trajectory": trajectory,
                        "chunk": chunk,
                        "diffusion_index": diffusion,
                        "stage": stage,
                        "minimal": _flat(a),
                        "mamba": _flat(m),
                        "target": _flat(target),
                    })
            if trajectory == 0 or (trajectory + 1) % 8 == 0:
                print(json.dumps({"event": "collect_progress", "manifest": manifest,
                                  "trajectories_done": trajectory + 1}), flush=True)
    _reset(raw)
    return records


def _combined(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
    for record in records:
        key = (record["trajectory"], record["chunk"], record["diffusion_index"])
        group = grouped.setdefault(key, {
            "trajectory": key[0], "chunk": key[1], "diffusion_index": key[2],
            "minimal_parts": [], "mamba_parts": [], "target_parts": [],
        })
        group["minimal_parts"].append(record["minimal"])
        group["mamba_parts"].append(record["mamba"])
        group["target_parts"].append(record["target"])
    return [{
        "trajectory": group["trajectory"],
        "chunk": group["chunk"],
        "diffusion_index": group["diffusion_index"],
        "minimal": torch.cat(group["minimal_parts"]),
        "mamba": torch.cat(group["mamba_parts"]),
        "target": torch.cat(group["target_parts"]),
    } for group in grouped.values()]


def _fit(train: list[dict[str, Any]], seed: int):
    return _fit_scalar(train), _fit_affine(train), _fit_gate(train, seed)


def _breakdowns(records, scalar, affine, gate):
    dimensions = {
        "stage": lambda r: r.get("stage", "combined"),
        "diffusion_index": lambda r: str(r["diffusion_index"]),
        "chunk": lambda r: str(r["chunk"]),
    }
    output = {}
    for dimension, key_fn in dimensions.items():
        groups = defaultdict(list)
        for record in records:
            groups[key_fn(record)].append(record)
        output[dimension] = {
            key: _evaluate(group, scalar, affine, gate)
            for key, group in sorted(groups.items())
        }
    return output


def main() -> int:
    args = _args()
    torch.manual_seed(args.seed)
    if args.reuse_records:
        cached = torch.load(args.reuse_records, map_location="cpu", weights_only=False)
        train_stage = cached["train"]
        dev_stage = cached["development"]
    else:
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        sys.path.insert(0, str(Path(args.ardy_root).resolve()))
        from ardy.model import load_model
        from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

        model = load_model(args.teacher_model, device=args.device, text_encoder=False,
                           checkpoints_dir=args.checkpoints_dir)
        install_ardy_sequence_kernel(model, ArdySequenceKernelConfig(
            mode="minimal", attention_backend="sdpa", window_size_chunks=4,
            num_sink_tokens=4,
        ))
        raw = model.denoiser.model
        minimal = torch.load(args.minimal_checkpoint, map_location=device, weights_only=False)
        raw.root_model.load_state_dict(minimal["root_model"], strict=True)
        raw.body_model.load_state_dict(minimal["body_model"], strict=True)
        raw.eval()
        mamba_root, mamba_body = _make_blocks(raw, 8, device)
        mamba = torch.load(args.mamba_checkpoint, map_location=device, weights_only=False)
        mamba_root.load_state_dict(mamba["root_model"], strict=True)
        mamba_body.load_state_dict(mamba["body_model"], strict=True)
        mamba_root.eval(); mamba_body.eval()
        train_stage = _collect(raw, mamba_root, mamba_body, args.train_manifest, device)
        dev_stage = _collect(raw, mamba_root, mamba_body, args.dev_manifest, device)
    train = _combined(train_stage)
    dev = _combined(dev_stage)
    scalar, affine, gate = _fit(train, args.seed)
    result = {
        "phase": "stage08_train_fit_dev_eval",
        "final_heldout_used": False,
        "hashes": {
            "minimal_checkpoint": _sha256(args.minimal_checkpoint),
            "mamba_checkpoint": _sha256(args.mamba_checkpoint),
            "train_manifest": _sha256(args.train_manifest),
            "dev_manifest": _sha256(args.dev_manifest),
        },
        "train_cases": len(train), "development_cases": len(dev),
        "fitted_scalar_weight_on_minimal": scalar,
        "fitted_affine": affine.tolist(),
        "train": _evaluate(train, scalar, affine, gate),
        "development": _evaluate(dev, scalar, affine, gate),
        "development_breakdown": _breakdowns(dev_stage, scalar, affine, gate),
    }
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / "complementarity.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"train": train_stage, "development": dev_stage,
                "gate_state": gate.state_dict(), "result": result},
               output / "records.pt")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
