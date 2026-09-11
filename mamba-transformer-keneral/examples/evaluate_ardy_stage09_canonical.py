"""Canonical isolated/coupled evaluator for the bounded Stage-09 matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
from train_ardy_mamba_cached import _make_blocks  # noqa: E402
from train_ardy_minimal_cached import TrajectoryDataset, _branch_zero, _call, _move  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ardy-root", required=True)
    p.add_argument("--checkpoints-dir", required=True)
    p.add_argument("--minimal-checkpoint", required=True)
    p.add_argument("--minimal-b-checkpoint", required=True)
    p.add_argument("--mamba-checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reuse-records", default=None)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=241)
    return p.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_attention(block):
    return hasattr(block, "prepare_accepted_commit") and hasattr(block, "reset_memory")


def reset_block(block):
    if is_attention(block):
        block.reset_memory()


def commit_attention_history(block, sample, current_chunk, stage, num_frames_per_token):
    if not is_attention(block) or current_chunk < 1:
        return
    clean = _call(sample, current_chunk, len(sample["calls"]) // sample["num_chunks"] - 1)
    inputs = _move(_branch_zero(clean[stage]["inputs"]), block._stack_parameter().device)
    history_valid = clean["outer_inputs"]["history_token_mask"][:1].bool()
    tokens_per_chunk = sample["chunk_frames"] // num_frames_per_token
    window = block.sequence_stack.layers[0].mixer.config.attention.window_size_chunks
    for history_chunk in range(max(0, current_chunk - window), current_chunk):
        selection = torch.zeros_like(inputs[1], dtype=torch.bool)
        start = history_chunk * tokens_per_chunk
        selection[:, start:start + tokens_per_chunk] = True
        if bool((selection.cpu() & ~history_valid).any()):
            raise RuntimeError("history selection includes a non-history token")
        block.prepare_accepted_commit(selection, chunk_id=history_chunk)
        block(*inputs)


class Capture:
    def __init__(self, root, body):
        self.outputs = {}
        self.handles = [
            root.register_forward_hook(lambda _m, _i, out: self.outputs.__setitem__("root", out.detach())),
            body.register_forward_hook(lambda _m, _i, out: self.outputs.__setitem__("body", out.detach())),
        ]

    def close(self):
        for handle in self.handles:
            handle.remove()


def flat(x):
    return x.detach().float().cpu().flatten()


def evaluate_candidate(raw, root, body, dataset, device):
    raw.root_model, raw.body_model = root, body
    raw.eval(); root.eval(); body.eval()
    records = []; elapsed = 0.0
    peak = 0
    selected = {0, dataset.manifest["diffusion_steps"] // 2,
                dataset.manifest["diffusion_steps"] - 1}
    capture = Capture(root, body)
    with torch.inference_mode():
        for trajectory, sample in enumerate(dataset.samples):
            steps = dataset.manifest["diffusion_steps"]
            for call_index, call in enumerate(sample["calls"]):
                diffusion = call_index % steps
                if diffusion not in selected:
                    continue
                chunk = call_index // steps
                reset_block(root); reset_block(body)
                commit_attention_history(root, sample, chunk, "root", raw.num_frames_per_token)
                commit_attention_history(body, sample, chunk, "body", raw.num_frames_per_token)
                capture.outputs.clear()
                outer = _move(call["outer_inputs"], device)
                torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    coupled = raw(**outer)
                torch.cuda.synchronize(device); elapsed += time.perf_counter() - started
                peak = max(peak, torch.cuda.max_memory_allocated(device))
                if set(capture.outputs) != {"root", "body"}:
                    raise RuntimeError("coupled ARDY call did not execute both selected stages")
                coupled_stage = dict(capture.outputs)
                # Isolated diagnostics use teacher-provided canonical block inputs.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    isolated_root = root(*_move(call["root"]["inputs"], device))
                    isolated_body = body(*_move(call["body"]["inputs"], device))
                record = {
                    "trajectory": trajectory, "chunk": chunk,
                    "diffusion_index": diffusion,
                    "coupled": flat(coupled),
                    "coupled_root": flat(coupled_stage["root"]),
                    "coupled_body": flat(coupled_stage["body"]),
                    "isolated_root": flat(isolated_root),
                    "isolated_body": flat(isolated_body),
                    "target": flat(call["teacher_output"]),
                    "root_target": flat(call["root"]["teacher_output"]),
                    "body_target": flat(call["body"]["teacher_output"]),
                }
                if not all(torch.isfinite(value).all() for key, value in record.items()
                           if isinstance(value, torch.Tensor)):
                    raise RuntimeError("non-finite canonical prediction")
                records.append(record)
            print(json.dumps({"event": "canonical_progress", "trajectory": trajectory + 1}), flush=True)
    capture.close(); reset_block(root); reset_block(body)
    return records, {"mean_forward_seconds": elapsed / len(records),
                     "peak_allocated_mib": peak / 2**20}


def mse(prediction, target):
    return float((prediction - target).square().mean())


def summarize(records, bootstrap_samples, seed):
    per_call = []
    for record in records:
        root_mse = mse(record["coupled_root"], record["root_target"])
        body_mse = mse(record["coupled_body"], record["body_target"])
        per_call.append({
            "total": root_mse + body_mse,
            "full_output": mse(record["coupled"], record["target"]),
            "root": root_mse,
            "body": body_mse,
            "isolated_root": mse(record["isolated_root"], record["root_target"]),
            "isolated_body": mse(record["isolated_body"], record["body_target"]),
            "chunk": record["chunk"], "diffusion_index": record["diffusion_index"],
            "trajectory": record["trajectory"],
        })
    output = {key: sum(row[key] for row in per_call) / len(per_call)
              for key in ("total", "full_output", "root", "body", "isolated_root", "isolated_body")}
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(per_call), (bootstrap_samples, len(per_call)), generator=generator)
    for key in ("total", "root", "body"):
        values = torch.tensor([row[key] for row in per_call], dtype=torch.float64)
        means = values[indices].mean(1)
        output[f"{key}_bootstrap95"] = [float(means.quantile(0.025)), float(means.quantile(0.975))]
    for dimension in ("chunk", "diffusion_index"):
        groups = defaultdict(list)
        for row in per_call: groups[str(row[dimension])].append(row)
        output[f"by_{dimension}"] = {
            name: {key: sum(row[key] for row in rows) / len(rows)
                   for key in ("total", "root", "body")}
            for name, rows in sorted(groups.items())
        }
    output["cases"] = len(per_call)
    return output


def ensemble_records(left, right, weight=0.5):
    result = []
    for a, b in zip(left, right):
        if (a["trajectory"], a["chunk"], a["diffusion_index"]) != (
            b["trajectory"], b["chunk"], b["diffusion_index"]
        ):
            raise RuntimeError("ensemble record alignment mismatch")
        row = dict(a)
        for key in ("coupled", "coupled_root", "coupled_body", "isolated_root", "isolated_body"):
            row[key] = weight * a[key] + (1 - weight) * b[key]
        result.append(row)
    return result


def parameter_count(*modules):
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def main():
    args = parse_args(); torch.manual_seed(args.seed)
    if args.reuse_records:
        records = torch.load(args.reuse_records, map_location="cpu", weights_only=False)
        summaries = {
            name: summarize(rows, args.bootstrap_samples, args.seed + index)
            for index, (name, rows) in enumerate(records.items())
        }
        for name in summaries:
            if name != "B0":
                summaries[name]["relative_total_gain_vs_B0"] = (
                    1.0 - summaries[name]["total"] / summaries["B0"]["total"]
                )
        previous_path = Path(args.reuse_records).with_name("canonical_results.json")
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        result = {
            "protocol": {"metric": "coupled root MSE + coupled body MSE (repository training objective)",
                         "diagnostic_metric": "per-element coupled full-output MSE",
                         "diffusion_indices": [0, 5, 9], "precision": "bf16",
                         "minimal_backend": "sdpa", "final_heldout_used": False,
                         "resummarized_from_frozen_records": True},
            "hashes": previous.get("hashes", {}), "summaries": summaries,
            "runtime": previous.get("runtime", {}),
        }
        output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
        torch.save(records, output / "canonical_records.pt")
        (output / "canonical_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2)); return 0
    device = torch.device(args.device); torch.cuda.set_device(device)
    sys.path.insert(0, str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model("Ardy-Core-RP-20FPS-Horizon40", device=args.device,
                       text_encoder=False, checkpoints_dir=args.checkpoints_dir)
    install_ardy_sequence_kernel(model, ArdySequenceKernelConfig(
        mode="minimal", attention_backend="sdpa", window_size_chunks=4, num_sink_tokens=4))
    raw = model.denoiser.model
    checkpoint_a = torch.load(args.minimal_checkpoint, map_location=device, weights_only=False)
    raw.root_model.load_state_dict(checkpoint_a["root_model"], strict=True)
    raw.body_model.load_state_dict(checkpoint_a["body_model"], strict=True)
    attention_a = (raw.root_model, raw.body_model)
    attention_b = (copy.deepcopy(raw.root_model), copy.deepcopy(raw.body_model))
    checkpoint_b = torch.load(args.minimal_b_checkpoint, map_location=device, weights_only=False)
    attention_b[0].load_state_dict(checkpoint_b["root_model"], strict=True)
    attention_b[1].load_state_dict(checkpoint_b["body_model"], strict=True)
    mamba = _make_blocks(raw, 8, device)
    checkpoint_m = torch.load(args.mamba_checkpoint, map_location=device, weights_only=False)
    mamba[0].load_state_dict(checkpoint_m["root_model"], strict=True)
    mamba[1].load_state_dict(checkpoint_m["body_model"], strict=True)
    dataset = TrajectoryDataset(args.manifest, None)
    candidates = {
        "B0": attention_a,
        "B1": mamba,
        "S1": (mamba[0], attention_a[1]),
        "D0": (attention_a[0], mamba[1]),
        "MinimalB": attention_b,
    }
    records = {}; runtime = {}; summaries = {}
    for index, (name, modules) in enumerate(candidates.items()):
        print(json.dumps({"event": "candidate_start", "candidate": name}), flush=True)
        records[name], runtime[name] = evaluate_candidate(raw, *modules, dataset, device)
        runtime[name]["active_parameters"] = parameter_count(*modules)
        summaries[name] = summarize(records[name], args.bootstrap_samples, args.seed + index)
    records["MinimalMambaAverage"] = ensemble_records(records["B0"], records["B1"])
    records["MinimalMinimalAverage"] = ensemble_records(records["B0"], records["MinimalB"])
    for index, name in enumerate(("MinimalMambaAverage", "MinimalMinimalAverage"), 10):
        summaries[name] = summarize(records[name], args.bootstrap_samples, args.seed + index)
    for name in ("B1", "S1", "D0", "MinimalB", "MinimalMambaAverage", "MinimalMinimalAverage"):
        base = summaries["B0"]["total"]
        summaries[name]["relative_total_gain_vs_B0"] = 1.0 - summaries[name]["total"] / base
    result = {
        "protocol": {"metric": "coupled root MSE + coupled body MSE (repository training objective)",
                     "diffusion_indices": [0, 5, 9], "precision": "bf16",
                     "minimal_backend": "sdpa", "final_heldout_used": False},
        "hashes": {"minimal": sha256(args.minimal_checkpoint),
                   "minimal_b": sha256(args.minimal_b_checkpoint),
                   "mamba": sha256(args.mamba_checkpoint),
                   "manifest": sha256(args.manifest)},
        "summaries": summaries, "runtime": runtime,
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    torch.save(records, output / "canonical_records.pt")
    (output / "canonical_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
