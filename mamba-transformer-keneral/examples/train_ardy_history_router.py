"""Build frozen real-motion routing features and train small oracle routers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))

from train_ardy_c1_cached import _commit_history_differentiable, _load_minimal  # noqa: E402
from train_ardy_minimal_cached import (  # noqa: E402
    TrajectoryDataset,
    _branch_zero,
    _call,
    _reset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--minimal-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--train-oracle", required=True)
    parser.add_argument("--eval-oracle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-trajectories", type=int, default=64)
    parser.add_argument("--max-eval-trajectories", type=int, default=16)
    parser.add_argument("--window-size-chunks", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--state-summary-dim", type=int, default=128)
    parser.add_argument("--router-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--tiny-examples", type=int, default=32)
    parser.add_argument("--tiny-steps", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-cache", default=None)
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _oracle_lookup(path: str) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    result = json.loads(Path(path).read_text())
    lookup = {}
    for case in result["cases"]:
        key = (
            int(case["prompt_id"]),
            int(case["seed"]),
            int(case["current_chunk"]),
            int(case["diffusion_index"]),
        )
        if key in lookup:
            raise RuntimeError(f"duplicate oracle case {key}")
        lookup[key] = case
    return lookup


def _block_features(block, available: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    if block._sequence_state is None:
        raise RuntimeError("history feature extraction requires committed state")
    summaries = []
    deltas = []
    chunk_parts: dict[int, list[torch.Tensor]] = {chunk: [] for chunk in available}
    for layer_state in block._sequence_state.layers:
        if layer_state.state_summary is None or layer_state.state_delta is None:
            raise RuntimeError("hybrid layer did not expose committed state summary/delta")
        summaries.append(layer_state.state_summary.float().flatten(1).cpu())
        deltas.append(layer_state.state_delta.float().flatten(1).cpu())
        chunks = {chunk.chunk_id: chunk for chunk in layer_state.attention_cache.chunks}
        if set(chunks) != set(available):
            raise RuntimeError(
                f"attention cache IDs {sorted(chunks)} do not match {list(available)}"
            )
        for chunk_id in available:
            key = chunks[chunk_id].key.float()
            chunk_parts[chunk_id].append(key.mean(dim=2).flatten(1).cpu())
    return (
        torch.cat(summaries, dim=1),
        torch.cat(deltas, dim=1),
        {chunk: torch.cat(parts, dim=1) for chunk, parts in chunk_parts.items()},
    )


def _mean_valid_tokens(inputs: tuple[Any, ...]) -> torch.Tensor:
    branch = _branch_zero(inputs)
    tokens = branch[0].float()
    valid = branch[1].bool()
    weights = valid[:, :, None].to(tokens.dtype)
    return ((tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)).cpu()


def _extract_split(
    raw,
    dataset: TrajectoryDataset,
    oracle: dict[tuple[int, int, int, int], dict[str, Any]],
    window_size_chunks: int,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    with torch.inference_mode():
        for sample in dataset.samples:
            for current_chunk in range(2, sample["num_chunks"]):
                _reset(raw)
                _commit_history_differentiable(raw, sample, current_chunk)
                root_state, root_delta, root_chunks = _block_features(
                    raw.root_model,
                    tuple(
                        range(
                            max(0, current_chunk - window_size_chunks), current_chunk
                        )
                    ),
                )
                body_state, body_delta, body_chunks = _block_features(
                    raw.body_model,
                    tuple(
                        range(
                            max(0, current_chunk - window_size_chunks), current_chunk
                        )
                    ),
                )
                available = tuple(sorted(root_chunks))
                state = torch.cat((root_state, body_state), dim=1).squeeze(0)
                delta = torch.cat((root_delta, body_delta), dim=1).squeeze(0)
                candidates = torch.stack(
                    [
                        torch.cat((root_chunks[chunk], body_chunks[chunk]), dim=1).squeeze(0)
                        for chunk in available
                    ]
                )
                matching = sorted(
                    (
                        (key, value)
                        for key, value in oracle.items()
                        if key[:3]
                        == (sample["prompt_id"], sample["seed"], current_chunk)
                    ),
                    key=lambda item: item[0][3],
                )
                if not matching:
                    raise RuntimeError("missing oracle labels for sequential feature")
                for key, label_case in matching:
                    diffusion_index = key[3]
                    call = _call(sample, current_chunk, diffusion_index)
                    token = torch.cat(
                        (
                            _mean_valid_tokens(call["root"]["inputs"]),
                            _mean_valid_tokens(call["body"]["inputs"]),
                        ),
                        dim=1,
                    ).squeeze(0)
                    oracle_chunk = int(label_case["oracle_top1"]["chunks"][0])
                    features.append(
                        {
                            "state": state.clone(),
                            "delta": delta.clone(),
                            "token": token,
                            "candidates": candidates.clone(),
                            "candidate_chunks": torch.tensor(available, dtype=torch.long),
                            "label": available.index(oracle_chunk),
                            "prompt_id": sample["prompt_id"],
                            "seed": sample["seed"],
                            "current_chunk": current_chunk,
                            "diffusion_index": diffusion_index,
                            "oracle_distance": current_chunk - oracle_chunk,
                        }
                    )
    _reset(raw)
    return features


def _build_feature_cache(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    sys.path.insert(0, str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    torch.manual_seed(args.seed)
    model = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )
    install_ardy_sequence_kernel(
        model,
        ArdySequenceKernelConfig(
            mode="hybrid",
            attention_backend="sdpa",
            window_size_chunks=args.window_size_chunks,
            num_sink_tokens=args.sink_tokens,
            mamba_dtype=torch.bfloat16,
            token_state_interaction="state_query",
            state_summary_dim=args.state_summary_dim,
            use_state_delta=True,
            state_conditioning_scale_init=0.0,
        ),
    )
    raw = model.denoiser.model
    initialization = _load_minimal(raw, args.minimal_checkpoint, device)
    raw.eval()
    for parameter in raw.parameters():
        parameter.requires_grad_(False)

    train_data = TrajectoryDataset(args.train_manifest, args.max_train_trajectories)
    eval_data = TrajectoryDataset(args.eval_manifest, args.max_eval_trajectories)
    cache = {
        "format_version": 1,
        "minimal_initialization": initialization,
        "mamba_initialization": {
            "source": "deterministic random initialization; no trained real-motion Mamba checkpoint exists",
            "seed": args.seed,
            "backbone_frozen": True,
            "state_projector_frozen": True,
        },
        "train": _extract_split(
            raw,
            train_data,
            _oracle_lookup(args.train_oracle),
            args.window_size_chunks,
        ),
        "eval": _extract_split(
            raw,
            eval_data,
            _oracle_lookup(args.eval_oracle),
            args.window_size_chunks,
        ),
    }
    del raw, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


class HistoryRouter(nn.Module):
    def __init__(self, query_dim: int, candidate_dim: int, hidden_dim: int, temperature: float) -> None:
        super().__init__()
        self.query = nn.Linear(query_dim, hidden_dim, bias=False)
        self.candidate = nn.Linear(candidate_dim, hidden_dim, bias=False)
        self.temperature = temperature

    def forward(
        self, query: torch.Tensor, candidates: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        query = F.normalize(query.float(), dim=-1)
        candidates = F.normalize(candidates.float(), dim=-1)
        q = F.normalize(self.query(query), dim=-1)
        k = F.normalize(self.candidate(candidates), dim=-1)
        scores = torch.einsum("bd,bkd->bk", q, k) / self.temperature
        return scores.masked_fill(~valid, float("-inf"))


def _pack(examples: list[dict[str, Any]], variant: str, device: torch.device) -> dict[str, torch.Tensor]:
    max_candidates = max(example["candidates"].shape[0] for example in examples)
    candidate_dim = examples[0]["candidates"].shape[1]
    candidates = torch.zeros(len(examples), max_candidates, candidate_dim)
    valid = torch.zeros(len(examples), max_candidates, dtype=torch.bool)
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    chunks = torch.full((len(examples), max_candidates), -1, dtype=torch.long)
    if variant == "token":
        query = torch.stack([example["token"] for example in examples])
    elif variant == "state":
        query = torch.stack([example["state"] for example in examples])
    elif variant == "state_delta":
        query = torch.stack(
            [torch.cat((example["state"], example["delta"])) for example in examples]
        )
    else:
        raise ValueError(f"unknown router variant {variant}")
    for index, example in enumerate(examples):
        count = example["candidates"].shape[0]
        candidates[index, :count] = example["candidates"]
        valid[index, :count] = True
        chunks[index, :count] = example["candidate_chunks"]
    return {
        "query": query.to(device),
        "candidates": candidates.to(device),
        "valid": valid.to(device),
        "labels": labels.to(device),
        "chunks": chunks.to(device),
    }


@torch.no_grad()
def _metrics(router: HistoryRouter, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    router.eval()
    scores = router(batch["query"], batch["candidates"], batch["valid"])
    labels = batch["labels"]
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    loss = F.cross_entropy(scores, labels)
    return {
        "loss": float(loss),
        "hit_at_1": float((ranks <= 1).float().mean()),
        "hit_at_2": float((ranks <= 2).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
    }


def _recent_metrics(examples: list[dict[str, Any]]) -> dict[str, float]:
    ranks = []
    for example in examples:
        # The heuristic ranking is newest to oldest. Candidate storage is
        # oldest to newest, so convert the oracle label to that rank.
        ranks.append(len(example["candidate_chunks"]) - example["label"])
    return {
        "hit_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "hit_at_2": sum(rank <= 2 for rank in ranks) / len(ranks),
        "mrr": statistics_fmean(1.0 / rank for rank in ranks),
    }


def _recent_stratified(examples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        str(distance): _recent_metrics(
            [example for example in examples if example["oracle_distance"] == distance]
        )
        for distance in sorted({example["oracle_distance"] for example in examples})
    }


def _router_stratified(
    router: HistoryRouter,
    batch: dict[str, torch.Tensor],
    examples: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result = {}
    for distance in sorted({example["oracle_distance"] for example in examples}):
        indices = torch.tensor(
            [
                index
                for index, example in enumerate(examples)
                if example["oracle_distance"] == distance
            ],
            device=batch["labels"].device,
        )
        subset = {key: value[indices] for key, value in batch.items()}
        result[str(distance)] = _metrics(router, subset)
    return result


def statistics_fmean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _fit(
    train: dict[str, torch.Tensor],
    evaluation: dict[str, torch.Tensor],
    args: argparse.Namespace,
    steps: int,
    seed: int,
) -> tuple[HistoryRouter, dict[str, Any]]:
    torch.manual_seed(seed)
    router = HistoryRouter(
        train["query"].shape[1],
        train["candidates"].shape[2],
        args.router_dim,
        args.temperature,
    ).to(train["query"].device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial_train = _metrics(router, train)
    initial_eval = _metrics(router, evaluation)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    router.train()
    for _ in range(steps):
        indices = torch.randint(
            len(train["labels"]),
            (min(args.batch_size, len(train["labels"])),),
            generator=generator,
        ).to(train["labels"].device)
        scores = router(
            train["query"][indices],
            train["candidates"][indices],
            train["valid"][indices],
        )
        loss = F.cross_entropy(scores, train["labels"][indices])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite router loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in router.parameters()]
        if not all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("missing/non-finite router gradients")
        optimizer.step()
    return router, {
        "initial_train": initial_train,
        "final_train": _metrics(router, train),
        "initial_eval": initial_eval,
        "final_eval": _metrics(router, evaluation),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    feature_path = (
        Path(args.feature_cache).resolve()
        if args.feature_cache
        else output_dir / "router_features.pt"
    )
    if feature_path.exists():
        cache = torch.load(feature_path, map_location="cpu", weights_only=False)
    else:
        cache = _build_feature_cache(args, device)
        torch.save(cache, feature_path)
    train_examples = cache["train"]
    eval_examples = cache["eval"]
    if not train_examples or not eval_examples:
        raise RuntimeError("empty router feature split")

    middle = sorted(
        {example["diffusion_index"] for example in train_examples}
    )[len({example["diffusion_index"] for example in train_examples}) // 2]
    tiny_examples = [
        example for example in train_examples if example["diffusion_index"] == middle
    ][: args.tiny_examples]
    if len(tiny_examples) < 16:
        raise RuntimeError("router tiny-overfit requires at least 16 examples")

    result: dict[str, Any] = {
        "format_version": 1,
        "phase": "stage06r_gate_b_frozen_router",
        "config": {
            **vars(args),
            "git_commit": _git_commit(),
            "minimal_sha256": _sha256(args.minimal_checkpoint),
            "feature_cache": str(feature_path),
        },
        "feature_provenance": {
            key: value for key, value in cache.items() if key not in ("train", "eval")
        },
        "counts": {
            "train": len(train_examples),
            "eval": len(eval_examples),
            "tiny": len(tiny_examples),
        },
        "recent": _recent_metrics(eval_examples),
        "recent_by_oracle_distance": _recent_stratified(eval_examples),
        "routers": {},
    }

    for variant_index, variant in enumerate(("token", "state", "state_delta")):
        tiny = _pack(tiny_examples, variant, device)
        tiny_router, tiny_result = _fit(
            tiny, tiny, args, args.tiny_steps, args.seed + variant_index
        )
        tiny_pass = (
            tiny_result["final_train"]["loss"]
            < 0.2 * tiny_result["initial_train"]["loss"]
            and tiny_result["final_train"]["hit_at_1"] >= 0.9
        )
        del tiny_router
        if not tiny_pass:
            result["routers"][variant] = {
                "tiny_overfit": tiny_result,
                "tiny_overfit_pass": False,
            }
            (output_dir / "router_results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            raise RuntimeError(f"{variant} router failed the mandatory tiny-overfit gate")

        train = _pack(train_examples, variant, device)
        evaluation = _pack(eval_examples, variant, device)
        router, full_result = _fit(
            train,
            evaluation,
            args,
            args.train_steps,
            args.seed + 100 + variant_index,
        )
        full_result["eval_by_oracle_distance"] = _router_stratified(
            router, evaluation, eval_examples
        )
        torch.save(router.state_dict(), output_dir / f"router_{variant}.pt")
        result["routers"][variant] = {
            "tiny_overfit": tiny_result,
            "tiny_overfit_pass": True,
            "full_training": full_result,
        }
        print(json.dumps({variant: result["routers"][variant]}, indent=2), flush=True)

    result["peak_memory_mib"] = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    (output_dir / "router_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
