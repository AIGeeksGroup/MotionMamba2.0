"""Map frozen router rankings to exact cached Minimal subset losses."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from train_ardy_history_router import HistoryRouter, _pack  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--oracle-results", required=True)
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--router-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tokens-per-chunk", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _oracle_lookup(path: str) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    result = json.loads(Path(path).read_text())
    return {
        (
            int(case["prompt_id"]),
            int(case["seed"]),
            int(case["current_chunk"]),
            int(case["diffusion_index"]),
        ): case
        for case in result["cases"]
    }


def _case_for(example: dict[str, Any], oracle) -> dict[str, Any]:
    key = (
        int(example["prompt_id"]),
        int(example["seed"]),
        int(example["current_chunk"]),
        int(example["diffusion_index"]),
    )
    if key not in oracle:
        raise RuntimeError(f"missing oracle case {key}")
    return oracle[key]


def _subset_result(case: dict[str, Any], chunks: tuple[int, ...]) -> dict[str, Any]:
    chunks = tuple(sorted(chunks))
    if not chunks:
        return case["no_history"]
    if chunks == tuple(case["available_chunks"]):
        return case["full"]
    candidates = (
        case["singleton_candidates"] if len(chunks) == 1 else case["pair_candidates"]
    )
    for candidate in candidates:
        if tuple(candidate["chunks"]) == chunks:
            return candidate
    raise RuntimeError(f"subset {chunks} was not evaluated for oracle case")


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [record["loss"] for record in records]
    return {
        "cases": len(records),
        "mean_loss": statistics.fmean(losses),
        "median_loss": statistics.median(losses),
        "mean_history_chunks": statistics.fmean(
            record["history_chunks"] for record in records
        ),
        "mean_history_kv_tokens": statistics.fmean(
            record["history_kv_tokens"] for record in records
        ),
        "mean_total_kv_tokens": statistics.fmean(
            record["total_kv_tokens"] for record in records
        ),
        "mean_attention_flop_ratio_vs_full": statistics.fmean(
            record["attention_flop_ratio_vs_full"] for record in records
        ),
    }


def _paired(candidate: list[dict[str, Any]], full: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candidate) != len(full):
        raise ValueError("paired routing results have different lengths")
    gains = [base["loss"] - value["loss"] for value, base in zip(candidate, full)]
    relative = [
        gain / max(abs(base["loss"]), 1e-12) for gain, base in zip(gains, full)
    ]
    return {
        "mean_absolute_gain": statistics.fmean(gains),
        "median_absolute_gain": statistics.median(gains),
        "mean_relative_gain": statistics.fmean(relative),
        "fraction_strictly_better": sum(gain > 1e-12 for gain in gains) / len(gains),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    examples = cache["eval"]
    oracle = _oracle_lookup(args.oracle_results)
    router_dir = Path(args.router_dir).resolve()

    rankings: dict[str, list[list[int]]] = {}
    for variant in ("token", "state", "state_delta"):
        batch = _pack(examples, variant, device)
        router = HistoryRouter(
            batch["query"].shape[1],
            batch["candidates"].shape[2],
            args.router_dim,
            args.temperature,
        ).to(device)
        router.load_state_dict(
            torch.load(
                router_dir / f"router_{variant}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        router.eval()
        with torch.inference_mode():
            scores = router(batch["query"], batch["candidates"], batch["valid"])
        order = scores.argsort(dim=1, descending=True).cpu()
        rankings[variant] = [
            [
                int(example["candidate_chunks"][position])
                for position in order[index]
                if position < len(example["candidate_chunks"])
            ]
            for index, example in enumerate(examples)
        ]

    policies: dict[str, list[dict[str, Any]]] = {
        "full": [],
        "recent_top1": [],
        "recent_top2": [],
        "oracle_top1": [],
        "oracle_top2": [],
    }
    for variant in rankings:
        policies[f"{variant}_top1"] = []
        policies[f"{variant}_top2"] = []

    for index, example in enumerate(examples):
        case = _case_for(example, oracle)
        available = tuple(int(chunk) for chunk in example["candidate_chunks"])
        policy_chunks: dict[str, tuple[int, ...]] = {
            "full": available,
            "recent_top1": available[-1:],
            "recent_top2": available[-2:],
            "oracle_top1": tuple(case["oracle_top1"]["chunks"]),
            "oracle_top2": tuple(case["oracle_top2"]["chunks"]),
        }
        for variant, orders in rankings.items():
            policy_chunks[f"{variant}_top1"] = tuple(orders[index][:1])
            policy_chunks[f"{variant}_top2"] = tuple(orders[index][:2])
        full_total_tokens = (
            args.sink_tokens + args.tokens_per_chunk * (len(available) + 1)
        )
        for policy, chunks in policy_chunks.items():
            subset = _subset_result(case, chunks)
            total_tokens = args.sink_tokens + args.tokens_per_chunk * (len(chunks) + 1)
            policies[policy].append(
                {
                    "loss": float(subset["loss"]),
                    "root_loss": float(subset["root_loss"]),
                    "body_loss": float(subset["body_loss"]),
                    "chunks": list(sorted(chunks)),
                    "history_chunks": len(chunks),
                    "history_kv_tokens": args.tokens_per_chunk * len(chunks),
                    "total_kv_tokens": total_tokens,
                    "attention_flop_ratio_vs_full": total_tokens / full_total_tokens,
                    "current_chunk": int(example["current_chunk"]),
                    "prompt_id": int(example["prompt_id"]),
                    "diffusion_index": int(example["diffusion_index"]),
                }
            )

    summary = {}
    for policy, records in policies.items():
        summary[policy] = {
            "overall": _summarize(records),
            "by_current_chunk": {
                str(chunk): _summarize(
                    [record for record in records if record["current_chunk"] == chunk]
                )
                for chunk in sorted({record["current_chunk"] for record in records})
            },
        }
        if policy != "full":
            summary[policy]["paired_vs_full"] = _paired(records, policies["full"])

    result = {
        "format_version": 1,
        "phase": "stage06r_gate_c_frozen_subset_lookup",
        "config": vars(args),
        "cases": len(examples),
        "summary": summary,
        "records": policies,
        "latency_and_peak_memory": (
            "not measured by subset lookup; run a real frozen-model benchmark only "
            "if the denoising gate passes"
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
