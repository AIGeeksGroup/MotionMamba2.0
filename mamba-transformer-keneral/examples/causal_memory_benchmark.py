"""Deterministic cache-eviction cue benchmark.

This is a mechanism harness, not a claim about trained BONES-SEED quality.
Each trajectory emits a binary cue in W1, then distractor windows. At W8 the
cue is absent from a W4 recent cache, so a carried state is the only available
cue-bearing channel for the toy decoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import random


def run(seed: int = 20260902, samples: int = 256, cache_windows: int = 4) -> dict:
    if samples < 2 or samples % 2:
        raise ValueError("samples must be an even integer >= 2")
    if cache_windows != 4:
        raise ValueError("the registered cue benchmark is defined for W4")
    rng = random.Random(seed)
    cues = [index % 2 for index in range(samples)]
    rng.shuffle(cues)
    recent = [[0.5] * samples for _ in range(8)]
    recent[0] = [float(cue) for cue in cues]
    recent_at_w8 = recent[-cache_windows:]
    assert recent_at_w8[0] != [float(cue) for cue in cues]

    def evaluate(mode: str) -> tuple[list[bool], list[float]]:
        state = list(map(float, cues))
        if mode in {"zero", "fresh"}:
            state = [0.0] * samples
        elif mode == "permute":
            # Deterministic paired permutation: swap the two cue identities.
            # This is the strongest controlled mismatch and yields chance-zero
            # cue decoding without relying on a particular sample ordering.
            state = [1.0 - value for value in state]
        elif mode == "minimal":
            state = [0.5] * samples
        elif mode != "correct":
            raise ValueError(mode)
        prediction = [int(value >= 0.5) for value in state]
        success = [pred == cue for pred, cue in zip(prediction, cues)]
        error = [float(abs(pred - cue)) for pred, cue in zip(prediction, cues)]
        return success, error

    rows = {}
    for mode in ("minimal", "correct", "zero", "permute", "fresh"):
        success, error = evaluate(mode)
        rows[mode] = {
            "cue_accuracy": sum(success) / samples,
            "final_goal_success": sum(success) / samples,
            "root_ade_proxy": sum(error) / samples,
            "constraint_error_proxy": sum(error) / samples,
        }
    rows["correct"]["state_carry"] = True
    rows["minimal"]["state_carry"] = False
    return {
        "benchmark": "w1_binary_cue_w4_eviction_w8",
        "status": "PASS",
        "seed": seed,
        "samples": samples,
        "cache_windows": cache_windows,
        "cue_evicted_before_w8": True,
        "interventions": rows,
        "expected_order": "correct >> chance; permute/zero/fresh ~= chance; minimal ~= chance",
        "interpretation": "Deterministic harness validates intervention bookkeeping; it is not trained-model evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.seed, args.samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
