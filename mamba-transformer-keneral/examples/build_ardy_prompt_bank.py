"""Build curated text-to-motion prompt banks for ARDY-Mamba distillation.

The v3 prompt bank proved that naive template scaling can hurt quality. This
generator keeps grammar simple and avoids physically odd action/style pairs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


ATOMIC_ACTIONS = [
    "walks forward",
    "walks backward",
    "walks to the left",
    "walks to the right",
    "walks in a circle",
    "runs forward",
    "jogs in place",
    "marches in place",
    "jumps",
    "jumps twice",
    "squats down",
    "stands up",
    "bends down",
    "kneels down",
    "turns around",
    "spins once",
    "waves",
    "claps",
    "raises both arms",
    "raises the right hand",
    "raises the left hand",
    "kicks with the right leg",
    "kicks with the left leg",
    "dances happily",
    "shuffles forward",
    "shuffles backward",
    "leans forward",
    "leans backward",
    "reaches forward",
    "reaches to the left",
    "reaches to the right",
    "points forward",
]

STYLE_BY_KIND = {
    "locomotion": ["slowly", "quickly", "carefully", "smoothly", "with small steps", "with steady steps"],
    "stationary": ["slowly", "carefully", "smoothly", "with control", "while keeping balance"],
    "upper_body": ["slowly", "quickly", "smoothly", "with both feet planted", "while standing still"],
    "dynamic": ["quickly", "energetically", "smoothly", "with control"],
}

ACTION_KIND = {
    "walks forward": "locomotion",
    "walks backward": "locomotion",
    "walks to the left": "locomotion",
    "walks to the right": "locomotion",
    "walks in a circle": "locomotion",
    "runs forward": "locomotion",
    "jogs in place": "locomotion",
    "marches in place": "locomotion",
    "shuffles forward": "locomotion",
    "shuffles backward": "locomotion",
    "jumps": "dynamic",
    "jumps twice": "dynamic",
    "kicks with the right leg": "dynamic",
    "kicks with the left leg": "dynamic",
    "dances happily": "dynamic",
    "squats down": "stationary",
    "stands up": "stationary",
    "bends down": "stationary",
    "kneels down": "stationary",
    "turns around": "stationary",
    "spins once": "stationary",
    "leans forward": "stationary",
    "leans backward": "stationary",
    "waves": "upper_body",
    "claps": "upper_body",
    "raises both arms": "upper_body",
    "raises the right hand": "upper_body",
    "raises the left hand": "upper_body",
    "reaches forward": "upper_body",
    "reaches to the left": "upper_body",
    "reaches to the right": "upper_body",
    "points forward": "upper_body",
}

VALID_TRANSITIONS = [
    ("walks forward", "turns around"),
    ("walks forward", "waves"),
    ("walks backward", "turns around"),
    ("walks to the left", "turns around"),
    ("walks to the right", "turns around"),
    ("runs forward", "slows to a walk"),
    ("jogs in place", "stops"),
    ("marches in place", "stops"),
    ("jumps", "lands softly"),
    ("jumps twice", "stands still"),
    ("squats down", "stands up"),
    ("bends down", "stands up"),
    ("kneels down", "stands up"),
    ("turns around", "waves"),
    ("spins once", "stands still"),
    ("waves", "lowers the hand"),
    ("raises both arms", "lowers both arms"),
    ("raises the right hand", "lowers the right hand"),
    ("raises the left hand", "lowers the left hand"),
    ("kicks with the right leg", "steps back"),
    ("kicks with the left leg", "steps back"),
    ("leans forward", "returns upright"),
    ("leans backward", "returns upright"),
    ("reaches forward", "returns the arms"),
    ("reaches to the left", "returns to center"),
    ("reaches to the right", "returns to center"),
    ("dances happily", "spins once"),
]


def _style_prompts(action: str) -> list[str]:
    kind = ACTION_KIND[action]
    return [f"A person {action} {style}." for style in STYLE_BY_KIND[kind]]


def _transition_prompts(first: str, second: str) -> list[str]:
    prompts = [
        f"A person {first}, then {second}.",
        f"A person {first} and then {second}.",
    ]
    kind = ACTION_KIND.get(first, "stationary")
    for style in STYLE_BY_KIND[kind][:3]:
        prompts.append(f"A person {first} {style}, then {second}.")
    return prompts


def build_curated_prompts(seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts: list[str] = []

    for action in ATOMIC_ACTIONS:
        prompts.append(f"A person {action}.")
        prompts.extend(_style_prompts(action))

    for first, second in VALID_TRANSITIONS:
        prompts.extend(_transition_prompts(first, second))

    # A small set of clean two-action compositions. Keep these readable and
    # physically plausible; do not use arbitrary random action pairs.
    compositions = [
        ("walks forward", "raising the right hand", "raises the right hand"),
        ("walks forward", "raising both arms", "raises both arms"),
        ("walks forward", "clapping", "claps"),
        ("walks backward", "waving", "waves"),
        ("jogs in place", "raising both arms", "raises both arms"),
        ("marches in place", "clapping", "claps"),
        ("turns around", "pointing forward", "points forward"),
        ("spins once", "waving", "waves"),
        ("squats down", "raising both arms", "raises both arms"),
        ("stands up", "waving", "waves"),
        ("dances happily", "waving", "waves"),
        ("walks in a circle", "waving", "waves"),
    ]
    for first, simultaneous, second in compositions:
        prompts.append(f"A person {first} while {simultaneous}.")
        prompts.append(f"A person {first}, then {second}.")

    unique = list(dict.fromkeys(prompts))
    rng.shuffle(unique)
    return unique


def split_prompts(prompts: list[str], train_count: int, eval_count: int) -> tuple[list[str], list[str]]:
    if train_count + eval_count > len(prompts):
        raise ValueError(
            f"Requested {train_count} train + {eval_count} eval prompts, but only {len(prompts)} are available."
        )
    return prompts[:train_count], prompts[train_count : train_count + eval_count]


def _write(path: Path, prompts: list[str], seed: int, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Curated ARDY-Mamba text-to-motion distillation prompts.",
        f"# split={split} num_prompts={len(prompts)} seed={seed}",
        *prompts,
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    args = parser.parse_args()

    prompts = build_curated_prompts(args.seed)
    train_prompts, eval_prompts = split_prompts(prompts, args.train_count, args.eval_count)
    _write(Path(args.train_out), train_prompts, args.seed, "train")
    _write(Path(args.eval_out), eval_prompts, args.seed, "eval")
    print(f"Wrote {len(train_prompts)} train prompts to {args.train_out}")
    print(f"Wrote {len(eval_prompts)} eval prompts to {args.eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
