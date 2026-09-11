"""Offline local LLM2Vec smoke with a multiprocessing-safe entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ARDY_ROOT = Path(__file__).resolve().parents[2] / "ardy"
sys.path.insert(0, str(ARDY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="a person walks forward")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    from ardy.model.load_model import load_text_encoder

    encoder = load_text_encoder(mode="local", device=args.device)
    captions = [args.prompt]
    with torch.inference_mode():
        features, lengths = encoder(captions)
    metrics = {
        "shape": list(features.shape),
        "lengths": lengths,
        "dtype": str(features.dtype),
        "finite": bool(torch.isfinite(features).all()),
        "device": str(features.device),
    }
    print(json.dumps(metrics, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        text_pad_mask = torch.ones(features.shape[:2], dtype=torch.bool)
        torch.save(
            {
                "prompt": args.prompt,
                "text_feat": features.detach().cpu(),
                "text_feat_pad_mask": text_pad_mask,
            },
            output,
        )
    return 0 if metrics["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
