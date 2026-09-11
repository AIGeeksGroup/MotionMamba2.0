"""Stage-07 Minimal/Mamba error complementarity on development data only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from train_ardy_mamba_cached import _make_blocks  # noqa: E402
from train_ardy_minimal_cached import (  # noqa: E402
    TrajectoryDataset,
    _commit_history,
    _move,
    _reset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--minimal-checkpoint", required=True)
    parser.add_argument("--mamba-checkpoint", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-trajectories", type=int, default=4)
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _flatten(root: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
    return torch.cat((root.flatten(), body.flatten()))


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    return float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-12))


def _cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cosine_similarity(x.double(), y.double(), dim=0))


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.mse_loss(prediction.float(), target.float()))


def _fit_scalar(records: list[dict[str, Any]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        direction = record["minimal"] - record["mamba"]
        residual = record["target"] - record["mamba"]
        numerator += float(torch.dot(direction.double(), residual.double()))
        denominator += float(torch.dot(direction.double(), direction.double()))
    return max(0.0, min(1.0, numerator / max(denominator, 1e-12)))


def _fit_affine(records: list[dict[str, Any]]) -> torch.Tensor:
    # One global affine fusion, not a per-output lookup table.
    # Accumulate the tiny normal equation in float64. A direct tall float32
    # lstsq was ill-conditioned for millions of motion elements and silently
    # returned near-zero weights plus a bad bias in the first Stage-08 run.
    gram = torch.zeros(3, 3, dtype=torch.float64)
    rhs = torch.zeros(3, dtype=torch.float64)
    for record in records:
        minimal = record["minimal"].double()
        mamba = record["mamba"].double()
        target = record["target"].double()
        design = torch.stack((minimal, mamba, torch.ones_like(minimal)), dim=1)
        gram += design.T @ design
        rhs += design.T @ target
    return torch.linalg.solve(gram, rhs).float()


class ScalarGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, minimal: torch.Tensor, mamba: torch.Tensor) -> torch.Tensor:
        features = torch.stack(
            (minimal, mamba, minimal - mamba, (minimal - mamba).abs()), dim=-1
        )
        gate = torch.sigmoid(self.net(features).squeeze(-1))
        return gate * minimal + (1.0 - gate) * mamba


def _fit_gate(records: list[dict[str, Any]], seed: int) -> ScalarGate:
    torch.manual_seed(seed)
    gate = ScalarGate()
    optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-3)
    minimal = torch.cat([record["minimal"] for record in records]).float()
    mamba = torch.cat([record["mamba"] for record in records]).float()
    target = torch.cat([record["target"] for record in records]).float()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(300):
        index = torch.randint(0, len(target), (min(32768, len(target)),), generator=generator)
        loss = F.mse_loss(gate(minimal[index], mamba[index]), target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return gate.eval()


def _evaluate(records, scalar, affine, gate) -> dict[str, Any]:
    policies: dict[str, list[float]] = {
        name: [] for name in
        ("minimal", "mamba", "average", "scalar", "affine", "gate", "oracle")
    }
    correlations = []
    cosines = []
    minimal_wins = 0
    mamba_wins = 0
    oracle_alphas = []
    for record in records:
        minimal, mamba, target = record["minimal"], record["mamba"], record["target"]
        error_a = target - minimal
        error_m = target - mamba
        correlations.append(_pearson(error_a, error_m))
        cosines.append(_cosine(error_a, error_m))
        minimal_loss = _mse(minimal, target)
        mamba_loss = _mse(mamba, target)
        minimal_wins += minimal_loss < mamba_loss
        mamba_wins += mamba_loss < minimal_loss
        direction = minimal - mamba
        alpha = float(
            torch.dot(direction.double(), (target - mamba).double())
            / torch.dot(direction.double(), direction.double()).clamp_min(1e-12)
        )
        alpha = max(0.0, min(1.0, alpha))
        oracle_alphas.append(alpha)
        with torch.no_grad():
            gated_prediction = gate(minimal.float(), mamba.float()).detach()
        predictions = {
            "minimal": minimal,
            "mamba": mamba,
            "average": 0.5 * (minimal + mamba),
            "scalar": scalar * minimal + (1.0 - scalar) * mamba,
            "affine": affine[0] * minimal + affine[1] * mamba + affine[2],
            "gate": gated_prediction,
            "oracle": alpha * minimal + (1.0 - alpha) * mamba,
        }
        for name, prediction in predictions.items():
            policies[name].append(_mse(prediction, target))
    mean = {name: sum(values) / len(values) for name, values in policies.items()}
    baseline = mean["minimal"]
    return {
        "cases": len(records),
        "mean_mse": mean,
        "relative_gain_vs_minimal": {
            name: (baseline - value) / baseline for name, value in mean.items()
        },
        "mean_error_pearson": sum(correlations) / len(correlations),
        "mean_error_cosine": sum(cosines) / len(cosines),
        "minimal_win_rate": minimal_wins / len(records),
        "mamba_win_rate": mamba_wins / len(records),
        "tie_rate": 1.0 - (minimal_wins + mamba_wins) / len(records),
        "mean_oracle_alpha_minimal": sum(oracle_alphas) / len(oracle_alphas),
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sys.path.insert(0, str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )
    install_ardy_sequence_kernel(
        model,
        ArdySequenceKernelConfig(
            mode="minimal", attention_backend="sdpa", window_size_chunks=4,
            num_sink_tokens=4,
        ),
    )
    raw = model.denoiser.model
    minimal_checkpoint = torch.load(
        args.minimal_checkpoint, map_location=device, weights_only=False
    )
    raw.root_model.load_state_dict(minimal_checkpoint["root_model"], strict=True)
    raw.body_model.load_state_dict(minimal_checkpoint["body_model"], strict=True)
    raw.eval()
    mamba_root, mamba_body = _make_blocks(raw, 8, device)
    mamba_checkpoint = torch.load(
        args.mamba_checkpoint, map_location=device, weights_only=False
    )
    mamba_root.load_state_dict(mamba_checkpoint["root_model"], strict=True)
    mamba_body.load_state_dict(mamba_checkpoint["body_model"], strict=True)
    mamba_root.eval()
    mamba_body.eval()
    dataset = TrajectoryDataset(args.dev_manifest, None)
    if not 0 < args.calibration_trajectories < len(dataset):
        raise ValueError("calibration split must leave development evaluation trajectories")
    records = []
    diffusion_steps = dataset.manifest["diffusion_steps"]
    selected = {0, diffusion_steps // 2, diffusion_steps - 1}
    with torch.inference_mode():
        for trajectory, sample in enumerate(dataset.samples):
            for call_index, call in enumerate(sample["calls"]):
                if call_index % diffusion_steps not in selected:
                    continue
                chunk = call_index // diffusion_steps
                _reset(raw)
                _commit_history(raw, sample, chunk)
                root_inputs = _move(call["root"]["inputs"], device)
                body_inputs = _move(call["body"]["inputs"], device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    minimal_root = raw.root_model(*root_inputs)
                    minimal_body = raw.body_model(*body_inputs)
                    mamba_root_prediction = mamba_root(*root_inputs)
                    mamba_body_prediction = mamba_body(*body_inputs)
                records.append(
                    {
                        "trajectory": trajectory,
                        "chunk": chunk,
                        "diffusion_index": call_index % diffusion_steps,
                        "minimal": _flatten(minimal_root, minimal_body).float().cpu(),
                        "mamba": _flatten(mamba_root_prediction, mamba_body_prediction).float().cpu(),
                        "target": _flatten(
                            call["root"]["teacher_output"], call["body"]["teacher_output"]
                        ).float().cpu(),
                    }
                )
    _reset(raw)
    calibration = [r for r in records if r["trajectory"] < args.calibration_trajectories]
    evaluation = [r for r in records if r["trajectory"] >= args.calibration_trajectories]
    scalar = _fit_scalar(calibration)
    affine = _fit_affine(calibration)
    gate = _fit_gate(calibration, args.seed)
    result = {
        "phase": "stage07_complementarity_development_only",
        "final_heldout_used": False,
        "calibration_trajectories": args.calibration_trajectories,
        "evaluation_trajectories": len(dataset) - args.calibration_trajectories,
        "fitted_scalar_weight_on_minimal": scalar,
        "fitted_affine": affine.tolist(),
        "calibration": _evaluate(calibration, scalar, affine, gate),
        "development_evaluation": _evaluate(evaluation, scalar, affine, gate),
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "complementarity.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"records": records, "result": result}, output / "predictions_dev.pt")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
