"""Canonical Stage-09 G0 fit with the validated 4096-token batch protocol."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
from evaluate_ardy_stage09_canonical import evaluate_candidate, reset_block, summarize  # noqa: E402
from train_ardy_stage09_structured import (  # noqa: E402
    make_wrapper, prepare_history, selected_calls,
)
from train_ardy_mamba_cached import _make_blocks  # noqa: E402
from train_ardy_minimal_cached import TrajectoryDataset, _move  # noqa: E402


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", choices=["H0", "S2"], required=True)
    p.add_argument("--ardy-root", required=True); p.add_argument("--checkpoints-dir", required=True)
    p.add_argument("--minimal-checkpoint", required=True); p.add_argument("--mamba-checkpoint", required=True)
    p.add_argument("--representations", required=True); p.add_argument("--train-manifest", required=True)
    p.add_argument("--dev-manifest", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=500); p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=211); p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def fit(correction, token, state, residual, token_dev, state_dev, residual_dev,
        steps, batch_size, learning_rate, seed):
    correction.train(); optimizer = torch.optim.AdamW(correction.parameters(), lr=learning_rate)
    generator = torch.Generator(device=token.device).manual_seed(seed)
    with torch.no_grad(): initial_dev = float(F.mse_loss(correction(token_dev, state_dev), residual_dev))
    first = None; gradient_finite = True
    for step in range(steps):
        index = torch.randint(len(token), (min(batch_size, len(token)),),
                              generator=generator, device=token.device)
        loss = F.mse_loss(correction(token[index], state[index]), residual[index])
        first = float(loss) if first is None else first
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient_finite = gradient_finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in correction.parameters())
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(json.dumps({"event": "offline_fit", "step": step + 1,
                              "loss": float(loss)}), flush=True)
    correction.eval()
    with torch.no_grad(): final_dev = float(F.mse_loss(correction(token_dev, state_dev), residual_dev))
    return {"initial_batch_mse": first, "initial_dev_mse": initial_dev,
            "final_dev_mse": final_dev, "dev_reduction": 1.0 - final_dev / initial_dev,
            "gradient_finite": gradient_finite, "steps": steps, "batch_size": batch_size}


def root_features(representations, wrapper, device):
    result = {}
    for split in ("train", "development"):
        tensors = representations[split]["root"][3]
        token = tensors["z"].float().to(device)
        state = torch.cat((tensors["state"], tensors["position"]), dim=-1).float().to(device)
        result[split] = (
            (token - wrapper.token_mean.squeeze(0)) / wrapper.token_std.squeeze(0),
            (state - wrapper.state_mean.squeeze(0)) / wrapper.state_std.squeeze(0),
            tensors["residual"].float().to(device),
        )
    return result


@torch.inference_mode()
def collect_body(raw, wrapper, dataset, device):
    buckets = {name: [] for name in ("token", "state", "residual")}
    for index, (sample, chunk, call) in enumerate(selected_calls(dataset)):
        prepare_history(raw, sample, chunk)
        _ = raw(**_move(call["outer_inputs"], device))
        valid = wrapper.last_valid
        buckets["token"].append(wrapper.last_token[valid].cpu())
        buckets["state"].append(wrapper.last_state[valid].cpu())
        target = call["body"]["teacher_output"].to(device).float()
        buckets["residual"].append((target - wrapper.last_attention_output)[valid].cpu())
        if index == 0 or (index + 1) % 96 == 0:
            print(json.dumps({"event": "body_feature_progress", "calls": index + 1}), flush=True)
    return {name: torch.cat(parts).to(device) for name, parts in buckets.items()}


def main():
    a = args(); random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device); torch.cuda.set_device(device)
    sys.path.insert(0, str(Path(a.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model("Ardy-Core-RP-20FPS-Horizon40", device=a.device,
                       text_encoder=False, checkpoints_dir=a.checkpoints_dir)
    install_ardy_sequence_kernel(model, ArdySequenceKernelConfig(
        mode="minimal", attention_backend="sdpa", window_size_chunks=4, num_sink_tokens=4))
    raw = model.denoiser.model
    minimal = torch.load(a.minimal_checkpoint, map_location=device, weights_only=False)
    raw.root_model.load_state_dict(minimal["root_model"], strict=True)
    raw.body_model.load_state_dict(minimal["body_model"], strict=True)
    attention = (raw.root_model, raw.body_model)
    mamba = _make_blocks(raw, 8, device)
    mamba_ckpt = torch.load(a.mamba_checkpoint, map_location=device, weights_only=False)
    mamba[0].load_state_dict(mamba_ckpt["root_model"], strict=True)
    mamba[1].load_state_dict(mamba_ckpt["body_model"], strict=True)
    reps = torch.load(a.representations, map_location="cpu", weights_only=False)
    root_wrapper = make_wrapper(attention[0], mamba[0], "root",
                                attention[0].output_linear.out_features, reps, device)
    body_wrapper = make_wrapper(attention[1], mamba[1], "body",
                                attention[1].output_linear.out_features, reps, device)
    metrics = {"candidate": a.candidate, "seed": a.seed, "final_heldout_used": False}
    if a.candidate == "H0":
        features = root_features(reps, root_wrapper, device)
        metrics["root_fit"] = fit(root_wrapper.correction, *features["train"],
                                   *features["development"], a.steps, a.batch_size,
                                   a.learning_rate, a.seed)
        raw.root_model = root_wrapper
    else:
        raw.root_model = mamba[0]
    raw.body_model = body_wrapper
    for parameter in raw.parameters(): parameter.requires_grad_(False)
    raw.eval()
    train = TrajectoryDataset(a.train_manifest, None); dev = TrajectoryDataset(a.dev_manifest, None)
    body_train = collect_body(raw, body_wrapper, train, device)
    body_dev = collect_body(raw, body_wrapper, dev, device)
    for parameter in body_wrapper.correction.parameters(): parameter.requires_grad_(True)
    metrics["body_fit"] = fit(
        body_wrapper.correction,
        body_train["token"], body_train["state"], body_train["residual"],
        body_dev["token"], body_dev["state"], body_dev["residual"],
        a.steps, a.batch_size, a.learning_rate, a.seed + 1,
    )
    records, runtime = evaluate_candidate(raw, raw.root_model, raw.body_model, dev, device)
    metrics["development"] = summarize(records, 2000, a.seed)
    metrics["runtime"] = runtime
    metrics["trainable_parameters"] = body_wrapper.trainable_parameters + (
        root_wrapper.trainable_parameters if a.candidate == "H0" else 0)
    output = Path(a.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"candidate": a.candidate, "seed": a.seed,
                  "root_correction": root_wrapper.correction_state_dict() if a.candidate == "H0" else None,
                  "body_correction": body_wrapper.correction_state_dict()}
    torch.save(checkpoint, output / "structured_g0.pt")
    torch.save(records, output / "development_records.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
