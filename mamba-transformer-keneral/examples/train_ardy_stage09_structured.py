"""Train the exact frozen-parent Stage-09 H0 or S2 G0 corrections."""

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
from evaluate_ardy_stage09_canonical import (  # noqa: E402
    commit_attention_history,
    evaluate_candidate,
    reset_block,
    summarize,
)
from train_ardy_mamba_cached import _make_blocks  # noqa: E402
from train_ardy_minimal_cached import TrajectoryDataset, _move  # noqa: E402
from ardy_adapter.stage09_structured import Stage09G0Block  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", choices=["H0", "S2"], required=True)
    p.add_argument("--ardy-root", required=True); p.add_argument("--checkpoints-dir", required=True)
    p.add_argument("--minimal-checkpoint", required=True); p.add_argument("--mamba-checkpoint", required=True)
    p.add_argument("--representations", required=True)
    p.add_argument("--train-manifest", required=True); p.add_argument("--dev-manifest", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--steps", type=int, default=500)
    p.add_argument("--tiny-only", action="store_true"); p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=211); p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def stats(representations, stage, source_name):
    tensors = representations["train"][stage][3]
    token = tensors["z"].float()
    state = tensors["mamba_hidden"].float() if source_name == "hidden" else torch.cat(
        (tensors["state"].float(), tensors["position"].float()), dim=-1)
    return token.mean(0), token.std(0).clamp_min(1e-4), state.mean(0), state.std(0).clamp_min(1e-4)


def make_wrapper(attention, mamba, stage, output_dim, representations, device):
    source_name = "recurrent" if stage == "root" else "hidden"
    token_mean, token_std, state_mean, state_std = stats(representations, stage, source_name)
    return Stage09G0Block(
        attention, mamba, layer=3, state_source=source_name, output_dim=output_dim,
        token_mean=token_mean, token_std=token_std,
        state_mean=state_mean, state_std=state_std, zero_output=True,
    ).to(device)


def selected_calls(dataset):
    result = []; steps = dataset.manifest["diffusion_steps"]
    selected = {0, steps // 2, steps - 1}
    for sample in dataset.samples:
        for index, call in enumerate(sample["calls"]):
            if index % steps in selected:
                result.append((sample, index // steps, call))
    return result


def prepare_history(raw, sample, chunk):
    reset_block(raw.root_model); reset_block(raw.body_model)
    with torch.no_grad():
        commit_attention_history(raw.root_model, sample, chunk, "root", raw.num_frames_per_token)
        commit_attention_history(raw.body_model, sample, chunk, "body", raw.num_frames_per_token)


def loss_for_call(raw, call, device):
    _ = raw(**_move(call["outer_inputs"], device))
    root_prediction = raw.root_model.last_output if isinstance(raw.root_model, Stage09G0Block) else None
    body_prediction = raw.body_model.last_output if isinstance(raw.body_model, Stage09G0Block) else None
    if root_prediction is None:
        # S2 uses the direct Mamba root. Capture its output through a one-shot hook.
        raise RuntimeError("direct root output capture must be installed by caller")
    root_target = call["root"]["teacher_output"].to(device).float()
    body_target = call["body"]["teacher_output"].to(device).float()
    return F.mse_loss(root_prediction.float(), root_target) + F.mse_loss(body_prediction.float(), body_target)


def main():
    args = parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); torch.cuda.set_device(device)
    sys.path.insert(0, str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model("Ardy-Core-RP-20FPS-Horizon40", device=args.device,
                       text_encoder=False, checkpoints_dir=args.checkpoints_dir)
    install_ardy_sequence_kernel(model, ArdySequenceKernelConfig(
        mode="minimal", attention_backend="sdpa", window_size_chunks=4, num_sink_tokens=4))
    raw = model.denoiser.model
    minimal = torch.load(args.minimal_checkpoint, map_location=device, weights_only=False)
    raw.root_model.load_state_dict(minimal["root_model"], strict=True)
    raw.body_model.load_state_dict(minimal["body_model"], strict=True)
    attention = (raw.root_model, raw.body_model)
    mamba = _make_blocks(raw, 8, device)
    mamba_checkpoint = torch.load(args.mamba_checkpoint, map_location=device, weights_only=False)
    mamba[0].load_state_dict(mamba_checkpoint["root_model"], strict=True)
    mamba[1].load_state_dict(mamba_checkpoint["body_model"], strict=True)
    representations = torch.load(args.representations, map_location="cpu", weights_only=False)
    root_wrapper = make_wrapper(attention[0], mamba[0], "root",
                                attention[0].output_linear.out_features, representations, device)
    body_wrapper = make_wrapper(attention[1], mamba[1], "body",
                                attention[1].output_linear.out_features, representations, device)
    if args.candidate == "H0":
        raw.root_model, raw.body_model = root_wrapper, body_wrapper
        trainable = list(root_wrapper.correction.parameters()) + list(body_wrapper.correction.parameters())
    else:
        raw.root_model, raw.body_model = mamba[0], body_wrapper
        trainable = list(body_wrapper.correction.parameters())
    for parameter in raw.parameters(): parameter.requires_grad_(False)
    for parameter in trainable: parameter.requires_grad_(True)
    raw.eval()

    captured_root = {"value": None}
    root_handle = raw.root_model.register_forward_hook(
        lambda _m, _i, output: captured_root.__setitem__("value", output)
    )

    train = TrajectoryDataset(args.train_manifest, None); dev = TrajectoryDataset(args.dev_manifest, None)
    calls = selected_calls(train); fixed = calls[0]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    generator = random.Random(args.seed)
    initial = None; final = None; initial_body = None; final_body = None; gradient_finite = True
    steps = 500 if args.tiny_only else args.steps
    for step in range(steps):
        sample, chunk, call = fixed if args.tiny_only else calls[generator.randrange(len(calls))]
        prepare_history(raw, sample, chunk); captured_root["value"] = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = raw(**_move(call["outer_inputs"], device))
            root_prediction = captured_root["value"]
            body_prediction = body_wrapper.last_output
            root_loss = F.mse_loss(root_prediction.float(), call["root"]["teacher_output"].to(device).float())
            body_loss = F.mse_loss(body_prediction.float(), call["body"]["teacher_output"].to(device).float())
            loss = root_loss + body_loss
        initial = float(loss) if initial is None else initial
        initial_body = float(body_loss) if initial_body is None else initial_body
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient_finite = gradient_finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in trainable)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
        final = float(loss); final_body = float(body_loss)
        if step == 0 or (step + 1) % 50 == 0:
            print(json.dumps({"event": "train", "candidate": args.candidate,
                              "seed": args.seed, "step": step + 1, "loss": final}), flush=True)

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "candidate": args.candidate, "seed": args.seed,
        "root_correction": root_wrapper.correction_state_dict() if args.candidate == "H0" else None,
        "body_correction": body_wrapper.correction_state_dict(),
    }
    torch.save(checkpoint, output / "structured_g0.pt")
    # Save/reload is checked on the exact trained modules.
    reloaded = torch.load(output / "structured_g0.pt", map_location=device, weights_only=False)
    if args.candidate == "H0": root_wrapper.load_correction_state_dict(reloaded["root_correction"])
    body_wrapper.load_correction_state_dict(reloaded["body_correction"])
    train_metrics = {"initial_loss": initial, "final_sampled_loss": final,
                     "gradient_finite": gradient_finite,
                     "sampled_reduction": 1.0 - final / initial,
                     "initial_body_loss": initial_body, "final_body_loss": final_body,
                     "body_reduction": 1.0 - final_body / initial_body,
                     "body_correction_parameter_norm": float(torch.sqrt(sum(
                         parameter.detach().float().square().sum() for parameter in body_wrapper.correction.parameters()))),
                     "steps": steps, "tiny_only": args.tiny_only}
    if not args.tiny_only:
        records, runtime = evaluate_candidate(raw, raw.root_model, raw.body_model, dev, device)
        dev_metrics = summarize(records, 2000, args.seed)
        torch.save(records, output / "development_records.pt")
    else:
        runtime = {}; dev_metrics = None
    result = {"candidate": args.candidate, "seed": args.seed,
              "final_heldout_used": False, "train": train_metrics,
              "development": dev_metrics, "runtime": runtime,
              "trainable_parameters": sum(parameter.numel() for parameter in trainable)}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    root_handle.remove(); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
