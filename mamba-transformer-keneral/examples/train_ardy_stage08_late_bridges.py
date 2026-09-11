"""Bounded frozen-parent G0/I1/I2 pilots for Stage 08.

This consumes TRAIN/DEVELOPMENT representations only.  It is deliberately
not an architecture integration: the purpose is to decide the Stage-08 hard
gate before modifying either frozen parent.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from probe_ardy_stage08_innovation import MLP, fit, normalize, source


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--representations", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--tiny-steps", type=int, default=500)
    p.add_argument("--batch-sequences", type=int, default=64)
    p.add_argument("--seed", type=int, default=211)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


class G0(nn.Module):
    def __init__(self, dz, ds, dout, width=64):
        super().__init__()
        self.net = MLP(dz + ds, dout, width)

    def forward(self, z, state, zhat, delta, mask=None):
        return self.net(torch.cat((z, state), dim=-1))


class I1(nn.Module):
    def __init__(self, dim, dout, width=64):
        super().__init__()
        self.delta = nn.Linear(dim, width)
        self.gate = nn.Linear(2 * dim, dout)
        self.out = nn.Linear(width, dout)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, z, state, zhat, delta, mask=None):
        correction = self.out(F.silu(self.delta(delta)))
        gate = torch.sigmoid(self.gate(torch.cat((zhat, delta), dim=-1)))
        return self.alpha * gate * correction


class I2(nn.Module):
    def __init__(self, dim, dout, width=64, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.out = nn.Linear(width, dout)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, z, state, zhat, delta, mask=None):
        tokens = self.in_proj(self.norm(delta))
        mixed, _ = self.attention(tokens, tokens, tokens,
                                  key_padding_mask=None if mask is None else ~mask,
                                  need_weights=False)
        return self.alpha * self.out(mixed)


def pad_sequences(values, sequence):
    sequence = sequence.to(values.device).long()
    unique, counts = torch.unique_consecutive(sequence, return_counts=True)
    if int(unique[-1] - unique[0] + 1) != len(unique):
        raise RuntimeError("sequence ids must be consecutive")
    length = int(counts.max())
    padded = values.new_zeros((len(unique), length, values.shape[-1]))
    mask = torch.zeros((len(unique), length), dtype=torch.bool, device=values.device)
    offset = 0
    for row, count_tensor in enumerate(counts):
        count = int(count_tensor); padded[row, :count] = values[offset:offset + count]
        mask[row, :count] = True; offset += count
    return padded, mask


def standardize_pair(train, development):
    train, mean, std = normalize(train)
    development, _, _ = normalize(development, mean, std)
    return train, development


def prepare(data, stage, layer, state_name, predictor, state_mean, state_std, device):
    result = {}
    for split in ("train", "development"):
        tensors = data[split][stage][layer]
        state = source(tensors, state_name).float().to(device)
        state = (state - state_mean) / state_std
        z = tensors["z"].float().to(device)
        with torch.no_grad():
            zhat = predictor(state)
            delta = F.layer_norm(z, z.shape[-1:]) - F.layer_norm(zhat, zhat.shape[-1:])
        result[split] = {"z": z, "state": state, "zhat": zhat,
                         "delta": delta, "residual": tensors["residual"].float().to(device),
                         "sequence": tensors["sequence"].long().to(device)}
    for name in ("z", "state", "zhat", "delta"):
        result["train"][name], result["development"][name] = standardize_pair(
            result["train"][name], result["development"][name])
    for split in ("train", "development"):
        mask = None
        for name in ("z", "state", "zhat", "delta", "residual"):
            value, current_mask = pad_sequences(result[split][name], result[split]["sequence"])
            if mask is not None and not torch.equal(mask, current_mask):
                raise RuntimeError("padded sequence masks differ across features")
            result[split][name] = value; mask = current_mask
        result[split]["mask"] = mask; del result[split]["sequence"]
    return result, int(result["train"]["mask"].shape[1])


def bridge_loss(model, batch):
    correction = model(batch["z"], batch["state"], batch["zhat"], batch["delta"], batch["mask"])
    return F.mse_loss(correction[batch["mask"]], batch["residual"][batch["mask"]])


def train_bridge(model, train, development, steps, batch_sequences, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=train["z"].device).manual_seed(seed)
    initial = None
    for _ in range(steps):
        idx = torch.randint(len(train["z"]), (min(batch_sequences, len(train["z"])),),
                            generator=generator, device=train["z"].device)
        batch = {name: value[idx] for name, value in train.items()}
        loss = bridge_loss(model, batch)
        initial = float(loss) if initial is None else initial
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        dev = float(bridge_loss(model, development))
    return {"initial_train_mse": initial, "development_mse": dev,
            "parameters": sum(p.numel() for p in model.parameters()),
            "alpha": float(model.alpha) if hasattr(model, "alpha") else None}


def main():
    args = parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); torch.cuda.set_device(device)
    data = torch.load(args.representations, map_location="cpu", weights_only=False)
    selected = {"root": (3, "recurrent"), "body": (3, "hidden")}
    results = {"final_heldout_used": False, "selected": {}, "stages": {}}
    combined = {name: {"squared": 0.0, "elements": 0} for name in ("Minimal", "G0", "I1", "I2")}
    for stage, (layer, state_name) in selected.items():
        tr = data["train"][stage][layer]; dv = data["development"][stage][layer]
        state = source(tr, state_name).float().to(device); stated = source(dv, state_name).float().to(device)
        state, state_mean, state_std = normalize(state); stated, _, _ = normalize(stated, state_mean, state_std)
        z = tr["z"].float().to(device); zd = dv["z"].float().to(device)
        predictor = MLP(state.shape[-1], z.shape[-1], 256).to(device)
        predictor_metric, _ = fit(predictor, state, z, stated, zd, args.steps, 4096, args.seed)
        predictor.eval()
        prepared, length = prepare(data, stage, layer, state_name, predictor,
                                   state_mean.to(device), state_std.to(device), device)
        dout = prepared["train"]["residual"].shape[-1]
        constructors = {
            "G0": lambda: G0(z.shape[-1], state.shape[-1], dout).to(device),
            "I1": lambda: I1(z.shape[-1], dout).to(device),
            "I2": lambda: I2(z.shape[-1], dout).to(device),
        }
        stage_result = {"sequence_length": length, "predictor": predictor_metric,
                        "minimal_development_mse": float(prepared["development"]["residual"][prepared["development"]["mask"]].square().mean()),
                        "bridges": {}}
        fixed = {name: value[:1] for name, value in prepared["train"].items()}
        for index, (name, constructor) in enumerate(constructors.items()):
            torch.manual_seed(args.seed + index)
            initialization_model = constructor().eval()
            with torch.no_grad():
                initialization_error = float(initialization_model(
                    fixed["z"], fixed["state"], fixed["zhat"], fixed["delta"], fixed["mask"]).abs().max())
            tiny_model = constructor()
            tiny_before = float(bridge_loss(tiny_model, fixed))
            tiny_metric = train_bridge(tiny_model, fixed, fixed, args.tiny_steps, 1, args.seed + index)
            tiny_after = tiny_metric["development_mse"]
            model = constructor()
            metric = train_bridge(model, prepared["train"], prepared["development"],
                                  args.steps, args.batch_sequences, args.seed + index)
            metric.update({"initialization_max_abs_correction": initialization_error,
                           "tiny_initial_mse": tiny_before, "tiny_final_mse": tiny_after,
                           "tiny_reduction": 1.0 - tiny_after / tiny_before})
            stage_result["bridges"][name] = metric
            print(f"bridge {stage}/{name}: tiny_reduction={metric['tiny_reduction']:.3%} "
                  f"dev_mse={metric['development_mse']:.8f}", flush=True)
        results["selected"][stage] = {"layer": layer, "state_source": state_name}
        results["stages"][stage] = stage_result
        elements = int(prepared["development"]["mask"].sum()) * dout
        combined["Minimal"]["squared"] += stage_result["minimal_development_mse"] * elements
        combined["Minimal"]["elements"] += elements
        for name in constructors:
            combined[name]["squared"] += stage_result["bridges"][name]["development_mse"] * elements
            combined[name]["elements"] += elements
    results["combined_development_mse"] = {
        name: values["squared"] / values["elements"] for name, values in combined.items()
    }
    baseline = results["combined_development_mse"]["Minimal"]
    results["combined_relative_gain"] = {
        name: 1.0 - mse / baseline for name, mse in results["combined_development_mse"].items()
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "late_bridge_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
