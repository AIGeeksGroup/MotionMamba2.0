"""Two-or-more GPU DDP smoke for the real HumanML3D/control training path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))
sys.path.insert(0, str(REPO / "examples"))
for path in ("/data3/zts/packages/state-spaces-mamba", "/data3/zts/packages/mamba_op_triton"):
    sys.path.insert(0, path)

from ardy_data import HumanML3DArdyDataset  # noqa: E402
from train_humanml3d_ardy_tiny import _build_batch, _loss, _replace_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--text-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["minimal", "hybrid"], required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(700 + rank)

    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model("core", device=str(device), text_encoder=False)
    _replace_stats(model, args.stats_dir)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    client = install_ardy_sequence_kernel(
        model,
        ArdySequenceKernelConfig(mode=args.mode, attention_backend="sdpa", window_size_chunks=4, num_sink_tokens=4),
    )
    raw = model.denoiser.model
    for parameter in list(raw.root_model.parameters()) + list(raw.body_model.parameters()):
        parameter.requires_grad_(True)
    raw.root_model = DDP(raw.root_model, device_ids=[local_rank], broadcast_buffers=False)
    raw.body_model = DDP(raw.body_model, device_ids=[local_rank], broadcast_buffers=False)
    parameters = list(raw.root_model.parameters()) + list(raw.body_model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps, device=device))

    dataset = HumanML3DArdyDataset(
        args.manifest, args.stats_dir, text_root=args.text_root, frames=80, horizon=40,
        condition_dropout=0.1, seed=321,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=99)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    iterator = iter(loader)
    losses = []
    all_finite = True
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            sampler.set_epoch(step + 1)
            iterator = iter(loader)
            batch = next(iterator)
        fixed = _build_batch(model, batch, device, fixed_noise=False)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = _loss(raw, client, fixed)
        loss.backward()
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        finite = bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)
        all_finite = all_finite and finite
        if not finite:
            raise RuntimeError(f"rank {rank}: missing/non-finite gradient")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss))

    tracked = next(raw.root_model.module.parameters()).detach().float()
    checksum = tracked.sum()
    checksums = [torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(checksums, checksum)
    maximum_rank_difference = float(torch.stack(checksums).sub(checksums[0]).abs().max())
    finite_tensor = torch.tensor(int(all_finite), device=device)
    dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
    gathered_losses = [None for _ in range(world)]
    dist.all_gather_object(gathered_losses, losses)
    if rank == 0:
        metrics = {
            "status": "PASS" if finite_tensor.item() == 1 and maximum_rank_difference == 0 else "FAIL",
            "mode": args.mode,
            "world_size": world,
            "steps": args.steps,
            "all_gradients_finite": bool(finite_tensor.item()),
            "parameter_checksum_max_rank_difference": maximum_rank_difference,
            "rank_losses": gathered_losses,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
