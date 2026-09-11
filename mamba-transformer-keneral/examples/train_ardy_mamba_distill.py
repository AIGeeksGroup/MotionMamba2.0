"""Milestone C (first version): distill ArdyMambaBlock from the frozen
pretrained ARDY denoiser -- no real training data/loop exists in the public
ARDY release (inference-only), so this bootstraps its own small pseudo
-dataset from the teacher's own generations rather than guessing at ARDY's
undisclosed training recipe.

Design (deliberately simple, per user direction: "we have the pretrain as a
reference, just make sure we don't train it broken"):

1. Use the FROZEN pretrained "core" checkpoint (original transformer
   root_model/body_model) to generate N clean hybrid-motion samples x_0
   (full denoising schedule, best quality) for a diverse set of text
   prompts, via `Ardy._generate_window` directly (single window,
   history_len=0) -- the exact same call the model's own `__call__` makes
   for one window, just intercepted before the multi-window/recentering
   logic. This needs NO real motion-capture data.
2. Swap `model.denoiser.model.root_model` / `.body_model` (the RAW
   denoiser -- see the Milestone B correction note for why `.model` is
   required, not `model.denoiser` directly) for freshly-constructed
   `ArdyMambaBlock` instances. Freeze every other parameter in the model
   (autoencoder, text encoder, embeddings, hybrid-motion projections) --
   only the two Mamba blocks are trainable.
3. Standard diffusion training objective: sample random timestep t over
   the FULL base schedule, noise the cached x_0 via `diffusion.q_sample`,
   run ONE forward pass through the raw (student) denoiser -- not the CFG
   wrapper, no conditioning dropout, this is a plain conditional model for
   now -- and minimize MSE against the pseudo-ground-truth x_0.
4. Log to Weights & Biases.

Known limitations (first version): the pseudo-dataset is small and
teacher-generated (not real motion capture), so this measures "can the
Mamba blocks learn to approximate the teacher's per-step denoising
behavior," not "does this match real motion data" -- a necessary
substitute given no training data exists, not a replacement for it if/when
real data becomes available. No CFG-dropout training (student is a single
conditional model, evaluated without CFG combination for now).

Run with (on the server):

CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data3/zts/cache/huggingface \
MAMBA_TRITON_OVERLAY=/data3/zts/packages/mamba_op_triton \
MAMBA_REPO=/data3/zts/packages/state-spaces-mamba \
PYTHONPATH=/data3/zts/ardy \
/data3/zts/conda_envs/ardy_env/bin/python \
    examples/train_ardy_mamba_distill.py \
    --steps 4000 \
    --prompts-file configs/ardy_mamba_distill_prompts_v2.txt \
    --ckpt-dir /data3/zts/results/ardy_mamba_distill_v2 \
    --wandb-project ardy-mamba-distill
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TRITON_OVERLAY = os.environ.get("MAMBA_TRITON_OVERLAY", "/tmp/mamba_op_triton")
MAMBA_REPO = os.environ.get("MAMBA_REPO", "/home/plote/mambamotion/basecode/mamba/state-spaces-mamba")

PROMPTS = [
    "A person walks forward.",
    "A person walks backward.",
    "A person jumps.",
    "A person waves.",
    "A person squats down.",
    "A person kicks with the right leg.",
    "A person turns around.",
    "A person raises both arms.",
    "A person walks in a circle.",
    "A person sits down.",
    "A person stands up.",
    "A person runs forward.",
    "A person claps their hands.",
    "A person bows.",
    "A person stretches.",
    "A person walks sideways.",
]


def load_prompts(path: str | None, max_prompts: int | None = None) -> list[str]:
    if path is None:
        prompts = list(PROMPTS)
    else:
        prompt_path = Path(path)
        prompts = [line.strip() for line in prompt_path.read_text().splitlines()]
        prompts = [p for p in prompts if p and not p.startswith("#")]
        if not prompts:
            raise ValueError(f"No prompts found in {prompt_path}")
    if max_prompts is not None:
        if max_prompts < 1:
            raise ValueError(f"--max-prompts must be >= 1, got {max_prompts}")
        prompts = prompts[:max_prompts]
    return prompts


def build_pseudo_dataset(
    model,
    device,
    prompts: list[str],
    num_denoising_steps: int | None = None,
    prompt_batch_size: int = 64,
):
    """One clean hybrid-motion x_0 per prompt in PROMPTS, generated with the
    FROZEN original transformer denoiser at (near-)full quality. Returns a
    list of dicts: {x0, text_feat, text_pad_mask, first_heading_angle}."""
    from ardy.model.ardy_model import get_three_mask_from_len  # noqa: F401 (parity import, unused here)

    num_denoising_steps = num_denoising_steps or int(model.diffusion.num_base_steps)
    gen_horizon_len = model.gen_horizon_len
    num_denoising_steps_t = torch.tensor([num_denoising_steps], device=device)
    use_timesteps = model.diffusion.space_timesteps(num_denoising_steps_t[0])[0]
    model.diffusion.calc_diffusion_vars(use_timesteps)
    indices = list(range(num_denoising_steps))[::-1]

    chunks = []
    if prompt_batch_size < 1:
        raise ValueError(f"prompt_batch_size must be >= 1, got {prompt_batch_size}")

    for start in range(0, len(prompts), prompt_batch_size):
        prompt_chunk = prompts[start : start + prompt_batch_size]
        batch_size = len(prompt_chunk)
        print(f"  teacher chunk {start // prompt_batch_size + 1}: prompts {start}-{start + batch_size - 1}", flush=True)
        text_feat, text_pad_mask = model._encode_text(prompt_chunk)
        first_heading_angle = torch.zeros(batch_size, device=device)
        global_transl = torch.zeros((batch_size, model.motion_rep.nfeats_dict["root_pos"]), device=device)
        with torch.no_grad():
            x0 = model._generate_window(
                None,
                global_transl,
                0,
                0,
                gen_horizon_len,
                text_feat,
                text_pad_mask,
                first_heading_angle,
                None,
                None,
                num_denoising_steps_t,
                (2.0, 2.0),
                indices,
            )
        chunks.append(
            {
                "x0": x0.detach(),
                "text_feat": text_feat.detach(),
                "text_pad_mask": text_pad_mask.detach(),
                "first_heading_angle": first_heading_angle.detach(),
            }
        )

    return {
        "x0": torch.cat([chunk["x0"] for chunk in chunks], dim=0),
        "text_feat": torch.cat([chunk["text_feat"] for chunk in chunks], dim=0),
        "text_pad_mask": torch.cat([chunk["text_pad_mask"] for chunk in chunks], dim=0),
        "first_heading_angle": torch.cat([chunk["first_heading_angle"] for chunk in chunks], dim=0),
    }


def training_step(raw_denoiser, diffusion, batch, idx, gen_horizon_len, device):
    from ardy.model.ardy_model import get_three_mask_from_len

    x0_full = batch["x0"][idx]
    text_feat = batch["text_feat"][idx]
    text_pad_mask = batch["text_pad_mask"][idx]
    first_heading_angle = batch["first_heading_angle"][idx]

    B, num_gen_tokens, _ = x0_full.shape
    history_len = torch.zeros(B, dtype=torch.long, device=device)
    generation_len = torch.full((B,), gen_horizon_len, dtype=torch.long, device=device)
    future_len = torch.zeros(B, dtype=torch.long, device=device)
    history_mask, generation_mask, future_mask = get_three_mask_from_len(
        history_len, generation_len, future_len, gen_horizon_len, device
    )
    history_token_mask = torch.zeros(B, num_gen_tokens, dtype=torch.bool, device=device)
    generation_token_mask = torch.ones(B, num_gen_tokens, dtype=torch.bool, device=device)
    future_token_mask = torch.zeros(B, num_gen_tokens, dtype=torch.bool, device=device)

    t = torch.randint(0, diffusion.num_base_steps, (B,), device=device)
    noise = torch.randn_like(x0_full)
    x_t = diffusion.q_sample(x0_full, t, noise)

    pred_x0 = raw_denoiser(
        x_t,
        history_len,
        generation_len,
        future_len,
        history_mask,
        generation_mask,
        future_mask,
        history_token_mask,
        generation_token_mask,
        future_token_mask,
        text_feat,
        text_pad_mask,
        t,
        first_heading_angle,
        None,
        None,
    )
    loss = F.mse_loss(pred_x0, x0_full)
    return loss


@torch.no_grad()
def eval_loss(raw_denoiser, diffusion, batch, gen_horizon_len, device, batch_size: int = 8) -> float:
    losses = []
    n_samples = batch["x0"].shape[0]
    for start in range(0, n_samples, batch_size):
        idx = torch.arange(start, min(start + batch_size, n_samples), device=device)
        loss = training_step(raw_denoiser, diffusion, batch, idx, gen_horizon_len, device)
        losses.append(loss.detach())
    return float(torch.stack(losses).mean().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--ckpt-dir", default="/data3/zts/results/ardy_mamba_distill")
    parser.add_argument("--wandb-project", default="ardy-mamba-distill")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--dataset-denoising-steps", type=int, default=None)
    parser.add_argument("--dataset-prompt-batch-size", type=int, default=64)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--eval-prompts-file", default=None)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        return 2
    torch.empty(1, device="cuda")

    sys.path.insert(0, TRITON_OVERLAY)
    sys.path.insert(0, MAMBA_REPO)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    import wandb
    from ardy.model import DEFAULT_MODEL, load_model
    from ardy.model.loading import get_env_var
    from ardy.model.registry import resolve_model_name
    from ardy.tools import seed_everything

    from ardy_adapter import ArdyMambaBlock

    device = "cuda:0"
    seed_everything(args.seed)
    prompts = load_prompts(args.prompts_file, args.max_prompts)
    eval_prompts = load_prompts(args.eval_prompts_file, None) if args.eval_prompts_file else []
    checkpoints_dir = get_env_var("CHECKPOINTS_DIR")
    resolved_model = resolve_model_name(DEFAULT_MODEL, checkpoints_dir=checkpoints_dir)
    print(f"Loading {resolved_model} ...", flush=True)
    model = load_model(resolved_model, device=device, checkpoints_dir=checkpoints_dir)
    print("Loaded.", flush=True)

    print(f"Bootstrapping pseudo-dataset from the frozen teacher ({len(prompts)} prompts, "
          f"{args.dataset_denoising_steps or int(model.diffusion.num_base_steps)} denoising steps)...", flush=True)
    dataset = build_pseudo_dataset(
        model,
        device,
        prompts,
        args.dataset_denoising_steps,
        prompt_batch_size=args.dataset_prompt_batch_size,
    )
    print(f"x0 shape: {tuple(dataset['x0'].shape)}", flush=True)
    eval_dataset = None
    if eval_prompts:
        print(
            f"Bootstrapping heldout pseudo-dataset from the frozen teacher ({len(eval_prompts)} prompts, "
            f"{args.dataset_denoising_steps or int(model.diffusion.num_base_steps)} denoising steps)...",
            flush=True,
        )
        eval_dataset = build_pseudo_dataset(
            model,
            device,
            eval_prompts,
            args.dataset_denoising_steps,
            prompt_batch_size=args.dataset_prompt_batch_size,
        )
        print(f"eval x0 shape: {tuple(eval_dataset['x0'].shape)}", flush=True)

    # -- swap in ArdyMambaBlock, freeze everything else --
    raw_denoiser = model.denoiser.model  # see Milestone B correction: NOT model.denoiser directly
    for p in model.parameters():
        p.requires_grad_(False)

    common_kwargs = dict(
        input_dim=-1,
        add_input_proj=False,
        skeleton=raw_denoiser.motion_rep.skeleton,
        llm_shape=raw_denoiser.llm_shape,
        use_text_mask=raw_denoiser.root_model.use_text_mask,
        latent_dim=raw_denoiser.latent_dim,
        num_layers=8,
        input_first_heading_angle=hasattr(raw_denoiser.root_model, "linear_first_heading_angle"),
        positional_encoding_mode=raw_denoiser.positional_encoding_mode,
        device=device,
    )
    mamba_root = ArdyMambaBlock(output_dim=raw_denoiser.root_model.output_linear.out_features, **common_kwargs).to(device)
    mamba_body = ArdyMambaBlock(output_dim=raw_denoiser.body_model.output_linear.out_features, **common_kwargs).to(device)
    raw_denoiser.root_model = mamba_root
    raw_denoiser.body_model = mamba_body
    assert model.denoiser.model.root_model is mamba_root
    assert model.denoiser.model.body_model is mamba_body
    for p in mamba_root.parameters():
        p.requires_grad_(True)
    for p in mamba_body.parameters():
        p.requires_grad_(True)

    trainable_params = list(mamba_root.parameters()) + list(mamba_body.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} total ({100 * n_trainable / n_total:.2f}%)", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # training always uses the FULL base schedule (bootstrap generation may
    # have left the diffusion buffers on a subsampled schedule)
    model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps, device=device))

    wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        config={
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_layers": 8,
            "num_prompts": len(prompts),
            "prompts_file": args.prompts_file,
            "eval_prompts_file": args.eval_prompts_file,
            "num_eval_prompts": len(eval_prompts),
            "seed": args.seed,
            "dataset_prompt_batch_size": args.dataset_prompt_batch_size,
            "trainable_params": n_trainable,
            "model": resolved_model,
        },
    )

    n_samples = dataset["x0"].shape[0]
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    mamba_root.train()
    mamba_body.train()
    for step in range(args.steps):
        idx = torch.randint(0, n_samples, (args.batch_size,), device=device)
        loss = training_step(raw_denoiser, model.diffusion, dataset, idx, model.gen_horizon_len, device)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"step {step}: loss={loss.item():.6f} grad_norm={grad_norm.item():.4f}", flush=True)
            wandb.log({"loss": loss.item(), "grad_norm": grad_norm.item(), "step": step})

        if eval_dataset is not None and args.eval_every and (step % args.eval_every == 0 or step == args.steps - 1):
            mamba_root.eval()
            mamba_body.eval()
            val_loss = eval_loss(
                raw_denoiser,
                model.diffusion,
                eval_dataset,
                model.gen_horizon_len,
                device,
                batch_size=args.eval_batch_size,
            )
            mamba_root.train()
            mamba_body.train()
            print(f"step {step}: val_loss={val_loss:.6f}", flush=True)
            wandb.log({"val_loss": val_loss, "step": step})

        if args.ckpt_every and (step % args.ckpt_every == 0 and step > 0 or step == args.steps - 1):
            ckpt_path = Path(args.ckpt_dir) / f"mamba_blocks_step{step}.pt"
            torch.save(
                {
                    "root_model": mamba_root.state_dict(),
                    "body_model": mamba_body.state_dict(),
                    "step": step,
                    "prompts": prompts,
                    "eval_prompts": eval_prompts,
                    "prompts_file": args.prompts_file,
                    "eval_prompts_file": args.eval_prompts_file,
                    "seed": args.seed,
                    "dataset_denoising_steps": args.dataset_denoising_steps,
                    "dataset_prompt_batch_size": args.dataset_prompt_batch_size,
                    "model": resolved_model,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}", flush=True)

    wandb.finish()
    print("TRAINING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
