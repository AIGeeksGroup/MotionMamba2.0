"""9.26-revise4 predicted-history bridge.

The bridge is deliberately isolated from the default trainer.  It uses the
same denoiser, codec, accepted-history commit and generated-supervision target
builder as production training, but replaces only the W1 source at the
post-generation boundary with a detached one-step clean-latent prediction.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import torch


def _nearest_timestep(diffusion, fraction: float) -> tuple[int, float]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("prediction-history noise fraction must be in (0,1)")
    alpha = diffusion.alphas_cumprod_base
    actual, index = torch.min((1.0 - alpha - fraction).abs(), dim=0)
    return int(index.item()), float((1.0 - alpha[index]).item())


@torch.no_grad()
def prepare_prediction_bridge_history(
    model, raw, client, fixed: dict, *, noise_fraction: float,
    target_frame: str = "local_continuation",
    decode_contract: str = "contextual_w1_v2",
) -> dict:
    """Build one detached predicted W1, commit it, then construct W2 loss.

    The caller must provide a three-window batch (W0/W1/W2) and invoke this
    before `_loss`.  The first forward predicts clean W1 from q(W1), while the
    second loss is the normal production loss for W2.  No optimizer or BPTT is
    used here.
    """
    from train_humanml3d_ardy_tiny import _generated_supervision_batch, _unwrap_raw
    from ardy_adapter.history_commit import canonical_history_prime
    from g1_motion_contract import decode_training_prediction

    if fixed.get("sequential_rollout_windows") != 3:
        raise ValueError("prediction bridge requires exactly three windows")
    if decode_contract not in {"contextual_w1_v2", "legacy_unprefixed_w1_v1"}:
        raise ValueError(f"unknown prediction bridge decode contract: {decode_contract}")
    if not hasattr(client, "controller") or client.controller is None:
        raise RuntimeError("prediction bridge requires the persistent history controller")
    horizon = int(model.gen_horizon_len)
    nfpt = int(model.hybrid.num_frames_per_token)
    batch = fixed["text"].shape[0]
    device = fixed["text"].device
    t, actual_fraction = _nearest_timestep(model.diffusion, noise_fraction)
    model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps, device=device))

    seed_hybrid = fixed["history_hybrid"][:, 0].detach()
    seed_heading = fixed["history_headings"][:, 0].detach()
    clean_w1 = fixed["history_hybrid"][:, 1].detach()
    noise = torch.randn_like(clean_w1)
    noisy_w1 = model.diffusion.q_sample(
        clean_w1, torch.full((batch,), t, device=device, dtype=torch.long), noise
    )
    history_len = torch.zeros(batch, device=device, dtype=torch.long)
    generation_len = torch.full((batch,), horizon, device=device, dtype=torch.long)
    future_len = torch.zeros(batch, device=device, dtype=torch.long)
    mask = torch.ones(batch, horizon, device=device, dtype=torch.bool)
    token_mask = torch.ones(batch, horizon // nfpt, device=device, dtype=torch.bool)
    motion_mask = torch.zeros(batch, horizon, model.motion_rep.motion_rep_dim,
                              device=device, dtype=clean_w1.dtype)
    observed = torch.zeros_like(motion_mask, dtype=clean_w1.dtype)
    denoiser = _unwrap_raw(raw)
    root_training, body_training = denoiser.root_model.training, denoiser.body_model.training
    denoiser.root_model.eval(); denoiser.body_model.eval()
    # Start the bridge from the same accepted W0 state as formal generation.
    client.reset_memory()
    canonical_history_prime(
        denoiser, seed_hybrid, text_feat=fixed["text"],
        text_pad_mask=fixed["text_mask"], first_heading_angle=seed_heading,
        next_generation_frames=horizon, chunk_id=0, detach=True,
        include_synthetic_generation=False,
    )
    try:
        predicted_w1 = raw(
            noisy_w1, history_len, generation_len, future_len,
            torch.zeros_like(mask), mask, torch.zeros_like(mask),
            torch.zeros_like(token_mask), token_mask, torch.zeros_like(token_mask),
            fixed["text"], fixed["text_mask"],
            torch.full((batch,), t, device=device, dtype=torch.long),
            seed_heading, motion_mask, observed,
        ).detach()
    finally:
        denoiser.root_model.train(root_training); denoiser.body_model.train(body_training)
        model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps, device=device))

    if predicted_w1.shape != clean_w1.shape or not torch.isfinite(predicted_w1).all():
        raise RuntimeError("predicted W1 clean latent is invalid")

    # Decode the predicted W1 with the real W0 decoder prefix for root/body
    # coupling, then retain the accepted latent after the canonical recenter.
    decode_fixed = dict(fixed)
    decode_fixed.update(
        decoder_history_hybrid=seed_hybrid,
        history_mask=torch.zeros(batch, horizon, device=device, dtype=torch.bool),
        generation_mask=torch.ones(batch, horizon, device=device, dtype=torch.bool),
        pad_mask=torch.ones(batch, horizon, device=device, dtype=torch.bool),
        lengths=torch.full((batch,), horizon, device=device, dtype=torch.long),
    )
    decoded_w1 = decode_training_prediction(predicted_w1, decode_fixed)
    physical_w1 = model.motion_rep.unnormalize(decoded_w1)
    # The denoiser prediction is the accepted latent. Recenter it using the
    # decoded root, but preserve the predicted root/body tokens as one version.
    accepted_hybrid, _, accepted_root = model._recenter_history(
        predicted_w1, torch.full((batch,), horizon - 1, device=device, dtype=torch.long),
        requantize=False,
    )
    if decode_contract == "contextual_w1_v2":
        # This is the W1 pose emitted by the committed decoder at inference.
        # The accepted latent remains the same; only the W2 target's physical
        # prefix now uses the pose that was actually shown downstream.
        accepted_physical = physical_w1.clone()
    else:
        legacy_motion = model.hybrid.get_explicit_motion_from_hybrid(
            accepted_hybrid, torch.ones(batch, horizon, device=device, dtype=torch.bool),
            torch.full((batch,), horizon, device=device, dtype=torch.long),
        )
        accepted_physical = model.motion_rep.unnormalize(legacy_motion).clone()
    accepted_origin = accepted_physical[:, -1, [0, 2]].clone()
    accepted_physical[:, :, 0] -= accepted_origin[:, None, 0]
    accepted_physical[:, :, 2] -= accepted_origin[:, None, 1]
    accepted_motion = model.motion_rep.normalize(accepted_physical)

    client.reset_memory()
    client.controller.prime_accepted_history(
        seed_hybrid, text_feat=fixed["text"], text_pad_mask=fixed["text_mask"],
        first_heading_angle=seed_heading,
    )
    next_heading = model.motion_rep.get_root_heading_angle(accepted_root)[:, -1]
    canonical_history_prime(
        denoiser, accepted_hybrid, text_feat=fixed["text"],
        text_pad_mask=fixed["text_mask"], first_heading_angle=next_heading,
        next_generation_frames=horizon, chunk_id=1, detach=True,
        include_synthetic_generation=False,
    )
    state_snapshot = (
        None if denoiser.root_model._sequence_state is None else denoiser.root_model._sequence_state.clone(),
        None if denoiser.body_model._sequence_state is None else denoiser.body_model._sequence_state.clone(),
    )

    generated_full = torch.cat((fixed["rollout_motion"][:, :horizon], accepted_motion), dim=1)
    supervision = _generated_supervision_batch(
        model, fixed, mode="hybrid", depth=1, generated_full=generated_full,
        physical=fixed["rollout_physical_motion"],
        full_mask=torch.zeros_like(fixed["rollout_motion_mask"], dtype=torch.bool),
        seed_physical=fixed["rollout_physical_motion"][:, :horizon],
        seed_hybrid=seed_hybrid, seed_heading=seed_heading,
        replay_history_chunks=[(accepted_hybrid.detach(), next_heading.detach())],
        state_snapshot=state_snapshot, bptt_start_depth=0,
        bptt_start_snapshot=None, teacher_state_snapshot=None,
        max_recovery_meters=0.0, target_frame=target_frame,
    )
    # The production accepted latent is the decoder context. Do not replace it
    # with a second codec encoding of generated_full.
    supervision["decoder_history_hybrid"] = accepted_hybrid.detach()
    supervision.update(
        generated_history_bptt_windows=0,
        generated_history_attach_seed_write=False,
        generated_history_state_consistency_weight=0.0,
        generated_history_cold_seed=False,
        generated_history_prefix_contract="g1_prediction_bridge_v1",
        generated_history_bridge_decode_contract=decode_contract,
        generated_history_noise_fraction=float(noise_fraction),
        generated_history_noise_fraction_actual=actual_fraction,
        generated_history_bridge_timestep=t,
        generated_history_bridge_predicted_w1=predicted_w1,
        generated_history_bridge_target_w1=clean_w1,
        generated_history_bridge_w1_latent_error=float((predicted_w1-clean_w1).pow(2).mean().sqrt().item()),
        generated_history_state_ready=True,
        generated_history_supervision_depth=1,
    )
    fixed.update(supervision)
    fixed["generated_history_supervision_batches"] = [supervision]
    fixed["generated_history_supervision_depths"] = [1]
    fixed["generated_history_bridge"] = True
    return fixed
