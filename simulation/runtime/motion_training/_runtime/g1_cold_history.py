"""Actual text-only cold W1 for subsequent training-target validation.

This accepts no GT motion, teacher seed, or spatial condition. It does not
construct or authorize a W2 target. Compatible supervision and teacher fallback
must be resolved separately before an optimizer can consume the prefix.
"""
import torch

from g1_motion_contract import GENERATION_CONTRACT, generate_with_committed_decode


@torch.no_grad()
def sample_generated_cold_prefix(model, client, text, text_mask, *,
                                 diffusion_steps=10, cfg_weight=(2., 2.)):
    from main3_runtime import validate_sampling_configuration
    validate_sampling_configuration(model, diffusion_steps, cfg_weight)
    if (text.ndim != 3 or text_mask.shape != text.shape[:2] or text_mask.dtype != torch.bool
            or not text_mask.any(-1).all() or not torch.isfinite(text).all()):
        raise ValueError('Cold-history development requires real unmasked text')
    if getattr(model, 'generation_contract', None) != GENERATION_CONTRACT:
        raise ValueError('Cold prefix requires committed decoder contract')
    controller = getattr(client, 'controller', None)
    if controller is None or not controller.commit_enabled:
        raise ValueError('Cold prefix requires Native accepted-history controller')
    raw = model.denoiser.model
    flags = (raw.root_model.training, raw.body_model.training)
    batch, horizon = len(text), int(model.gen_horizon_len)
    device = text.device
    client.reset_memory()
    raw.root_model.eval()
    raw.body_model.eval()
    controller.begin_training_snapshot_capture([1])
    finished = False
    succeeded = False
    try:
        generated = generate_with_committed_decode(model, ['']*batch, horizon,
            decoder_prefix=None, num_denoising_steps=diffusion_steps,
            pad_mask=torch.ones(batch, horizon, device=device, dtype=torch.bool),
            motion_mask=torch.zeros(batch, horizon, model.motion_rep.motion_rep_dim, device=device, dtype=torch.bool),
            observed_motion=torch.zeros(batch, horizon, model.motion_rep.motion_rep_dim, device=device),
            first_heading_angle=torch.zeros(batch, device=device), crop_history_length=0,
            text_feat=text, text_pad_mask=text_mask, cfg_weight=cfg_weight,
            progress_bar=lambda values: values)
        snapshots, chunks = controller.end_training_snapshot_capture()
        finished = True
        if (controller.trace.accepted_commits != 1 or set(snapshots) != {1}
                or len(chunks) != 1 or generated.shape != (batch, horizon, model.motion_rep.motion_rep_dim)
                or not torch.isfinite(generated).all()):
            raise RuntimeError('Cold prefix did not commit exactly one finite generated window')
        succeeded = True
        return dict(prefix_origin='generated_cold', generated_windows=1, teacher_seed_frames=0,
                    motion=generated, physical=model.motion_rep.unnormalize(generated),
                    state_snapshot=snapshots[1], accepted_hybrid=chunks[0][0],
                    next_heading=chunks[0][1], supervised_target=None,
                    target_authorized=False)
    finally:
        if not finished:
            controller.end_training_snapshot_capture()
        if not succeeded:
            client.reset_memory()
        raw.root_model.train(flags[0])
        raw.body_model.train(flags[1])
        # Inference uses a spaced schedule; q_sample for training needs base t.
        model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps, device=device))
