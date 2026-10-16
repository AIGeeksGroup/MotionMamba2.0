"""Main3 native generation, common metrics, and constraint-only postprocessing."""
from __future__ import annotations

from types import SimpleNamespace
import types
import math
import torch


def install_native_history_cap(model, frames=260):
    original = model._generate_window

    def generate(self, history_sequence, global_transl, history_start_frame,
                 history_end_frame, total_frames, text_feat, text_pad_mask,
                 first_heading_angle, *args, **kwargs):
        start = max(history_start_frame, history_end_frame - frames)
        if start > history_start_frame and history_sequence is not None:
            root, _ = self.hybrid.get_root_and_latent_body_motion_from_hybrid(history_sequence)
            physical = self.motion_rep.global_root_stats.unnormalize(root)
            first_heading_angle = self.motion_rep.get_root_heading_angle(physical)[:, start]
        return original(history_sequence, global_transl, start, history_end_frame,
                        total_frames, text_feat, text_pad_mask, first_heading_angle, *args, **kwargs)

    model._generate_window = types.MethodType(generate, model)


def load_generator(checkpoint_path, mode, stats_dir, device='cuda:0'):
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel
    from train_humanml3d_ardy_tiny import _replace_stats, _load_denoiser_weights
    from evaluation_checkpoint_config import sequence_config_kwargs
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    from g1_motion_contract import CONTRACT, inspect_stats, validate_checkpoint
    config = checkpoint.get('config', {})
    # Read-only provenance for service health and isolated acceptance reports.
    if config.get('motion_contract') == CONTRACT:
        stats_dir = config['stats_dir']
    metadata = inspect_stats(stats_dir)
    validate_checkpoint(checkpoint, metadata)
    model = load_model('g1', device=device, text_encoder=False)
    model.training_codec_contract = config.get('training_codec_contract', 'legacy')
    model.checkpoint_training_manifest = config.get('manifest')
    _replace_stats(model, stats_dir)
    model.generation_contract = config.get('generation_contract', 'legacy')
    if checkpoint['mode'] != mode or checkpoint.get('model') != 'g1':
        raise ValueError('checkpoint architecture mismatch')
    if mode == 'original':
        client = model
        install_native_history_cap(model)
    else:
        options, _ = sequence_config_kwargs(SimpleNamespace(), checkpoint)
        client = install_ardy_sequence_kernel(model, ArdySequenceKernelConfig(
            mode=mode, attention_backend='sdpa', window_size_chunks=4,
            num_sink_tokens=4, **options))
    raw = model.denoiser.model
    for key in ['root_model', 'body_model']:
        current = getattr(raw, key).state_dict()
        if set(current) != set(checkpoint[key]):
            raise ValueError(f'incomplete checkpoint keys for {key}')
    _load_denoiser_weights(raw, checkpoint)
    from g1_padding_contract import install_padding_contract
    install_padding_contract(model,config.get('padding_contract','legacy'))
    from g1_root_body_gradient import configure_root_body_gradient
    configure_root_body_gradient(raw,config.get('root_body_gradient_scale',0.))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.skeleton.nbjoints == 34 and model.motion_rep.fps == 25
    assert model.motion_rep.motion_rep_dim == 414 and model.gen_horizon_len == 52
    return model, client


def validate_sampling_configuration(model, num_denoising_steps, cfg_weight):
    """Reject incompatible schedules before seeding or mutating generation state."""
    if not isinstance(num_denoising_steps,int) or isinstance(num_denoising_steps,bool) or num_denoising_steps < 1:
        raise ValueError('num_denoising_steps must be a positive integer')
    maximum = int(model.diffusion.num_base_steps)
    if num_denoising_steps > maximum:
        raise ValueError(f'num_denoising_steps must not exceed this model\'s num_base_steps={maximum}; got {num_denoising_steps}')
    if len(cfg_weight)!=2 or any(not math.isfinite(v) or v<0 for v in cfg_weight):
        raise ValueError('cfg_weight must contain finite nonnegative text/constraint weights')


def generate(model, client, mode, motion, mask, text, caption, prefix, seed, *,
             num_denoising_steps=10, cfg_weight=(2.0, 2.0)):
    validate_sampling_configuration(model, num_denoising_steps, cfg_weight)
    device = motion.device
    total = motion.shape[1]
    text_mask = torch.ones(text.shape[:2], device=device, dtype=torch.bool)
    observed = motion * mask
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    common = dict(num_denoising_steps=num_denoising_steps, cfg_weight=tuple(cfg_weight), text_feat=text,
                  text_pad_mask=text_mask, progress_bar=lambda values: values)
    trace = []
    committed_outputs = []
    saved_commit_hook = None
    if mode != 'original':
        client.reset_memory()
        block = client.controller.root_block
        original = model._generate_window
        def traced(*args, **kwargs):
            state = block._sequence_state
            cache = None if state is None else state.layers[0].attention_cache
            trace.append(dict(window=len(trace) + 1, cache_chunk_ids=[] if cache is None else list(cache.chunk_ids)))
            return original(*args, **kwargs)
        model._generate_window = traced
    try:
        with torch.inference_mode():
            if mode == 'original':
                return model([caption], total, pad_mask=torch.ones(1, total, device=device, dtype=torch.bool),
                             motion_mask=mask, observed_motion=observed,
                             first_heading_angle=None if prefix else torch.zeros(1, device=device),
                             crop_history_length=None,
                             init_history_sequence=motion[:, :prefix] if prefix else None,
                             **common)[:, prefix:], trace
            heading = torch.zeros(1, device=device)
            if prefix:
                initial = motion[:, :prefix]
                physical = model.motion_rep.unnormalize(initial)
                heading = torch.atan2(physical[:, -1, 4], physical[:, -1, 3])
                hybrid, _ = model.hybrid.get_hybrid_motion_from_explicit(
                    initial, torch.tensor([prefix], device=device),
                    torch.ones(1, prefix, device=device, dtype=torch.bool))
                client.controller.prime_accepted_history(hybrid, text_feat=text,
                    text_pad_mask=text_mask, first_heading_angle=heading)
            from g1_motion_contract import GENERATION_CONTRACT, generate_with_committed_decode
            if getattr(model, 'generation_contract', 'legacy') == GENERATION_CONTRACT:
                length = total - prefix
                result = generate_with_committed_decode(model, [caption], length,
                    decoder_prefix=hybrid[:, -13:] if prefix else None,
                    pad_mask=torch.ones(1, length, device=device, dtype=torch.bool),
                    motion_mask=mask[:, prefix:], observed_motion=observed[:, prefix:],
                    first_heading_angle=heading, crop_history_length=0, **common)
                return result, trace
            if getattr(model, 'decoder_history_frames', None) is not None:
                from g1_motion_contract import decode_committed_window
                from ardy.model.ardy_model import translate_normalized_root_motion
                saved_commit_hook = model._sequence_kernel_post_window_hook
                decoder_previous = hybrid[:, -13:].clone() if prefix else None
                translation = torch.zeros(1, 3, device=device)
                def committed(sequence, center, root, next_heading, window):
                    nonlocal decoder_previous, translation
                    saved_commit_hook(sequence, center, root, next_heading, window)
                    translation = translation + center
                    accepted = sequence[:, -model.gen_horizon_len//model.hybrid.num_frames_per_token:]
                    local_root, body = model.hybrid.get_root_and_latent_body_motion_from_hybrid(accepted)
                    world_root = translate_normalized_root_motion(local_root, translation, model.motion_rep)
                    world = model.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(world_root, body)
                    from g1_motion_contract import GENERATION_CONTRACT
                    valid_frames = (min(model.gen_horizon_len, total-prefix-window*model.gen_horizon_len)
                        if getattr(model, 'generation_contract', 'legacy') == GENERATION_CONTRACT else None)
                    decoded, decoder_previous = decode_committed_window(model, world, decoder_previous,
                                                                        valid_frames=valid_frames)
                    committed_outputs.append(decoded)
                model._sequence_kernel_post_window_hook = committed
            length = total - prefix
            result = model([caption], length,
                         pad_mask=torch.ones(1, length, device=device, dtype=torch.bool),
                         motion_mask=mask[:, prefix:], observed_motion=observed[:, prefix:],
                         first_heading_angle=heading, crop_history_length=0, **common)
            if committed_outputs:
                result = torch.cat(committed_outputs, dim=1)[:, :length]
            return result, trace
    finally:
        if mode != 'original':
            model._generate_window = original
            if saved_commit_hook is not None:
                model._sequence_kernel_post_window_hook = saved_commit_hook


def geometric_metrics(model, prediction, target, goal_frames, goal_joints):
    from evaluate_humanml3d_table45 import _rigid_pa_mpjpe, _angle_mae_degrees, _skate_counts
    with torch.no_grad():
        pred = model.motion_rep.inverse(prediction, is_normalized=True)
        gt = model.motion_rep.inverse(target, is_normalized=True)
        p, g = pred['posed_joints'], gt['posed_joints']
        root = torch.linalg.vector_norm(pred['root_positions'] - gt['root_positions'], dim=-1)[0] * 100
        constraint = []
        windows = [None] * ((len(root) + 51) // 52)
        for f, j in zip(goal_frames, goal_joints):
            difference = p[0, f, j] - g[0, f, j]
            if j == 0:
                difference = difference[[0, 2]]
            value = float(difference.norm() * 100)
            constraint.append(value)
            windows[f // 52] = value
        bad, count = _skate_counts(p, pred['foot_contacts'], list(model.skeleton.foot_joint_idx), 25)
        feet = p[:, :, list(model.skeleton.foot_joint_idx)]
        speed = torch.linalg.vector_norm(feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]], dim=-1) * 25
        contact = (feet[:, 1:, :, 1] < 0.05) & (feet[:, :-1, :, 1] < 0.05)
        jitter = torch.linalg.vector_norm(p[:, 2:] - 2*p[:, 1:-1] + p[:, :-2], dim=-1).mean()
        values = dict(constraint_error_cm=sum(constraint)/len(constraint),
                      final_goal_constraint_cm=constraint[-1], root_ade_cm=float(root.mean()),
                      root_fde_cm=float(root[-1]), pa_mpjpe_cm=float(_rigid_pa_mpjpe(p, g)*100),
                      jitter_m_per_frame2=float(jitter), acceleration_m_s2=float(jitter*625),
                      angle_mae_deg=float(_angle_mae_degrees(pred['local_rot_mats'], gt['local_rot_mats'])),
                      foot_skate_percent_proxy=100*bad/count,
                      geometric_contact_skate_percent=100*float(((speed > 0.15)&contact).sum())/max(int(contact.sum()),1),
                      contact_speed_m_s=float(speed[contact].mean()) if contact.any() else None,
                      root_by_window=[float(root[s:s+52].mean()) for s in range(0, len(root), 52)],
                      constraint_by_window=windows)
        return values, pred, gt


def optimize_root_translation(model, prediction, given_target, goal_frames, goal_joints, iterations=50):
    """Same root-translation variables/objective for all models; no unseen GT loss."""
    with torch.no_grad():
        physical = model.motion_rep.unnormalize(prediction).detach()
        pred = model.motion_rep.inverse(prediction, is_normalized=True)['posed_joints'].detach()
        supplied = model.motion_rep.inverse(given_target, is_normalized=True,
                                             posed_joints_from='positions')['posed_joints'].detach()
        targets = [(supplied[0, f, j] - pred[0, f, j]).detach() for f, j in zip(goal_frames, goal_joints)]
    shift = torch.zeros_like(physical[:, :, :3], requires_grad=True)
    optimizer = torch.optim.Adam([shift], lr=0.02)
    with torch.enable_grad():
        for _ in range(iterations):
            optimizer.zero_grad()
            terms = []
            for f, j, difference in zip(goal_frames, goal_joints, targets):
                error = shift[0, f] - difference
                terms.append(error[[0, 2]].square().mean() if j == 0 else error.square().mean())
            loss = torch.stack(terms).mean() + 0.001*shift.square().mean() + 0.1*(shift[:, 1:]-shift[:, :-1]).square().mean()
            loss.backward()
            optimizer.step()
    physical = physical.clone()
    physical[:, :, :3] += shift.detach()
    return model.motion_rep.normalize(physical)
