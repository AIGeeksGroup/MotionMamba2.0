"""Cold on-policy prefixes retaining deployment's accepted latent/state path.

The label builder remains separate: fixing prefix fidelity does not certify
phase-matched GT labels as physically or semantically compatible.
"""
import math
import torch
from g1_motion_contract import GENERATION_CONTRACT, generate_with_committed_decode

PREFIX_CONTRACT = 'g1_cold_accepted_latents_v1'


def check_target_seams(model, supervision, accepted_tail, bounds):
    """Same calibrated raw AND actual-decoder seam check; no label rewriting."""
    from audit_g1_raw_seam_gate_20260921 import seam_values
    from g1_motion_contract import decode_training_prediction
    expected={'root_velocity_change_m_s','root_height_step_m','local_rotation_step_rad',
              'joint_step_m','contact_foot_speed_m_s'}
    if set(bounds)!=expected or any(not math.isfinite(v) or v<=0 for v in bounds.values()):
        raise ValueError('Invalid calibrated seam bounds')
    rep=model.motion_rep
    history=rep.unnormalize(accepted_tail)
    history=rep.translate_2d(history,-history[:,-1,[0,2]])
    decoded=decode_training_prediction(supervision['target'],supervision)
    values={
        'raw':seam_values(rep,history,rep.unnormalize(supervision['motion'])),
        'actual_decoder':seam_values(rep,history,rep.unnormalize(decoded))}
    rows=[]
    for i in range(len(history)):
        row={kind:dict(values=metrics[i],failed=[k for k,bound in bounds.items()
                   if metrics[i][k] is None or not math.isfinite(metrics[i][k]) or metrics[i][k]>bound]) for kind,metrics in values.items()}
        row['passed']=all(not row[kind]['failed'] for kind in values)
        rows.append(row)
    return rows


def require_compatible_targets(checks):
    if not checks or any(not row['passed'] for row in checks):
        raise ValueError('Incompatible generated-history target: raw/actual-decoder seam failed; do not optimize this batch')


@torch.no_grad()
def prepare_cold_generated_history(model, raw, client, fixed, *, mode,
        diffusion_steps, cfg_weight_text, cfg_weight_constraint,
        max_recovery_meters=0., max_supervision_points=1, bptt_windows=0,
        attach_seed_write=False, state_consistency_weight=0.,
        root_endpoint_loss_weight=0., target_frame='local_continuation',
        seam_bounds=None, require_target_compatibility=False):
    from train_humanml3d_ardy_tiny import (
        _unwrap_raw, _generated_supervision_batch, select_generated_history_supervision_depths)
    from main3_runtime import validate_sampling_configuration
    validate_sampling_configuration(model,diffusion_steps,(cfg_weight_text,cfg_weight_constraint))
    if require_target_compatibility and seam_bounds is None:
        raise ValueError('Training with accepted prefixes requires calibrated target seam bounds')
    if (mode!='hybrid' or getattr(model,'generation_contract',None)!=GENERATION_CONTRACT
            or bptt_windows or attach_seed_write or state_consistency_weight
            or target_frame not in {'local_continuation','phase_compatible_continuation'}
            or max_recovery_meters or root_endpoint_loss_weight<0):
        raise ValueError('Accepted cold prefix supports text-only detached Native histories; unsupported options cannot silently fall back')
    horizon=int(model.gen_horizon_len);rollout=int(fixed['sequential_rollout_windows'])
    if rollout<3 or fixed['rollout_physical_motion'].shape[1]<rollout*horizon:
        raise ValueError('Need cold seed, generated history and a complete source target')
    denoiser=_unwrap_raw(raw);controller=client.controller
    depths=select_generated_history_supervision_depths(rollout-2,max_supervision_points)
    windows=rollout-1;frames=windows*horizon;batch=len(fixed['text']);device=fixed['text'].device
    flags=(denoiser.root_model.training,denoiser.body_model.training)
    client.reset_memory();denoiser.root_model.eval();denoiser.body_model.eval()
    controller.begin_training_snapshot_capture(sorted({1,*[d+1 for d in depths]}))
    finished=False
    try:
        zeros=torch.zeros(batch,frames,model.motion_rep.motion_rep_dim,device=device)
        generated=generate_with_committed_decode(model,['']*batch,frames,decoder_prefix=None,
            num_denoising_steps=diffusion_steps,pad_mask=torch.ones(batch,frames,device=device,dtype=torch.bool),
            motion_mask=zeros.bool(),observed_motion=zeros,first_heading_angle=torch.zeros(batch,device=device),
            crop_history_length=0,text_feat=fixed['text'],text_pad_mask=fixed['text_mask'],
            cfg_weight=(cfg_weight_text,cfg_weight_constraint),progress_bar=lambda values:values)
        snapshots,chunks=controller.end_training_snapshot_capture();finished=True
        assert controller.trace.accepted_commits==windows and len(chunks)==windows
        assert set(snapshots)=={1,*[d+1 for d in depths]}
        assert generated.shape==(batch,frames,model.motion_rep.motion_rep_dim) and torch.isfinite(generated).all()
    except Exception:
        if not finished:controller.end_training_snapshot_capture()
        client.reset_memory()
        raise
    finally:
        denoiser.root_model.train(flags[0]);denoiser.body_model.train(flags[1])
        model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps,device=device))
    physical=fixed['rollout_physical_motion']
    full_mask=torch.zeros_like(fixed['rollout_motion_mask'],dtype=torch.bool)
    seed_physical=model.motion_rep.unnormalize(generated[:,:horizon])
    supervisions=[]
    for depth in depths:
        supervision=_generated_supervision_batch(model,fixed,mode=mode,depth=depth,
            generated_full=generated,physical=physical,full_mask=full_mask,seed_physical=seed_physical,
            seed_hybrid=chunks[0][0],seed_heading=chunks[0][1],replay_history_chunks=chunks[1:depth+1],
            state_snapshot=snapshots[depth+1],bptt_start_depth=depth,bptt_start_snapshot=snapshots[depth+1],
            teacher_state_snapshot=None,max_recovery_meters=0.,target_frame=target_frame)
        # Each accepted chunk is already recentered at its own final root.
        # The next target uses precisely that local frame. Keep its actual
        # body tokens instead of encoding the decoded prefix a second time.
        actual=chunks[depth][0].detach().clone()
        reencoded=supervision['decoder_history_hybrid']
        supervision['decoder_prefix_reencoding_max_abs']=float((actual-reencoded).abs().max())
        supervision['decoder_history_hybrid']=actual
        supervision.update(generated_history_bptt_windows=0,generated_history_attach_seed_write=False,
            generated_history_state_consistency_weight=0.,generated_history_root_endpoint_loss_weight=root_endpoint_loss_weight,
            generated_history_cold_seed=True,generated_history_seed_source='current_weight_text_only_accepted_latents',
            generated_history_gt_seed_frames=0,generated_history_prefix_contract=PREFIX_CONTRACT,
            generated_history_accepted_windows=depth+1)
        if seam_bounds is not None:
            stop=(depth+1)*horizon
            supervision['generated_history_target_checks']=check_target_seams(model,supervision,
                generated[:,stop-2:stop],seam_bounds)
            if require_target_compatibility:
                require_compatible_targets(supervision['generated_history_target_checks'])
        supervisions.append(supervision)
    fixed.update(supervisions[-1])
    fixed['generated_history_supervision_batches']=supervisions
    fixed['generated_history_supervision_depths']=depths
    # Retain these tensors for assertions/audits. No extra model forwards.
    fixed['generated_history_actual_motion']=generated.detach()
    fixed['generated_history_actual_chunks']=chunks
    fixed['generated_history_actual_snapshots']=snapshots
    return fixed
