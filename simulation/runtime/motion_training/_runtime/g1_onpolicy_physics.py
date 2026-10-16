"""Replay a sampled final denoising call with its actual accepted history.

Diffusion trajectory and earlier commits are detached. Gradients cover the
last denoising call and frozen decoder, not full DDIM or long-history BPTT.
This module neither edits generated outputs nor creates paired GT targets.
"""
from contextlib import contextmanager
import torch

from g1_motion_contract import GENERATION_CONTRACT


def ordinary_copy(value):
    """Inference tensors cannot be saved by autograd; clone outside its mode."""
    with torch.inference_mode(False), torch.no_grad():
        if isinstance(value, torch.Tensor):return value.detach().clone()
        if isinstance(value, tuple):return tuple(ordinary_copy(v) for v in value)
        if isinstance(value, list):return [ordinary_copy(v) for v in value]
        if isinstance(value, dict):return {k:ordinary_copy(v) for k,v in value.items()}
        return value


@contextmanager
def capture_final_calls(model, client, windows):
    """Capture selected one-based windows and their *production* decoder input.

    Isolated process only: the committed-decode function is wrapped while this
    context is open. Every hook is restored, including on generation failure.
    """
    import g1_motion_contract as contract
    if getattr(model,'generation_contract',None)!=GENERATION_CONTRACT:
        raise ValueError('Requires committed decoding')
    if (getattr(model.autoencoder,'encode_with_quantization',False)
            and type(model.autoencoder).__module__ != 'ardy.model.autoencoder.fsq'):
        raise ValueError('Unverified quantizer gradient contract')
    # Official FSQ requantize/detokenize already use round_ste and clamping.
    # Reuse them unchanged: forward is discrete-exact; backward is the existing
    # biased straight-through surrogate, not a derivative of integer rounding.
    if not windows or any(not isinstance(w,int) or w<1 for w in windows):
        raise ValueError('Expected positive window numbers')
    controller=client.controller
    original_decode=contract.decode_committed_window
    original_commit=model._sequence_kernel_post_window_hook
    records={}
    window=0
    translation=None
    pending=None
    prefix_chunks=[]

    def capture(module,args,output):
        nonlocal window,pending
        if len(args)!=18:raise RuntimeError('Unexpected CFG denoiser interface')
        if not (args[14]==0).all():return
        window+=1
        pending=None
        if window not in windows:return
        with torch.inference_mode(False),torch.no_grad():
            states=tuple(None if block._sequence_state is None else block._sequence_state.clone()
                         for block in [controller.root_block,controller.body_block])
        pending=dict(window=window,call=ordinary_copy(args),prediction=ordinary_copy(output),
                     state_snapshot=states,previous_translation=ordinary_copy(translation),
                     prefix_chunks=ordinary_copy(prefix_chunks))
        records[window]=pending

    def commit(sequence,center,root,heading,index):
        nonlocal translation
        original_commit(sequence,center,root,heading,index)
        tokens=int(model.gen_horizon_len)//model.hybrid.num_frames_per_token
        prefix_chunks.append(ordinary_copy((sequence[:,-tokens:],heading)))
        translation=center.clone() if translation is None else translation+center

    def decode(current_model,world,previous=None,*,valid_frames=None):
        explicit,next_context=original_decode(current_model,world,previous,valid_frames=valid_frames)
        if pending is not None:
            pending.update(world=ordinary_copy(world),decoder_previous=ordinary_copy(previous),
                           decoded=ordinary_copy(explicit),valid_frames=valid_frames)
        return explicit,next_context

    handle=model.denoiser.register_forward_hook(capture)
    model._sequence_kernel_post_window_hook=commit
    contract.decode_committed_window=decode
    try:
        yield records
        if set(records)!=set(windows) or any('decoded' not in r for r in records.values()):
            raise RuntimeError('Requested windows were not captured and decoded')
    finally:
        handle.remove()
        contract.decode_committed_window=original_decode
        model._sequence_kernel_post_window_hook=original_commit


def replay_final_call(model,client,record):
    """Differentiable replay through the exact recenter/translate/decode path."""
    from ardy.model.ardy_model import translate_normalized_root_motion
    from g1_motion_contract import decode_committed_window
    client.controller.restore_training_snapshot(record['state_snapshot'])
    # The completed-window counter is not in the tensor snapshot. Set the
    # runtime depth explicitly; no periodic reset/intervention is used here.
    client.controller.set_runtime_mamba_output_scale_for_depth(record['window']-1)
    prediction=model.denoiser(*record['call'])
    horizon=int(model.gen_horizon_len)
    tokens=horizon//model.hybrid.num_frames_per_token
    current,center,_=model._recenter_history(prediction[:,:tokens],
        torch.full((len(prediction),),horizon-1,device=prediction.device,dtype=torch.long),requantize=True)
    root,body=model.hybrid.get_root_and_latent_body_motion_from_hybrid(current)
    translation=center if record['previous_translation'] is None else record['previous_translation']+center
    world=model.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(
        translate_normalized_root_motion(root,translation,model.motion_rep),body)
    decoded,_=decode_committed_window(model,world,record['decoder_previous'],valid_frames=record['valid_frames'])
    return prediction,world,decoded


@torch.no_grad()
def refresh_prefix_state(model,client,record):
    """Recompute projected KV/Mamba states after any parameter-version change.

    Reusing an old projected-cache snapshot after optimizer updates is invalid.
    Accepted motion stays fixed here; this is not regeneration of the past.
    """
    from ardy_adapter.history_commit import canonical_history_prime
    if len(record['prefix_chunks']) != record['window']-1:
        raise ValueError('Incomplete accepted-prefix replay')
    client.reset_memory()
    raw=model.denoiser.model
    for index,(history,heading) in enumerate(record['prefix_chunks']):
        client.controller.set_runtime_mamba_output_scale_for_depth(index)
        canonical_history_prime(raw,history,text_feat=record['call'][12],
            text_pad_mask=record['call'][13],first_heading_angle=heading,
            next_generation_frames=int(model.gen_horizon_len),chunk_id=index,detach=True)
    for block in [client.controller.root_block,client.controller.body_block]:
        block._ensure_state(record['call'][2].shape[0])
    snapshot=tuple(b._sequence_state.clone() for b in [client.controller.root_block,client.controller.body_block])
    return dict(record,state_snapshot=snapshot)


def physical_terms(rep,decoded,reference,spec_path):
    """Raw limit and fixed-reference-contact foot objectives on sampled output.

    Contact labels and the reference are detached. No gradient can lower this
    objective merely by changing predicted contact labels. Full motion checks
    still have to reject static/low-contact/semantically incorrect solutions.
    """
    from g1_joint_limits import converter_for,exported_joint_angles,load_spec
    geometry=rep.inverse(decoded,is_normalized=True)
    _,bounds,limited=load_spec(str(spec_path))
    angles=exported_joint_angles(geometry['local_rot_mats'],converter_for(rep.skeleton))
    lo,hi=bounds.to(angles).unbind(-1)
    excess=(torch.relu(lo-angles)+torch.relu(angles-hi))[:,:,limited.to(angles.device)]
    squared=excess.square()
    # Report both the current training reduction and a sparse tail diagnostic.
    # Per-joint temporal max gives each violating joint a non-diluted path.
    mean_limit=squared.mean()
    tail_limit=squared.amax(1).mean()
    with torch.no_grad():
        contacts=rep.inverse(reference.detach(),is_normalized=True)['foot_contacts'].bool()
    feet=geometry['posed_joints'][:,:,list(rep.skeleton.foot_joint_idx)]
    velocity=(feet[:,1:,:,[0,2]]-feet[:,:-1,:,[0,2]])*rep.fps
    mask=contacts[:,1:] & contacts[:,:-1]
    foot=velocity[mask].square().mean() if mask.any() else decoded.sum()*0
    return dict(limit_mean=mean_limit,limit_tail=tail_limit,foot=foot,
                contact_pairs=mask.sum().detach(),max_excess_deg=torch.rad2deg(excess.max()).detach(),
                violating_frame_fraction=(excess>1e-5).any(-1).float().mean().detach())
