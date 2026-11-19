"""Autoregressive sampling with continuously visible sparse terminal controls."""
import inspect
import random
import numpy as np
import torch
from motion_mamba2.model.decoder_history import generate_with_committed_decode

def generate(model,client,initial,indices,values,texts,features,seed,*,steps=10,cfg=(2.,2.)):
    device=model.device;windows=len(texts);frames=windows*52
    history=torch.as_tensor(initial,dtype=torch.float32,device=device)[None]
    normalized=model.motion_rep.normalize(history)
    coords=torch.as_tensor(indices,dtype=torch.long,device=device)
    goal=history.new_zeros(1,1,414);goal[:,:,coords]=torch.as_tensor(values,device=device)
    goal=model.motion_rep.normalize(goal)
    observed=history.new_zeros(1,frames,414);mask=torch.zeros_like(observed)
    mask[:,-1,coords]=1;observed[:,-1,coords]=goal[:,0,coords]
    if int(mask.count_nonzero())!=8:raise ValueError('Expected eight sparse terminal control scalars')
    text={s:torch.as_tensor(features[s],dtype=torch.float32,device=device).reshape(1,1,4096) for s in set(texts)}
    text_mask=torch.ones(1,1,dtype=torch.bool,device=device)
    heading=torch.atan2(history[:,-1,4],history[:,-1,3])
    original=model._generate_window;signature=inspect.signature(original);events=[]
    def window(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();i=len(events)
        if i>=windows or bound.arguments.get('target_motion') is not None:raise ValueError('Invalid generation boundary')
        bound.arguments['text_feat']=text[texts[i]];bound.arguments['text_pad_mask']=text_mask
        current_mask=bound.arguments['motion_mask'];current_observed=bound.arguments['observed_motion']
        if int(current_mask.count_nonzero())!=8:raise ValueError('Sparse future goal was lost or duplicated')
        if torch.count_nonzero(current_observed.masked_select(current_mask==0)):raise ValueError('Unobserved controls contain data')
        events.append(dict(window=i+1,prompt=texts[i],visible_scalars=8))
        return original(*bound.args,**bound.kwargs)
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model._generate_window=window
    try:
        with torch.inference_mode():
            client.reset_memory()
            valid=torch.ones(1,24,dtype=torch.bool,device=device)
            hybrid,_=model.hybrid.get_hybrid_motion_from_explicit(normalized,valid.sum(1),valid)
            client.controller.prime_accepted_history(hybrid,text_feat=text[texts[0]],text_pad_mask=text_mask,first_heading_angle=heading)
            prediction=generate_with_committed_decode(model,[texts[0]],frames,
                decoder_prefix=hybrid[:,-52//model.num_frames_per_token:],num_denoising_steps=steps,
                pad_mask=torch.ones(1,frames,dtype=torch.bool,device=device),first_heading_angle=heading,
                motion_mask=mask,observed_motion=observed,cfg_weight=tuple(cfg),crop_history_length=0,
                text_feat=text[texts[0]],text_pad_mask=text_mask,progress_bar=lambda x:x)
            geometry=model.motion_rep.inverse(prediction,is_normalized=True)
            physical=model.motion_rep.unnormalize(prediction)[0].float().cpu().numpy()
            joints=geometry['posed_joints'][0].float().cpu().numpy()
            root=geometry['root_positions'][0].float().cpu().numpy()
    finally:model._generate_window=original
    if len(events)!=windows:raise ValueError('Incomplete rollout')
    return dict(physical=physical,joints=joints,root=root),events
