"""Current-weight priming of accepted history, independent of pool generation."""
from dataclasses import dataclass
import numpy as np
import torch
from ardy_adapter.history_commit import canonical_history_prime
from ardy.model.ardy_model import translate_normalized_root_motion
from g1_online_controls import conditions
from g1_motion_contract import CONTRACT,decode_training_prediction
from .io import identity,copy_state
from .schema import validate_record

@dataclass
class HistoryContext:
    chunks:list
    physical:torch.Tensor
    collector_checkpoint_sha256:str
    state_version:str='accepted_history_v3_history_only'
    def prime(self,m,c,text):
        raw=m.denoiser.model;was=raw.training;raw.eval();c.reset_memory()
        with torch.no_grad():
            for i,(tokens,kwargs) in enumerate(self.chunks):
                assert kwargs['include_synthetic_generation'] is False
                canonical_history_prime(raw,tokens,**kwargs)
        raw.train(was);ctl=c.controller;n=len(self.chunks)
        ctl._completed_windows=ctl._accepted_commits=ctl._next_chunk_id=n;ctl._last_text_feat=text;ctl._last_text_pad_mask=torch.ones(text.shape[:2],device=text.device,dtype=torch.bool)
        return dict(history_frames=int(self.physical.shape[1]),prime_chunks=n,physical_hash=identity(self.physical),collector=self.collector_checkpoint_sha256,weight_state_rebuilt=True,cached_numerical_state_reused=False,explicit_history_frames=0,duplicated_frames=0)

def teacher_batch(m,c,r,cfg,old_cfg):
    validate_record(r);item=copy_state(torch.load(r['label'],map_location='cpu',weights_only=False),m.device)
    hist=HistoryContext(item['chunks'],item['physical'],r['collector_checkpoint_sha256']);hr=hist.prime(m,c,item['target']['text'])
    target=item['target']['target_latent'];prefix=item['target']['decoder_prefix'];device=m.device
    valid=torch.ones(1,52,device=device,dtype=torch.bool);tokens=torch.ones(1,13,device=device,dtype=torch.bool);z=torch.zeros(1,device=device,dtype=torch.long);lengths=z+52
    gen=torch.Generator(device=device).manual_seed(r['seed']);t=torch.randint(m.diffusion.num_base_steps,(1,),generator=gen,device=device);noise=torch.randn(target.shape,generator=gen,device=device,dtype=target.dtype)
    obs,mask=conditions(m,item['target'].get('control'),item['translation']);root=translate_normalized_root_motion(m.motion_rep.extract_root(obs),-item['translation'],m.motion_rep);obs=m.motion_rep.concat_root_body(root,m.motion_rep.extract_body(obs))*mask
    rng=np.random.default_rng(r['seed']);td=bool(rng.random()<cfg['text_dropout']);cd=bool(rng.random()<cfg['constraint_dropout']);text=item['target']['text'].clone()
    if td:text.zero_()
    if cd:obs.zero_();mask.zero_()
    m.diffusion.calc_diffusion_vars(torch.arange(m.diffusion.num_base_steps,device=device))
    f=dict(x=m.diffusion.q_sample(target,t,noise),target=target,decoder_history_hybrid=prefix,history_len=z,generation_len=lengths,future_len=z,history_mask=~valid,generation_mask=valid,future_mask=~valid,history_token_mask=~tokens,generation_token_mask=tokens,future_token_mask=~tokens,text=text,text_mask=torch.tensor([[not td]],device=device),heading=item['heading'],motion_mask=mask.float(),observed=obs,pad_mask=valid,lengths=lengths,timestep=t,noise=noise,generated_history_state_ready=True,sequential_cache=True,motion_contract=CONTRACT,hybrid_converter=m.hybrid)
    with torch.no_grad():f['supervised_motion']=decode_training_prediction(target,f).detach()
    rec=dict(r,history=hr,text_dropped=td,condition_dropped=cd,F_nonzero=False,F_before_drop=False,F_visible_after_drop=False,effective_G_frames=52,effective_G_tokens=13,G_goal_scalars=int(mask.sum()),observed_scalars=int(mask.sum()),target_hash=identity(target),input_hash=identity([f['x'],text,obs,mask]),noise_hash=identity(noise))
    return f,rec
