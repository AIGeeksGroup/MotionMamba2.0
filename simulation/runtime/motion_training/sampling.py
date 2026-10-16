"""Accepted-window lifecycle shared by collection and evaluation."""
import contextlib,math,time
import numpy as np
import torch
from ardy.model.ardy_model import translate_normalized_root_motion
from g1_online_controls import conditions
from g1_control_intents import compile_intent,VERSION
from g1_motion_contract import decode_committed_window
from ardy_adapter.history_commit import canonical_history_prime
from .io import copy_state,identity
from ._compat.metrics import metrics

def fresh_noise(seed,n):
    gen=torch.Generator().manual_seed(seed);return [torch.randn(1,13,148,generator=gen) for _ in range(n)]
def measure(m,dec):
    row,q=metrics(m.motion_rep,dec);row['leg_range_deg']=float(np.rad2deg(np.ptp(q[:,7:19],axis=0)).mean())
    return row,q
def world_motion(m,local,translation):
    root,body=m.hybrid.get_root_and_latent_body_motion_from_hybrid(local)
    return m.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(translate_normalized_root_motion(root,translation,m.motion_rep),body)
def localize(m,world,origin):return world_motion(m,world,-origin)
class NoiseProbe:
    def __init__(self,m):
        self.m=m;self.original=m.denoising_step;self.noise=None;self.receipts=[]
        def step(*args,**kw):
            a=list(args)
            if int(a[12][0])==9 and self.noise is not None:
                assert self.noise.numel()==a[0][a[8]].numel(), 'Noise shape does not match generation region'
                a[0]=a[0].clone();a[0][a[8]]=self.noise.to(a[0]).reshape(-1,a[0].shape[-1]);self._mask=a[8].clone();self.pending=True
            return self.original(*a,**kw)
        self.pending=False;m.denoising_step=step
        def verify(module,args):
            if self.pending:
                value=args[2];expected=self.noise.to(value).reshape(-1,value.shape[-1]);assert torch.equal(value[self._mask],expected),'CFG noise mismatch'
                self.receipts.append(identity(value));self.pending=False
        self.handle=m.denoiser.register_forward_pre_hook(verify)
    def begin(self,noise):self.noise=noise
    def close(self):self.handle.remove();self.m.denoising_step=self.original
@contextlib.contextmanager
def capture_context():
    import ardy_adapter.diffusion_memory as dm
    chunks=[];original=dm.canonical_history_prime
    def capture(raw,h,**kw):
        chunks.append((h.detach().clone(),copy_state(kw,h.device)));return original(raw,h,**kw)
    dm.canonical_history_prime=capture
    try:yield chunks
    finally:dm.canonical_history_prime=original
class Native:
    def __init__(self,m,c,p,text):
        self.m=m;self.c=c.controller;self.p=p;self.text=text;self.mask=torch.ones(text.shape[:2],device=m.device,dtype=torch.bool);self.reset()
    def reset(self):
        self.c.reset();self.translation=torch.zeros(1,3,device=self.m.device);self.heading=torch.zeros(1,device=self.m.device);self.previous=None;self.physical=None;self.i=0
    def accept(self,clean):
        seq,center,root=self.m._recenter_history(clean,torch.tensor([51],device=self.m.device),requantize=True)
        self.heading=self.m.motion_rep.get_root_heading_angle(self.m.motion_rep.global_root_stats.unnormalize(root))[:,-1]
        self.m._sequence_kernel_post_window_hook(seq,center,root,self.heading,self.i);self.translation+=center
        world=world_motion(self.m,seq[:,-13:],self.translation);dec,self.previous=decode_committed_window(self.m,world,self.previous)
        ph=self.m.motion_rep.unnormalize(dec);self.physical=ph if self.physical is None else torch.cat((self.physical,ph),1);self.i+=1
        row,q=measure(self.m,dec);return row,q,dec,world,clean
    def step(self,noise,control=None,future_observation=None):
        self.p.begin(noise);obs,mask=conditions(self.m,control,self.translation)
        if future_observation is not None:obs,mask=future_observation
        before=self.c._accepted_commits
        seq=self.m._generate_window(None,self.translation,0,0,obs.shape[1],self.text,self.mask,self.heading,mask,obs,torch.tensor([10],device=self.m.device),(2.,2.),list(range(9,-1,-1)),progress_bar=lambda x:x,target_motion=None,cfg_type=None)
        out=self.accept(seq[:,:13].clone());assert self.c._accepted_commits==before+1;return out
def prime_gt(m,c,p,physical,text):
    from g1_bounded_codec import encode_windows
    runner=Native(m,c,p,text);n=physical.shape[1];assert n%52==0
    normalized=m.motion_rep.normalize(physical);pad=torch.ones(normalized.shape[:2],device=m.device,dtype=torch.bool);tokens,_=encode_windows(m.hybrid,normalized,pad.sum(1),pad);receipt=[]
    import ardy_adapter.diffusion_memory as dm
    for i in range(n//52):
        hybrid=tokens[:,i*13:(i+1)*13];local,center,root=m._recenter_history(hybrid,torch.tensor([51],device=m.device),requantize=False)
        heading=m.motion_rep.get_root_heading_angle(m.motion_rep.global_root_stats.unnormalize(root))[:,-1]
        dm.canonical_history_prime(m.denoiser.model,local,text_feat=text,text_pad_mask=runner.mask,first_heading_angle=heading,next_generation_frames=52,chunk_id=i,include_synthetic_generation=False)
        runner.previous=hybrid;runner.translation=center;runner.heading=heading;runner.i=i+1;receipt.append(dict(chunk=i,frames_end=(i+1)*52,latent=identity(local)))
    runner.physical=physical.clone();runner.c._completed_windows=runner.c._accepted_commits=runner.c._next_chunk_id=n//52;runner.c._last_text_feat=text;runner.c._last_text_pad_mask=runner.mask
    return runner,receipt
def payload(runner,chunks,text,control=None):
    return dict(meta=dict(depth=runner.i),chunks=copy_state(chunks,runner.m.device),heading=runner.heading.clone(),translation=runner.translation.clone(),previous=runner.previous.clone(),physical=runner.physical.clone(),target=dict(text=text,control=control or {}))
def target_check(m,world,previous,origin,expected):
    from g1_motion_contract import decode_training_prediction,CONTRACT
    target=localize(m,world,origin);prefix=localize(m,previous,origin);valid=torch.ones(1,52,device=m.device,dtype=torch.bool)
    f=dict(motion_contract=CONTRACT,hybrid_converter=m.hybrid,history_mask=~valid,generation_mask=valid,decoder_history_hybrid=prefix)
    decoded=decode_training_prediction(target,f);ph=m.motion_rep.unnormalize(decoded);ph[:,:,0]+=origin[:,None,0];ph[:,:,2]+=origin[:,None,2]
    actual=m.motion_rep.normalize(ph);a=m.motion_rep.inverse(actual,is_normalized=True);b=m.motion_rep.inverse(expected,is_normalized=True)
    root=float((a['root_positions']-b['root_positions']).abs().max());joints=float((a['posed_joints']-b['posed_joints']).norm(dim=-1).mean()*100)
    return target,prefix,dict(root_max_m=root,joints_mean_cm=joints,passed=root<=1e-5 and joints<=.3)
class OfficialSampler:
    def __init__(self,o):self.o=o;self.p=NoiseProbe(o)
    def step(self,runner,noise,history_frames=196,control=None):
        o=self.o;m=runner.m;ph=runner.physical[:,-history_frames:];history,origin,heading=o._encode_init_history(o.motion_rep.normalize(ph),1);n=ph.shape[1]
        assert float((origin-runner.translation).abs().max())<1e-5
        steps=torch.tensor([10],device=m.device);o.diffusion.calc_diffusion_vars(o.diffusion.space_timesteps(10)[0]);obs,mask=conditions(o,control,origin)
        obs=torch.cat((torch.zeros(1,n,414,device=m.device),obs),1);mask=torch.cat((torch.zeros(1,n,414,device=m.device,dtype=torch.bool),mask),1);self.p.begin(noise)
        seq=o._generate_window(history,origin,0,n,n+52,runner.text,runner.mask,heading,mask,obs,steps,(2.,2.),list(range(9,-1,-1)),progress_bar=lambda x:x,target_motion=None,cfg_type=None)
        world=world_motion(o,seq[:,-13:],origin);shape=world[:,:,:20].shape;root=world[:,:,:20].reshape(1,13,4,5);s=o.motion_rep.global_root_stats;t=m.motion_rep.global_root_stats
        world[:,:,:20]=((root*s.std_eps.to(world.device)+s.mean.to(world.device)-t.mean.to(world.device))/t.std_eps.to(world.device)).reshape(shape)
        runner.c._last_text_feat=runner.text;runner.c._last_text_pad_mask=runner.mask
        return runner.accept(localize(m,world,runner.translation))
    def close(self):self.p.close()
def command(runner,kind,event_id,state,j):
    h=float(runner.heading[0]);origin=runner.translation[0,[0,2]].cpu().tolist()
    if j==0:state=dict(heading=h,origin=origin)
    baseh=state['heading']
    if kind in ['text','release','free_resume','continue','release_hold']:return {},state,dict(kind=kind)
    if kind=='heading':return {'heading':baseh+.3},state,dict(kind=kind)
    speed={'turn':.6,'stop':0.,'restart':.6,'assisted_resume':.6,'sparse':.6,'multi':.35}.get(kind)
    if speed is None:raise ValueError(kind)
    targeth=baseh+(.3 if kind=='turn' else 0)
    ctl=dict(version=VERSION,intent='decelerate' if kind=='stop' else 'restart' if kind in ['restart','assisted_resume'] else 'move',velocity_xz=[speed*math.sin(targeth),speed*math.cos(targeth)],event_id=event_id)
    if kind!='stop':ctl['heading']=targeth
    velocity=[0.,0.] if runner.physical is None else ((runner.physical[0,-1,[0,2]]-runner.physical[0,-6,[0,2]])*5).cpu().tolist()
    if kind=='multi':
        from ._compat.absolute_goal import compile_goal,VERSION as GOAL_VERSION
        if j==0:state=dict(heading=h,goal=(np.asarray(origin)+np.array([math.sin(h),math.cos(h)])*2).tolist())
        keyboard=dict(version=VERSION,intent='move',velocity_xz=[.35*math.sin(state['heading']),.35*math.cos(state['heading'])],event_id=event_id,heading=state['heading'])
        mouse=dict(version=GOAL_VERSION,intent='absolute_target',event_id=event_id,heading=state['heading'],target_xz=state['goal'])
        if j==0:
            effective,planned,receipt=compile_intent(keyboard,None,origin,velocity,h,runner.i*52);state['keyboard_planned']=planned;priority='keyboard_velocity_over_mouse_target'
        else:
            effective,planned,receipt=compile_goal(mouse,state.get('planned'),origin,velocity,h,runner.i*52);state['planned']=planned;priority='mouse_target_after_keyboard_release'
        return effective,state,dict(receipt,requested=keyboard if j==0 else mouse,mouse_request=mouse,keyboard_request=keyboard if j==0 else None,priority=priority,keyboard_translation_effective=j==0)
    effective,planned,receipt=compile_intent(ctl,state.get('planned'),origin,velocity,h,runner.i*52);state['planned']=planned
    return effective,state,dict(receipt,target_speed=speed,desired_heading=targeth)
