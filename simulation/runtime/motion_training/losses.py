"""Versioned registry. v1 reductions are project choices, not a private trainer."""
import torch
import torch.nn.functional as F
from g1_training_precision import prediction_loss_context
from g1_motion_contract import decode_training_prediction,CONTRACT
from ._compat import legacy_loss

FORWARD_KEYS=['x','history_len','generation_len','future_len','history_mask','generation_mask','future_mask','history_token_mask','generation_token_mask','future_token_mask','text','text_mask','timestep','heading','motion_mask','observed']
def mean_valid(value,mask):
    mask=mask.expand_as(value).to(value);num=(value*mask).flatten(1).sum(1);den=mask.flatten(1).sum(1)
    return (num/den.clamp_min(1)).mean(),den
def decode_G(pred,f):
    rows=[]
    for i in range(pred.shape[0]):
        selected=pred[i:i+1,f['generation_token_mask'][i]]
        prefix=f.get('decoder_history_hybrid');prefix=None if prefix is None else prefix[i:i+1]
        valid=torch.ones(1,52,device=pred.device,dtype=torch.bool)
        rows.append(decode_training_prediction(selected,dict(motion_contract=CONTRACT,hybrid_converter=f['hybrid_converter'],history_mask=~valid,generation_mask=valid,decoder_history_hybrid=prefix)))
    return torch.cat(rows)
def components(raw,f,pred):
    gen=f['generation_token_mask'][:,:,None];values=F.smooth_l1_loss(pred.float(),f['target'].float(),beta=1,reduction='none')
    hybrid,hn=mean_valid(values,gen);rd=raw.nframe_root_dim;assert rd==20 and pred.shape[-1]==148
    root,_=mean_valid(values[:,:,:rd],gen);body,_=mean_valid(values[:,:,rd:],gen)
    dec=decode_G(pred.float(),f);target=f['supervised_motion'].float();body_slice=raw.motion_rep.body_slice
    decoded=(dec[:,:,body_slice].float()-target[:,:,body_slice]).abs().flatten(1).mean(1).mean()
    obs=torch.stack([o[g] for o,g in zip(f['observed'],f['generation_mask'])]);mask=torch.stack([o[g] for o,g in zip(f['motion_mask'],f['generation_mask'])]).bool()
    goal,gn=mean_valid((dec.float()-obs.float()).abs(),mask)
    fk=raw.motion_rep.inverse(dec,is_normalized=True)['posed_joints'][:,:,1:].float()
    positions=raw.motion_rep.inverse(dec,is_normalized=True,posed_joints_from='positions')['posed_joints'][:,:,1:].float()
    consistency=(fk-positions).norm(dim=-1).flatten(1).mean(1).mean()
    terms=dict(hybrid=hybrid,decoded_body=decoded,goal=goal,fk_consistency=consistency)
    total=sum(terms.values());details=dict(values={k:float(v.detach()) for k,v in terms.items()},hybrid_root=float(root.detach()),hybrid_body=float(body.detach()),denominators=dict(hybrid=hn.tolist(),decoded_body=target[:,:,body_slice].numel()//len(target),goal=gn.tolist(),fk=52*fk.shape[2]),units=dict(hybrid='normalized SmoothL1 beta1',decoded_body='normalized L1',goal='visible normalized explicit L1',fk_consistency='metres Euclidean'),weights={k:1 for k in terms},old_extensions_enabled=False)
    return total,details,terms
def compute(raw,client,f,version,return_graph=False):
    if version=='legacy_preserved':
        out=legacy_loss._loss(raw,client,f,True)
        return out[0],dict(values={str(i):float(v.detach()) for i,v in enumerate(out)},old_extensions_enabled=True),{}
    assert version=='ardy_informed_v1'
    if not f.get('generated_history_state_ready'):client.reset_memory()
    prediction=raw(*(f[k] for k in FORWARD_KEYS))
    with prediction_loss_context(raw,prediction):total,details,terms=components(raw,f,prediction)
    if return_graph:details['prediction']=prediction
    return total,details,terms
