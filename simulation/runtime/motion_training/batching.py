import time,copy,contextlib,json,fcntl
from pathlib import Path
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from . import data,losses,runtime,distributed as parallel
from .io import identity

ORIGINAL_DECODE=losses.decode_G
def decode_group(pred,f):
    gen=f['generation_token_mask']
    if not torch.equal(gen,gen[:1].expand_as(gen)):return ORIGINAL_DECODE(pred,f)
    selected=pred[:,gen[0]]
    valid=torch.ones(pred.shape[0],52,device=pred.device,dtype=torch.bool)
    from g1_motion_contract import decode_training_prediction,CONTRACT
    return decode_training_prediction(selected,dict(motion_contract=CONTRACT,hybrid_converter=f['hybrid_converter'],history_mask=~valid,generation_mask=valid,decoder_history_hybrid=f.get('decoder_history_hybrid')))

def corrected_build(m,c,record,cfg,audit=True):
    original=data.identity
    if not audit:data.identity=lambda x:'recorded_in_frozen_plan'
    try:f,r=data.build_gt(m,c,record,cfg)
    finally:data.identity=original
    visible=bool(f['motion_mask'][:,record['H']+52:].any())
    if record['F'] and not visible:
        f['future_mask'].zero_();f['future_token_mask'].zero_();f['future_len'].zero_()
        f['pad_mask']=f['history_mask']|f['generation_mask'];f['lengths']=f['history_len']+f['generation_len']
    r['future_visibility_contract']='mask_no_visible_future_v1'
    return f,r

def collate(items):
    fields=items[0];out={}
    for k,v in fields.items():
        vals=[f[k] for f in items]
        if torch.is_tensor(v):out[k]=torch.cat(vals,0)
        elif k=='decoder_history_hybrid':
            assert all(x is None for x in vals);out[k]=None
        else:out[k]=v
    return out

def split_field(f,i):
    return {k:v[i:i+1] if torch.is_tensor(v) else v for k,v in f.items()}

def independent_components(raw,f,pred):
    from g1_training_precision import prediction_loss_context
    values=[];details=[]
    with prediction_loss_context(raw,pred):
        for i in range(pred.shape[0]):
            value,info,_=losses.components(raw,split_field(f,i),pred[i:i+1]);values.append(value);details.append(info)
    return torch.stack(values).mean(),dict(samples=details)

class LossModule(nn.Module):
    def __init__(self,raw,c):super().__init__();self.raw=raw;self.client=c
    def forward(self,f):
        self.client.reset_memory()
        prediction=self.raw(*(f[k] for k in losses.FORWARD_KEYS))
        return independent_components(self.raw,f,prediction)

def install():losses.decode_G=ORIGINAL_DECODE
def wrap(raw,c):
    mod=LossModule(raw,c)
    if parallel.world()>1:
        mod=DistributedDataParallel(mod,device_ids=[torch.cuda.current_device()],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True,bucket_cap_mb=25)
    return mod

def update(m,c,ck,opt,cfg,records,step,category,budget,gradient_details=False):
    assert len(records)==cfg['batch_size']; global_batch=cfg['batch_size']; micro_size=cfg['per_rank_microbatch']
    raw=m.denoiser.model;raw.train();m.autoencoder.eval()
    if not hasattr(c,'_batch_loss'):c._batch_loss=wrap(raw,c)
    opt.zero_grad(set_to_none=True);lr=runtime.set_update_lr(opt);begin=time.perf_counter()
    assert parallel.world() in [1,2,4,8] and global_batch%parallel.world()==0
    width=global_batch//parallel.world(); assert width%micro_size==0;local=records[parallel.rank()*width:(parallel.rank()+1)*width]
    audit=step in [1,100,500] or step%cfg.get('audit_every',1000)==0
    receipts=[];infos=[];loss_values=[];build_seconds=0.;compute_seconds=0.
    microbatches=width//micro_size
    for micro in range(microbatches):
        tick=time.perf_counter();subset=local[micro*micro_size:(micro+1)*micro_size];items=[]
        for r in subset:
            f,rec=corrected_build(m,c,r,cfg,audit=audit);items.append(f);receipts.append(rec)
        f=collate(items);torch.manual_seed(subset[0]['seed']);torch.cuda.manual_seed_all(subset[0]['seed'])
        torch.cuda.synchronize();built=time.perf_counter();build_seconds+=built-tick
        sync=c._batch_loss.no_sync() if micro<microbatches-1 and parallel.world()>1 else contextlib.nullcontext()
        with sync:
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,info=c._batch_loss(f)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite batched objective')
            (loss/microbatches).backward()
        torch.cuda.synchronize();compute_seconds+=time.perf_counter()-built
        infos.append(info);loss_values.append(float(loss.detach()));del f,loss,items
    clip=torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg['clip'],error_if_nonfinite=True)
    opt.step();opt.training_schedule['completed']+=1;torch.cuda.synchronize();end=time.perf_counter()
    summary=dict(loss=sum(loss_values)/microbatches,info=infos if audit else None,build=build_seconds,forward_backward=compute_seconds,seconds=end-begin,slots=len(local),clip=float(clip))
    parts=parallel.gather(summary)
    detailed=parallel.gather(receipts) if audit else None
    if parallel.leader():fast_ticket(budget,category,recipe=cfg['name'],step=step,world_size=parallel.world(),sample_slots=global_batch)
    parallel.barrier();torch.cuda.synchronize();wall=time.perf_counter()-begin
    return dict(step=step,lr=lr,scheduler_step=opt.training_schedule['completed'],loss=sum(x['loss']*x['slots'] for x in parts)/global_batch,
        seconds=wall,batch_size=global_batch,per_rank_batch=micro_size,accumulation_steps=microbatches,world_size=parallel.world(),batched_forward=True,
        phases_by_rank=parts,receipts=detailed,ids=[r['id'] for r in records],audit=audit,gradient_backend='reference',future_contract='mask_no_visible_future_v1')

def fast_ticket(budget,category,**details):
    # Durable append and cached counts; reconcile once after interrupted writes.
    from .io import read,write,append,rows
    root=budget.root;ledger=root/'UPDATES.jsonl';state=root/'COUNTS.json'
    with (root/'budget.lock').open('a') as fp:
        fcntl.flock(fp,fcntl.LOCK_EX);size=ledger.stat().st_size if ledger.exists() else 0
        counts=read(state) if state.exists() else dict(ledger_bytes=-1,counts={})
        if counts['ledger_bytes']!=size:
            from collections import Counter
            counts=dict(ledger_bytes=size,counts=dict(Counter(x['category'] for x in rows(ledger))))
        limits=read(root/'LIMITS.json');assert counts['counts'].get(category,0)<limits['caps'][category]
        assert sum(v for k,v in counts['counts'].items() if k!='teacher')<limits['total_updates']
        append(ledger,dict(category=category,time=time.time(),**details));counts['counts'][category]=counts['counts'].get(category,0)+1;counts['ledger_bytes']=ledger.stat().st_size;write(state,counts)
