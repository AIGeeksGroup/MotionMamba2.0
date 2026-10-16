"""Family isolated physical GT with explicit H/G/F and exact text features."""
import collections, hashlib, math
from pathlib import Path
import numpy as np
import torch
from ardy_data.humanml3d import sample_control_condition
from ardy.model.ardy_model import get_three_mask_from_len
from g1_motion_contract import CONTRACT
from .schema import VERSION,TimeLayout,validate_record
from .io import read,write,rows,identity,sha
from ._compat.fusion_index import excluded,caption_task_tags

BUCKETS=[(0,0),(52,0),(196,0),(0,196),(52,144),(144,52)]
CONDITIONS=['none','root_trajectory','root_waypoint','full_keyframe','joint_position','joint_rotation']
def text_key(t):return hashlib.sha256(t.encode()).hexdigest()
def embedding(text,roots,device='cuda'):
    for root in roots:
        p=Path(root)/(text_key(text)+'.npy')
        if p.exists():
            a=np.load(p);assert a.size==4096 and np.isfinite(a).all();return torch.tensor(a,device=device,dtype=torch.float32).reshape(1,1,4096)
    raise FileNotFoundError('Missing exact text: '+text)
def intervals(r,n):
    if r['frames']<n:return []
    spans=r.get('caption_time_spans')
    if not spans:return [(0,r['frames']-n,t,None) for t in r.get('captions',[]) if t.strip()]
    out=[];origin=r.get('native_start_sec',0)
    for s in spans:
        lo=max(0,math.ceil((s['native_start_sec']-origin)*25-1e-6));hi=min(r['frames']-n,math.ceil((s['native_end_sec']-origin)*25-1e-6)-n)
        if lo<=hi and s['text'].strip():out.append((lo,hi,s['text'],s))
    return out
def weighted_cycle(values,counts,n,rng):
    out=[]
    while len(out)<n:
        cycle=[v for v,c in zip(values,counts) for _ in range(c)];rng.shuffle(cycle);out+=cycle
    return out[:n]
def prepare(cfg,dest):
    assert [(x['H'],x['F'],x['p']) for x in cfg['HGF']]==[(0,0,.10),(52,0,.25),(196,0,.20),(0,196,.10),(52,144,.20),(144,52,.15)]
    assert cfg['source_ratios']==dict(bones=.7,humanml3d=.15,hiphi=.15) and cfg['empty_spatial']==.2
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'MANIFEST.json').exists():
        result=read(dest/'MANIFEST.json')
        assert len(result['plan'])>=cfg['max_updates']*cfg['gt_slots'],'Manifest too short'
        if cfg.get('frozen_manifest'):assert sha(dest/'MANIFEST.json')==sha(cfg['frozen_manifest'])
        return result
    if cfg.get('frozen_manifest'):
        result=read(cfg['frozen_manifest'])
        assert result['index_sha256']==sha(cfg['data_index']) and result['exclusions_sha256']==sha(cfg['exclusions'])
        assert len(result['plan'])>=cfg['max_updates']*cfg['gt_slots']
        write(dest/'MANIFEST.json',result);assert sha(dest/'MANIFEST.json')==sha(cfg['frozen_manifest'])
        return result
    records=rows(cfg['data_index']);exclude=read(cfg['exclusions']);rng=np.random.default_rng(cfg['seed']);groups={};devfamilies=set()
    # Dedicated additional DEV families selected before either candidate exists.
    for source in ['bones','humanml3d','hiphi']:
        candidates=sorted({r['source_family_id'] for r in records if r['dataset']==source and r['project_split']=='train' and not excluded(r,exclude) and intervals(r,248)})
        assert len(candidates)>10
        devfamilies.update(candidates[:2])
    for source in ['bones','humanml3d','hiphi']:
      for h,f in BUCKETS:
        by=collections.defaultdict(list)
        for r in records:
            if r['dataset']!=source or r['project_split']!='train' or excluded(r,exclude) or r['source_family_id'] in devfamilies:continue
            if intervals(r,h+52+f):by[r['source_family_id']].append(r)
        assert by,('empty bucket',source,h,f);groups[source,h,f]=by
    n=cfg['max_updates']*cfg['gt_slots'];sources=weighted_cycle(['bones','humanml3d','hiphi'],[14,3,3],n,rng)
    hf=weighted_cycle(BUCKETS,[2,5,4,2,4,3],n,rng);conditions=weighted_cycle(CONDITIONS,[5,4,4,4,4,4],n,rng)
    # Sorting the same family keys per sample becomes prohibitive at500k
    # exposures. Cache the identical ordered keys; RNG calls/order are unchanged.
    family_keys={key:sorted(group) for key,group in groups.items()}
    plan=[];f_nonempty=0
    for i,(source,(h,f),condition) in enumerate(zip(sources,hf,conditions)):
        group=groups[source,h,f];family=family_keys[source,h,f][int(rng.integers(len(group)))];rr=group[family];r=rr[int(rng.integers(len(rr)))];opts=intervals(r,h+52+f);lo,hi,caption,span=opts[int(rng.integers(len(opts)))];start=int(rng.integers(lo,hi+1))
        isfuture=False
        if f and condition!='none':isfuture=(f_nonempty%2==0);f_nonempty+=1
        plan.append(validate_record(dict(schema=VERSION,id=f'{cfg["name"]}:{i}',source_family_id=family,dataset=source,split='TRAIN',feature_path=r['feature_path'],sample_id=r['sample_id'],crop_start=start,crop_end=start+h+52+f,caption=caption,caption_span=span,caption_tags=caption_task_tags(caption),native_start_sec=r.get('native_start_sec',0),supervision='GT',H=h,F=f,G=52,fps=25,skeleton='g1',representation='physical414',normalizer_version=CONTRACT,codec_version='g1_history_codec_104_v1',condition=condition,out_of_horizon=isfuture,seed=int(cfg['seed']+1000+i),quality_status=r.get('status'),retarget_quality=r.get('kinematic_status'),source_motion_id=r.get('source_motion_id'),valid_frames=h+52+f,earlier_prefix_frames=0,horizontal_rotation_rad=0.,augmentation='identity; world translation applied jointly to motion and conditions')))
    dev=[]
    for family in sorted(devfamilies):
        r=next(r for r in records if r['source_family_id']==family and intervals(r,248));lo,hi,caption,span=intervals(r,248)[0]
        dev.append(dict(family=family,dataset=r['dataset'],feature_path=r['feature_path'],crop_start=lo,caption=caption,caption_span=span,H=0,F=196,G=52,seeds=[202610034401,202610034402],condition='root_waypoint',constraint_frame=103))
    info=dict(schema=VERSION,recipe=cfg['name'],plan=plan,additional_dev=dev,dev_families=sorted(devfamilies),index_sha256=sha(cfg['data_index']),exclusions_sha256=sha(cfg['exclusions']),planned_exposures=n,unique_families=len({r['source_family_id'] for r in plan}),source_counts=dict(collections.Counter(sources)),HGF_counts={str(k):v for k,v in collections.Counter(hf).items()},conditions=dict(collections.Counter(conditions)),future_nonempty=f_nonempty,future_observed_before_drop=sum(r['out_of_horizon'] for r in plan),future_predictions_supervised=False,project_choices=True)
    write(dest/'MANIFEST.json',info);return info

def masks(m,h,f,device):
    total=h+52+f;hl=torch.tensor([h],device=device);gl=torch.tensor([52],device=device);fl=torch.tensor([f],device=device)
    hm,gm,fm=get_three_mask_from_len(hl,gl,fl,total,device)
    return hl,gl,fl,hm,gm,fm

def build_gt(m,c,record,cfg,physical_override=None,observed_override=None):
    r=validate_record(record);h,f=r['H'],r['F'];layout=TimeLayout(h,f).validate();device=m.device
    a=np.load(r['feature_path'],mmap_mode='r')[r['crop_start']:r['crop_end']].copy() if physical_override is None else np.array(physical_override,copy=True)
    assert a.shape==(layout.frames,414) and a.dtype==np.float32 and np.isfinite(a).all()
    physical=torch.tensor(a,device=device)[None];origin=physical[:,max(h-1,0),[0,2]].clone();physical[:,:,0]-=origin[:,None,0];physical[:,:,2]-=origin[:,None,1]
    rng=np.random.default_rng(r['seed']);cond=sample_control_condition(physical[0],None,rng=rng,horizon=52,generation_start=h,control=r['condition'] if r['condition']!='none' else 'root_waypoint',out_of_horizon=r['out_of_horizon'],condition_dropout=0)
    mask=cond['motion_mask'][None];observed=m.motion_rep.normalize(cond['observed_motion'][None])*mask
    if r['condition']=='none':mask.zero_();observed.zero_()
    if observed_override is not None:observed,mask=observed_override;observed=observed.clone();mask=mask.clone()
    pre=bool(mask[:,h+52:].any());textdrop=bool(rng.random()<cfg['text_dropout']);controldrop=bool(rng.random()<cfg['constraint_dropout'])
    if controldrop:mask.zero_();observed.zero_()
    text=embedding(r['caption'],cfg['text_roots'],device)
    if textdrop:text.zero_()
    # Encode accepted H alone, and G with <=52 physical history. F truth is never encoded.
    from g1_bounded_codec import encode_windows
    with torch.no_grad():
        known=m.motion_rep.normalize(physical[:,:h+52]);hp=None
        if h:
            pad=torch.ones(1,h,device=device,dtype=torch.bool);ht,_=encode_windows(m.hybrid,known[:,:h],pad.sum(1),pad)
            hp=ht[:,-13:]
        start=max(0,h-52);context=known[:,start:h+52];pad=torch.ones(context.shape[:2],device=device,dtype=torch.bool)
        targetG,_=m.hybrid.get_hybrid_motion_from_explicit(context,pad.sum(1),pad);targetG=targetG[:,-13:]
        clean=torch.zeros(1,layout.frames//4,148,device=device)
        if h:clean[:,:h//4]=ht
        clean[:,h//4:h//4+13]=targetG
    hl,gl,fl,hm,gm,fm=masks(m,h,f,device);htm=hm.reshape(1,-1,4).all(-1);gtm=gm.reshape(1,-1,4).all(-1);ftm=fm.reshape(1,-1,4).all(-1)
    m.diffusion.calc_diffusion_vars(torch.arange(m.diffusion.num_base_steps,device=device))
    gen=torch.Generator(device=device).manual_seed(r['seed']);t=torch.randint(m.diffusion.num_base_steps,(1,),generator=gen,device=device);noise=torch.randn(clean.shape,generator=gen,device=device)
    x=torch.zeros_like(clean);x[htm]=clean[htm];x[gtm]=m.diffusion.q_sample(clean,t,noise)[gtm]
    c.reset_memory();m.diffusion.calc_diffusion_vars(torch.arange(m.diffusion.num_base_steps,device=device))
    fixed=dict(x=x,target=clean,text=text,text_mask=torch.tensor([[not textdrop]],device=device),timestep=t,noise=noise,heading=torch.atan2(physical[:,max(h-1,0),4],physical[:,max(h-1,0),3]),history_len=hl,generation_len=gl,future_len=fl,history_mask=hm,generation_mask=gm,future_mask=fm,history_token_mask=htm,generation_token_mask=gtm,future_token_mask=ftm,observed=observed,motion_mask=mask.float(),hybrid_converter=m.hybrid,motion_contract=CONTRACT,pad_mask=torch.ones_like(hm),lengths=hl+gl+fl,decoder_history_hybrid=hp,supervised_motion=known[:,h:h+52],sequential_cache=False)
    rec=dict(r,text_dropped=textdrop,condition_dropped=controldrop,F_nonzero=f>0,F_before_drop=pre,F_visible_after_drop=bool(mask[:,h+52:].any()),observed_scalars=int(mask.sum()),G_goal_scalars=int(mask[:,h:h+52].sum()),effective_G_frames=52,effective_G_tokens=13,latent_codec_context_frames=min(h,52)+52,decoder_history_frames=min(h,52),target_hash=identity(targetG),input_hash=identity([x,text,observed,mask]),noise_hash=identity(noise),physical_G_hash=identity(physical[:,h:h+52]),earlier_prefix_primed=0)
    return fixed,rec
