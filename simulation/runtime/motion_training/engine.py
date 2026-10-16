"""Single microbatch engine, full-boundary checkpoint and strict recipe resume."""
import gc,time,math,os
from pathlib import Path
import numpy as np
import torch
from . import runtime,losses,data
from .io import Budget,read,write,append,rows,sha,identity,copy_state
from ._compat import legacy_loss
from . import distributed as parallel

def config(path):
    import yaml,json
    text=Path(path).read_text()
    try:c=json.loads(text)
    except json.JSONDecodeError:c=yaml.safe_load(text)
    for key in ['lr','eps','weight_decay','clip']:
        if key in c:c[key]=float(c[key])
    assert c['optimizer_state_policy'] in ['inherit','reset','fresh'] and c['batch_size'] in [8,10,16,32,64,128,256,512]
    if c['optimizer_state_policy']=='fresh':assert c.get('initialization_policy')=='fresh_denoiser'
    if c['loss']=='ardy_informed_v1':assert all(v==0 for v in c['extensions'].values()),'New foundation must not inherit old extensions'
    return c
def recipe_hash(cfg):return identity(cfg)
def group(n):return 'root' if n.startswith('root_model.') else 'body' if n.startswith('body_model.') else 'condition'
def norms(named):
    total={k:0. for k in ['root','body','condition']}
    for n,t in named:
        if t is not None:total[group(n)]+=float(t.detach().float().square().sum())
    return {k:math.sqrt(v) for k,v in total.items()}
def save(path,m,ck,opt,cfg,step,manifest,pool):
    rng_by_rank=parallel.gather(runtime.rng())
    rank_state_hashes=parallel.gather(identity([m.denoiser.model.state_dict(),opt.state_dict()])) if parallel.world()>1 else []
    if rank_state_hashes:assert len(set(rank_state_hashes))==1,'Distributed ranks diverged'
    if not parallel.leader():return
    path=Path(path);assert not path.exists(),path
    raw=m.denoiser.model
    state={k:v for k,v in ck.items() if k not in ['root_model','body_model','denoiser_aux','optimizer','rng_by_rank','model','framework']}
    state.update(root_model=copy_state(raw.root_model.state_dict()),body_model=copy_state(raw.body_model.state_dict()),denoiser_aux=copy_state(legacy_loss._denoiser_aux_state(raw)),optimizer=copy_state(opt.state_dict()),rng_by_rank=[runtime.rng()],step=ck.get('_framework_global_start',ck.get('step',16600))+step,framework=dict(version='framework_v1',stage_step=step,recipe_hash=recipe_hash(cfg),effective_recipe=cfg,manifest_path=str(manifest),manifest_hash=sha(manifest),pool_path=str(pool) if pool else None,pool_hash=sha(pool) if pool else None,sampler_cursor=step*cfg['batch_size'],accumulation_boundary='complete_optimizer_update_only',scheduler=None,scaler=None,optimizer_state_policy='inherit',source_imports=runtime.import_sources()),training_precision_contract='g1_root_output_geometry_fp32_v1')
    state['framework'].update(version='framework_v2',scheduler=copy_state(getattr(opt,'training_schedule',None)),optimizer_state_policy=cfg['optimizer_state_policy'],initialization_policy=cfg.get('initialization_policy','weights_new_stage' if cfg['optimizer_state_policy']=='reset' else 'carry_state_new_stage'))
    state['rng_by_rank']=rng_by_rank;state['framework']['world_size']=parallel.world()
    state['framework']['rank_state_hashes']=rank_state_hashes
    if ck.get('_resource_transition'):
        state['framework']['resource_transition']=ck['_resource_transition']
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
def build(m,c,ck,record,cfg):
    if record['supervision']=='GT':return data.build_gt(m,c,record,cfg)
    if record['supervision']=='same_state_teacher':
        from .history import teacher_batch
        return teacher_batch(m,c,record,cfg,ck['config'])
    if record['supervision']=='legacy_fixture':
        fixed=torch.load(record['path'],map_location=m.device,weights_only=False)
        fixed['hybrid_converter']=m.hybrid
        return fixed,dict(record,noise_hash=identity(fixed['noise']))
    raise ValueError(record['supervision'])
def update(m,c,ck,opt,cfg,records,step,category,budget,gradient_details=False):
    assert len(records)==cfg['batch_size'];raw=m.denoiser.model;opt.zero_grad(set_to_none=True);details=[];values=[];started=time.time();component_grads={}
    lr=runtime.set_update_lr(opt)
    before={n:p.detach().float().cpu().clone() for n,p in runtime.master_named(raw,opt)}
    phase_seconds=dict(build=0.,forward_loss=0.,backward=0.,reduce=0.,optimizer=0.)
    for lane,record in enumerate(records):
        if lane%parallel.world()!=parallel.rank():continue
        torch.cuda.synchronize();tick=time.perf_counter()
        f,rec=build(m,c,ck,record,cfg);raw.train();m.autoencoder.eval();torch.manual_seed(record['seed']);torch.cuda.manual_seed_all(record['seed'])
        torch.cuda.synchronize();phase_seconds['build']+=time.perf_counter()-tick;tick=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):value,info,terms=losses.compute(raw,c,f,cfg['loss'])
        torch.cuda.synchronize();phase_seconds['forward_loss']+=time.perf_counter()-tick
        assert torch.isfinite(value),('nonfinite',record['id'])
        if gradient_details and lane==0:
            named=list(raw.named_parameters())
            for k,term in terms.items():
                grads=torch.autograd.grad(term,[p for _,p in named],retain_graph=True,allow_unused=True)
                component_grads[k]=norms([(n,g) for (n,p),g in zip(named,grads)]);del grads
        torch.cuda.synchronize();tick=time.perf_counter()
        (value/cfg['batch_size']).backward();torch.cuda.synchronize();phase_seconds['backward']+=time.perf_counter()-tick
        values.append(float(value.detach()));details.append(dict(lane=lane,**rec,loss=info));del value,f,terms
    tick=time.perf_counter();parallel.reduce_gradients(raw);torch.cuda.synchronize();phase_seconds['reduce']=time.perf_counter()-tick
    if parallel.world()>1:
        parts=parallel.gather(dict(values=values,details=details));values=[v for part in parts for v in part['values']];details=sorted([r for part in parts for r in part['details']],key=lambda r:r['lane'])
    grads=norms([(n,p.grad) for n,p in raw.named_parameters()]);clip=float(torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg['clip'],error_if_nonfinite=True))
    if parallel.leader():budget.ticket(category,recipe=cfg['name'],step=step,world_size=parallel.world())
    parallel.barrier();tick=time.perf_counter();opt.step();opt.training_schedule['completed']+=1;torch.cuda.synchronize();phase_seconds['optimizer']=time.perf_counter()-tick
    delta=norms([(n,p.detach().float().cpu()-before[n]) for n,p in runtime.master_named(raw,opt)])
    return dict(step=step,lr=lr,scheduler_step=opt.training_schedule['completed'],loss=float(np.mean(values)),receipts=details,gradient_norms=grads,component_gradients_first_sample=component_grads,clip=clip,master_delta=delta,seconds=time.time()-started,phase_seconds_by_rank=parallel.gather(phase_seconds),world_size=parallel.world(),effective_batch=cfg['batch_size'],gradient_backend='reference')
def train(cfgpath,out,until=None,resume=None,parent=None,pool=None,category=None):
    cfg=config(cfgpath);dest=Path(out);dest.mkdir(parents=True,exist_ok=True);budget=Budget(cfg['budget_root']);until=cfg['max_updates'] if until is None else until
    assert 0<=until<=cfg['max_updates'];category=category or ('C' if cfg['name']=='closed_loop_v1' else 'B')
    if cfg.get('execution_authorization')=='PLAN_ONLY':raise RuntimeError('R3 is planned only; explicit launch authorization and frozen resource limits required')
    if (dest/'STOP.json').exists():raise RuntimeError('Branch stopped; do not resume damaged recipe')
    with parallel.timed(budget,'train:'+str(dest)):
        manifest=dest/'data/MANIFEST.json'
        if parallel.leader():data.prepare(cfg,manifest.parent)
        parallel.barrier();plan=read(manifest)
        cp=Path(resume or parent or cfg['parent'])
        fresh_seed=cfg.get('initialization_seed',cfg['seed']) if not resume and cfg.get('initialization_policy')=='fresh_denoiser' else None
        m,c,ck=runtime.load(cp,True,fresh_seed=fresh_seed);opt=runtime.optimizer(m,ck,cfg,resume=bool(resume));start=0
        if fresh_seed is None:
            parent_rank = 0
            if not resume and cfg.get('restore_parent_rank_rng'):
                assert ck['framework']['world_size']==parallel.world(), 'Parent rank RNG requires unchanged world size'
                assert len(ck['rng_by_rank'])==parallel.world()
                parent_rank=parallel.rank()
            runtime.restore_rng(ck['rng_by_rank'][parent_rank],fallback_seed=cfg['seed'] if not resume else None)
        if resume:
            saved=ck['framework']
            if saved['recipe_hash']!=recipe_hash(cfg):
                gatepath=os.environ.get('MOTION_MICROBATCH_GATE')
                assert gatepath,'Recipe changed: explicit admitted microbatch transition required'
                gate=read(gatepath);before=saved['effective_recipe']
                assert gate['passed'] and gate['kind']=='micro16_to64_same_global512'
                assert str(cp)==gate['resume_checkpoint'] and sha(cp)==gate['resume_sha256']
                assert saved['recipe_hash']==gate['old_recipe_hash'] and recipe_hash(cfg)==gate['new_recipe_hash']
                diff={k for k in set(before)|set(cfg) if before.get(k)!=cfg.get(k)}
                assert diff=={'per_rank_microbatch'} and before['per_rank_microbatch']==16 and cfg['per_rank_microbatch']==64
                assert before['batch_size']==cfg['batch_size']==512
                ck['_resource_transition']=dict(kind=gate['kind'],gate=gatepath,global_batch=512,old_micro=16,new_micro=64,optimizer_and_scheduler_inherited=True)

            saved_world=saved.get('world_size',1)
            if saved_world!=parallel.world():
                from .resource_switch import validate_transition
                transition=validate_transition(os.environ.get('MOTION_WORLD_SIZE_GATE'),cfg,cp,saved_world,parallel.world())
                ck['_resource_transition']=transition
                runtime.restore_rng(ck['rng_by_rank'][parallel.rank()%saved_world])
            else:
                runtime.restore_rng(ck['rng_by_rank'][parallel.rank()])
                if saved.get('resource_transition'):ck['_resource_transition']=saved['resource_transition']
            assert saved['manifest_hash']==sha(manifest),'Sampler plan changed'
            start=saved['stage_step'];assert len(rows(dest/'train.jsonl'))==start,'Receipt/cursor mismatch'
            if not pool:pool=saved['pool_path']
            if pool and saved['pool_path']==str(pool):assert saved['pool_hash']==sha(pool)
        ck['_framework_global_start']=ck.get('step',16600)-start
        if category=='C':
            assert pool,'Current-history pool required';pool=Path(pool);pooldata=read(pool);assert pooldata['labels']
            if not resume or ck['framework'].get('pool_path')!=str(pool):
                assert pooldata['checkpoint_sha256']==sha(cp),'Initial/refreshed pool must match the current student checkpoint'
            from .collectors import schedule_pool
            event_plan=schedule_pool(pooldata,until-start,start=start)
        if parallel.leader():write(dest/'EFFECTIVE_CONFIG.json',dict(recipe=cfg,recipe_hash=recipe_hash(cfg),parent=str(cp),parent_sha256=sha(cp),optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],imports=runtime.import_sources(),stage_start=start,world_size=parallel.world()))
        if start==0 and not (dest/'checkpoint0.pt').exists():save(dest/'checkpoint0.pt',m,ck,opt,cfg,0,manifest,pool)
        for step in range(start+1,until+1):
            records=plan['plan'][(step-1)*cfg['gt_slots']:step*cfg['gt_slots']]
            if category=='C':records+=event_plan[step-start-1]
            rec=update(m,c,ck,opt,cfg,records,step,category,budget,gradient_details=step in [1,32,64,128])
            if parallel.leader():append(dest/'train.jsonl',rec);print('UPDATE',cfg['name'],step,round(rec['loss'],5),round(rec['seconds'],2),flush=True)
            if step in cfg['savepoints'] or step==until:save(dest/f'checkpoint{step}.pt',m,ck,opt,cfg,step,manifest,pool)
            if cfg.get('boundary_request_path') and step%100==0:
                requests=parallel.gather(Path(cfg['boundary_request_path']).exists())
                if any(requests):
                    target=dest/f'checkpoint{step}.pt'
                    if not target.exists():save(target,m,ck,opt,cfg,step,manifest,pool)
                    if parallel.leader():write(dest/'STATE.json',dict(status='BOUNDARY_REQUEST',step=step,checkpoint=str(target),recipe_hash=recipe_hash(cfg),model_qualified=False))
                    parallel.barrier()
                    return str(target)
        if parallel.leader():write(dest/'STATE.json',dict(status='SEGMENT_COMPLETE',step=until,checkpoint=str(dest/f'checkpoint{until}.pt'),recipe_hash=recipe_hash(cfg),model_qualified=False))
        parallel.barrier()
    return str(dest/f'checkpoint{until}.pt')
