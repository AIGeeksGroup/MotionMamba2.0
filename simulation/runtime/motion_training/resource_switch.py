"""Explicit, gated resource-only resume; never changes the training recipe.

Activated only by an output-local RESOURCE_SWITCH.json request. The running
four-rank process is untouched. Independent probe weights are discarded.
"""
import copy
import gc
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch
from .io import read,write,rows,sha,identity

PROJECT=Path(__file__).resolve().parents[2]
LAUNCH=PROJECT/'examples/train_motion.sh'


def argument(argv,name):
    return argv[argv.index(name)+1] if name in argv else None


def recipe_config(path):
    # Match engine.config's effective scalar types before hashing the recipe.
    cfg=read(path)
    for key in ['lr','eps','weight_decay','clip']:
        if key in cfg:cfg[key]=float(cfg[key])
    return cfg


def validate_transition(path,cfg,checkpoint,old_world,new_world):
    if not path:raise ValueError('World size changed without explicit resource gate')
    gate=read(path)
    if cfg['batch_size']==32:
        assert gate['kind']=='batch32_resource_transition' and gate['passed']
        assert gate['from_ranks']==old_world==2 and gate['to_ranks']==new_world==4
        assert gate['effective_batch']==32 and cfg['gt_slots']==32
        assert gate['resume_checkpoint']==str(Path(checkpoint).resolve())
        assert gate['resume_sha256']==sha(checkpoint)
        assert gate['recipe_identity']==identity(cfg)
        assert gate['engine_sha256']==sha(Path(__file__).with_name('engine.py'))
        assert gate['batching_sha256']==sha(Path(__file__).with_name('batching.py'))
        return dict(kind='resource_only_resume',from_ranks=2,to_ranks=4,gate=str(Path(path).resolve()),
            recipe_unchanged=True,effective_batch=32,per_rank_batch=8,
            optimizer_master_scheduler_inherited=True,
            rng_policy='saved rank modulo old world; each physical sample/noise/dropout seed unchanged',
            numerical_policy='validated update tolerance, not bit-identical across world sizes')

    if gate.get('kind')=='isolated_topology_probe':
        # This authorization permits only the bounded discarded probe, never
        # the real continuation. Exercise the actual cross-world resume path.
        assert cfg['max_updates']==10 and cfg['budget_root']==gate['probe_budget']
        limits=read(Path(cfg['budget_root'])/'LIMITS.json')
        assert limits['caps']==dict(checks=21,B=0,C=0,teacher=0)
        assert limits['total_updates']==21 and limits['gpu_hours']==2
    else:
        assert gate['passed']
    assert gate['from_ranks']==old_world and gate['to_ranks']==new_world
    assert gate['effective_batch']==cfg['batch_size']==10
    assert gate['resume_checkpoint']==str(Path(checkpoint).resolve())
    assert gate['resume_sha256']==sha(checkpoint)
    assert gate['recipe_identity']==identity(cfg)
    assert gate['engine_sha256']==sha(Path(__file__).with_name('engine.py'))
    return dict(kind='resource_only_resume',from_ranks=old_world,to_ranks=new_world,gate=str(Path(path).resolve()),
        recipe_unchanged=True,optimizer_master_scheduler_inherited=True,
        rng_policy='saved rank modulo old world; per-record input/noise/dropout seeds unchanged',
        numerical_policy=('isolated probe awaiting admission' if gate.get('kind')=='isolated_topology_probe'
                          else 'cross-world admitted tolerance; not bit-identical cross-world resume'))


def values(checkpoint):
    opt=checkpoint['optimizer']
    model={**{'root_model.'+k:v for k,v in checkpoint['root_model'].items()},
           **{'body_model.'+k:v for k,v in checkpoint['body_model'].items()},**checkpoint['denoiser_aux']}
    return {str(i):opt['master_parameters'].get(str(i),model[s['name'].removeprefix('module.')])
            for i,s in enumerate(opt['compute_signature'])}


def compare_updates(parent,a,b):
    dot=aa=bb=err=0.
    pa,va,vb=values(parent),values(a),values(b)
    for key,p in pa.items():
        x=(va[key].float()-p.float()).flatten().double();y=(vb[key].float()-p.float()).flatten().double()
        dot+=float(x@y);aa+=float(x@x);bb+=float(y@y);err+=float((x-y).square().sum())
    return dict(master_update_cosine=dot/math.sqrt(max(aa*bb,1e-60)),master_update_relative_l2=math.sqrt(err/max(aa,1e-30)))


def invoke(args,log,ranks,devices,transition_gate=None):
    env=dict(os.environ,MOTION_SKIP_RESOURCE_SWITCH='1',MOTION_NPROC=str(ranks),CUDA_VISIBLE_DEVICES=devices)
    env.pop('MOTION_WORLD_SIZE_GATE',None)
    if transition_gate:env['MOTION_WORLD_SIZE_GATE']=str(transition_gate)
    cmd=['bash',str(LAUNCH),*map(str,args)]
    write(Path(log).with_suffix('.command.json'),dict(command=cmd,ranks=ranks,devices=devices))
    with Path(log).open('a') as stream:
        process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            code=process.wait(timeout=600)
            if code:raise subprocess.CalledProcessError(code,cmd)
        except BaseException:
            # Kill only this probe's process group, including torchrun workers.
            # An unrelated evaluator and the parent tmux must remain alive.
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
            raise


def gate_probe(request,argv):
    root=Path(request['probe_root']);root.mkdir(parents=True,exist_ok=True)
    result=root/'RESULT.json'
    if result.exists():
        existing=read(result)
        assert existing['engine_sha256']==sha(Path(__file__).with_name('engine.py')),'Stale resource gate: engine changed'
        return existing
    cfg=recipe_config(argument(argv,'--config'));cp=Path(argument(argv,'--resume')).resolve()
    assert shutil.disk_usage(root).free>12*cp.stat().st_size+15*1024**3,'Insufficient probe/checkpoint reserve'
    parent=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    assert parent['framework']['world_size']==4
    manifest=read(parent['framework']['manifest_path']);start=parent['framework']['stage_step']
    subset=copy.deepcopy(manifest);subset['plan']=manifest['plan'][start*10:start*10+100]
    assert len(subset['plan'])==100
    subset['probe_subset']=dict(original_manifest=str(parent['framework']['manifest_path']),
        original_stage_step=start,original_metadata_describes_full_plan=True,actual_exposures=100)
    write(root/'MANIFEST.json',subset)
    probe_cfg=dict(cfg,parent=str(cp),initialization_policy='carry_state_new_stage',optimizer_state_policy='inherit',
        frozen_manifest=str(root/'MANIFEST.json'),budget_root=str(root/'budget'),max_updates=10,savepoints=[0,1,2,10])
    write(root/'config.json',probe_cfg)
    write(root/'budget/LIMITS.json',dict(caps=dict(checks=21,B=0,C=0,teacher=0),total_updates=21,gpu_hours=2))
    write(root/'PROTOCOL.json',dict(question='Does8rank preserve the update contract and accelerate identical next100 samples?',
        parent=str(cp),parent_sha256=sha(cp),original_stage_step=start,global_batch=10,
        actual_probe_updates=21,probe_weights_never_promoted=True,
        thresholds=request['thresholds'],reuse='Existing4rank/FP32/codec tests are not repeated; new8rank topology only',
        samples=[dict(id=r['id'],seed=r['seed'],H=r['H'],F=r['F']) for r in subset['plan']],
        rank_workloads=dict(four=[3,3,2,2],eight=[2,2,1,1,1,1,1,1]),
        speed='median updates3to10, same sample sequence; excludes startup/checkpoint',
        evaluator='Independent GPU4 process retained; timings reflect actual shared resource availability'))
    # All rank numbers are local to the explicit visible-device list.
    folder=root/'four'
    if not (folder/'checkpoint10.pt').exists():
        invoke(['train','--config',root/'config.json','--output',folder,'--until',10,'--category','checks'],root/'four.log',4,'0,1,2,3')
    probe_parent=root/'four/checkpoint0.pt'
    authorization=root/'PROBE_AUTHORIZATION.json'
    write(authorization,dict(kind='isolated_topology_probe',probe_budget=probe_cfg['budget_root'],
        from_ranks=4,to_ranks=8,effective_batch=10,resume_checkpoint=str(probe_parent.resolve()),
        resume_sha256=sha(probe_parent),recipe_identity=identity(probe_cfg),
        engine_sha256=sha(Path(__file__).with_name('engine.py'))))
    if not (root/'eight/checkpoint10.pt').exists():
        invoke(['train','--config',root/'config.json','--output',root/'eight','--until',10,
                '--resume',probe_parent,'--category','checks'],root/'eight.log',8,'0,1,2,3,4,5,6,7',authorization)
    zero_keys=['root_model','body_model','denoiser_aux','optimizer','step']
    baseline={k:identity(parent[k]) for k in zero_keys}
    preserved={}
    for name in ['four','eight']:
        zero=torch.load(root/name/'checkpoint0.pt',map_location='cpu',weights_only=False,mmap=True)
        preserved[name]={k:identity(zero[k])==baseline[k] for k in zero_keys}
        preserved[name]['scheduler']=zero['framework']['scheduler']==parent['framework']['scheduler']
        del zero;gc.collect()
    # Verify8rank actual1+1resume equals the continuous2 update already saved.
    split=root/'eight_split';split.mkdir(exist_ok=True)
    if not (split/'checkpoint1.pt').exists():os.link(root/'eight/checkpoint1.pt',split/'checkpoint1.pt')
    if not (split/'data/MANIFEST.json').exists():write(split/'data/MANIFEST.json',read(root/'eight/data/MANIFEST.json'))
    if not (split/'train.jsonl').exists():
        (split/'train.jsonl').write_text((root/'eight/train.jsonl').read_text().splitlines()[0]+'\n')
    if not (split/'checkpoint2.pt').exists():
        invoke(['train','--config',root/'config.json','--output',split,'--until',2,'--resume',split/'checkpoint1.pt','--category','checks'],root/'resume.log',8,'0,1,2,3,4,5,6,7')
    ca=torch.load(root/'four/checkpoint1.pt',map_location='cpu',weights_only=False,mmap=True)
    cb=torch.load(root/'eight/checkpoint1.pt',map_location='cpu',weights_only=False,mmap=True)
    updates=compare_updates(parent,ca,cb)
    ar,br=rows(root/'four/train.jsonl'),rows(root/'eight/train.jsonl')
    exact_samples=len(ar)==len(br)==10 and all(len(r['receipts'])==10 for r in ar+br) and all(all(x[k]==y[k] for k in ['id','seed','input_hash','target_hash','noise_hash','physical_G_hash'])
        for a,b in zip(ar,br) for x,y in zip(a['receipts'],b['receipts']))
    lossdiff=abs(ar[0]['loss']-br[0]['loss'])
    del ca,cb,parent;gc.collect()
    ca=torch.load(root/'eight/checkpoint2.pt',map_location='cpu',weights_only=False,mmap=True)
    cb=torch.load(split/'checkpoint2.pt',map_location='cpu',weights_only=False,mmap=True)
    resumed={key:identity(ca[key])==identity(cb[key]) for key in ['root_model','body_model','denoiser_aux','optimizer','rng_by_rank','step']}
    resumed['scheduler']=ca['framework']['scheduler']==cb['framework']['scheduler']
    resumed['cursor']=ca['framework']['sampler_cursor']==cb['framework']['sampler_cursor']==20
    hashes=ca['framework']['rank_state_hashes'];rank_equal=len(hashes)==8 and len(set(hashes))==1
    four=statistics.median(r['seconds'] for r in ar[2:]);eight=statistics.median(r['seconds'] for r in br[2:])
    thresholds=request['thresholds']
    numerical=all(all(v.values()) for v in preserved.values()) and exact_samples and lossdiff<=thresholds['loss_absolute_tolerance'] and updates['master_update_cosine']>=thresholds['master_update_cosine_min'] and updates['master_update_relative_l2']<=thresholds['master_update_relative_l2_max'] and all(resumed.values()) and rank_equal
    out=dict(passed=numerical and eight<four*thresholds['speed_ratio_max'],numerical_passed=numerical,
        sample_hashes_exact=exact_samples,first_loss_absolute_difference=lossdiff,**updates,
        parent_state_preserved=preserved,eight_resume_exact=resumed,eight_ranks_equal=rank_equal,four_seconds=four,eight_seconds=eight,
        speedup=four/eight,from_ranks=4,to_ranks=8,effective_batch=10,
        resume_checkpoint=str(cp),resume_sha256=sha(cp),recipe_identity=identity(cfg),
        engine_sha256=sha(Path(__file__).with_name('engine.py')),actual_probe_updates=21)
    write(result,out);return out


def dispatch(argv):
    out=Path(argument(argv,'--output'));requestfile=out/'RESOURCE_SWITCH.json'
    request=read(requestfile);cfg=recipe_config(argument(argv,'--config'));cp=argument(argv,'--resume')
    assert Path(argument(argv,'--config')).resolve()==Path(request['config']).resolve()
    assert request['status'] in ['REQUESTED_AT_NEXT_BOUNDARY','EIGHT_ACTIVE','FOUR_RETAINED']
    env=dict(os.environ,MOTION_SKIP_RESOURCE_SWITCH='1',MOTION_NPROC='4',CUDA_VISIBLE_DEVICES='0,1,2,3')
    env.pop('MOTION_WORLD_SIZE_GATE',None)
    if cp and request['status']!='FOUR_RETAINED':
        try:result=gate_probe(request,argv)
        except Exception as error:
            import traceback
            result=dict(passed=False,error=str(error),traceback=traceback.format_exc())
            write(Path(request['probe_root'])/'FAILURE.json',result)
        if result['passed']:
            saved=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
            before=saved['framework']['world_size'];start=saved['framework']['stage_step'];del saved;gc.collect()
            # Keep the72GPUh main cap. Recompute for8ranks, including probe cost.
            used=sum(r['seconds']*r.get('devices',1) for r in rows(Path(cfg['budget_root'])/'GPU_TIME.jsonl'))/3600
            probe_cost=sum(r['seconds']*r.get('devices',1) for r in rows(Path(request['probe_root'])/'budget/GPU_TIME.jsonl'))/3600
            predicted=(int(argument(argv,'--until'))-start)*result['eight_seconds']*8/3600
            # Check the complete remaining registered tranche before changing
            # topology, so a slower-than-linear speedup cannot exhaust its cap.
            remaining=(cfg['max_updates']-start)*result['eight_seconds']*8/3600
            eight_budget_ok=used+probe_cost+remaining<read(Path(cfg['budget_root'])/'LIMITS.json')['gpu_hours']*.75
            if eight_budget_ok:
                env.update(MOTION_NPROC='8',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
                if before==4:env['MOTION_WORLD_SIZE_GATE']=str(Path(request['probe_root'])/'RESULT.json')
                else:assert before==8
                request.update(status='EIGHT_ACTIVE',gate=str(Path(request['probe_root'])/'RESULT.json'),speedup=result['speedup'])
                request.update(estimated_segment_training_seconds=predicted*3600/8,
                    estimated_training_end_epoch=time.time()+predicted*3600/8,
                    used_main_gpu_hours=used,probe_gpu_hours=probe_cost,projected_segment_gpu_hours=predicted,
                    projected_remaining_training_gpu_hours=remaining)
            elif before==8:
                raise RuntimeError('Eight-rank next segment exceeds remaining registered compute reserve; keep checkpoint, do not silently expand budget')
            else:
                request.update(status='FOUR_RETAINED',reason='Eight-rank remaining tranche exceeds72GPUh reserve',gate=str(Path(request['probe_root'])/'RESULT.json'))
        else:
            request.update(status='FOUR_RETAINED',reason='Eight-rank numerical/resume/speed gate did not pass',result=result)
    write(requestfile,request)
    write(out/'RESOURCE_LAUNCH.json',dict(time=time.time(),argv=argv,actual_ranks=int(env.get('MOTION_NPROC','1')),
        devices=env.get('CUDA_VISIBLE_DEVICES'),gate=env.get('MOTION_WORLD_SIZE_GATE'),request=request))
    os.execvpe('bash',['bash',str(LAUNCH),*argv],env)


if __name__=='__main__':dispatch(sys.argv[1:])
