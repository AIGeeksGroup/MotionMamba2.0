"""Frozen legacy DEV with formal runtime imports; no historical main programs."""
import math,time,gc
from pathlib import Path
import numpy as np
import torch
from . import runtime,sampling,data
from .engine import config as load_config
from .io import read,write,rows,sha,identity,copy_state,Budget
from ._compat.event_scores import phase,event_score
from ._compat.window_score import score_window
from ._compat.motion_checks import summarize
from ._compat.semantics import semantic_features

L=Path('/data3/zts/result/g1_loss_alignment_execution_20261003')
F=Path('/data3/zts/result/g1_foundation_fusion_execution_20261002')
R=Path('/data3/zts/result/g1_joint_interaction_training_20261001')

def text_for(cfg,prompt,device='cuda'):
    try:return data.embedding(prompt,cfg['text_roots'],device)
    except FileNotFoundError:
        for item in read(R/'TEXTS.json').values():
            if item['prompt']==prompt:return torch.tensor(np.load(item['path']),device=device,dtype=torch.float32).reshape(1,1,4096)
        raise
def set_text(runner,spec,action,cfg):
    prompt=spec['prompt'] if action=='turn' else 'A person stands still.' if action=='hold' else read(R/'TEXTS.json')[action+'_0']['prompt']
    runner.text=text_for(cfg,prompt,runner.m.device)
def initialize(m,c,p,spec,cfg,captured):
    a=np.load(spec['feature_path'],mmap_mode='r')[spec['crop_start']:spec['crop_start']+104].copy()
    ph=torch.tensor(a,device=m.device)[None];offset=ph[:,:1,[0,2]].clone();ph[:,:,[0,2]]-=offset
    captured.clear();runner,_=sampling.prime_gt(m,c,p,ph,text_for(cfg,spec['prompt'],m.device));return runner,offset
def control(runner,spec,offset,action,kind,j,state):
    if kind=='gt_turn':
        a=np.load(spec['feature_path'],mmap_mode='r');start=spec['crop_start']+104+j*52;ph=a[start:start+52];assert len(ph)==52
        ctl=dict(root2d=[dict(frame=f,xz=(ph[f,[0,2]]-offset[0,0].cpu().numpy()).tolist(),heading=float(np.arctan2(ph[f,4],ph[f,3]))) for f in [25,51]])
        return ctl,state,dict(kind=kind,ground_truth_path_not_future_body=True)
    if kind in ['free_resume','release_hold','text','continue']:return {},state,dict(kind=kind,visible_resume=kind=='free_resume')
    return sampling.command(runner,{'assisted_resume':'restart'}.get(kind,kind),spec['source_family_id']+':'+{'assisted_resume':'restart'}.get(kind,kind),state,j)
def event_plan(group):
    return dict(T=[('turn','gt_turn',3)],W=[('walk','continue',20),('backwalk','text',4)],S=[('walk','stop',3),('walk','free_resume',4),('walk','stop',3),('walk','assisted_resume',4),('hold','release_hold',4)],X=[('wave','text',4),('walk','text',4),('walk','multi',4),('walk','free_resume',4)])[group]

def foundation(m,c,ck,p,cfg,dest,sentinel):
    dest.mkdir(parents=True,exist_ok=True);result=[];limits=np.asarray(read(ck['config']['joint_limit_spec'])['bounds'])
    if cfg.get('evaluation_split','DEV')=='FINAL':
        out=dict(rows=[],windows=0,function=0,joint=0,scope='FINAL interaction only; no DEV foundation counted as heldout')
        write(dest/'RESULT.json',out);return out
    contract=read(F/'EVAL_CONTRACT.json');aliases=dict(turn='dance',rhythm='wave',leg_raise='kick',bend='bow')
    for spec in contract['dev_sources']:
        if sentinel and spec['source_index']!=0:continue
        a=np.load(spec['feature_path'],mmap_mode='r')[spec['crop_start']:spec['crop_end']].copy();physical=torch.tensor(a,device=m.device)[None];physical[:,:,[0,2]]-=physical[:,:1,[0,2]].clone()
        for ni,seed in enumerate(spec['seeds']):
          for cell in contract['dev_cells']:
            key=f'{spec["action"]}_{spec["source_index"]}_{ni}_{cell}';target=dest/(key+'.json')
            if target.exists():result.append(read(target));continue
            text=text_for(cfg,spec['evaluation_text'],m.device)
            if cell!='cold_text':runner,receipt=sampling.prime_gt(m,c,p,physical[:,:104],text)
            else:runner=sampling.Native(m,c,p,text);receipt=[]
            previous=runner.physical;ctl={}
            if cell=='gt_accepted_control':ctl={'root2d':[dict(frame=f,xz=physical[0,104+f,[0,2]].cpu().tolist(),heading=float(torch.atan2(physical[0,104+f,4],physical[0,104+f,3]))) for f in [25,51]]}
            noise=sampling.fresh_noise(seed,1)[0];row,q,dec,_,_=runner.step(noise,ctl)
            met,_,_,_=summarize(m,dec,[row],aliases.get(spec['action'],spec['action']),limits,m.motion_rep.normalize(previous[:,-52:]) if previous is not None else None)
            ph=m.motion_rep.unnormalize(dec)[0].cpu().numpy();sem=semantic_features(q,spec['action'],ph)
            path=max((float(np.linalg.norm(ph[x['frame'],[0,2]]-x['xz'])) for x in ctl.get('root2d',[])),default=None)
            heading=max((float(abs(np.rad2deg(np.arctan2(np.sin(np.arctan2(ph[x['frame'],4],ph[x['frame'],3])-x['heading']),np.cos(np.arctan2(ph[x['frame'],4],ph[x['frame'],3])-x['heading']))))) for x in ctl.get('root2d',[])),default=None)
            artifact=dest/(key+'.npz');np.savez_compressed(artifact,qpos=q,physical=ph,reference_physical=physical[0,104:].cpu().numpy(),fps=25)
            rec=dict(id=key,action=spec['action'],source_index=spec['source_index'],noise_index=ni,seed=seed,source_id=spec['source_id'],family=spec['source_family_id'],cell=cell,prompt=spec['evaluation_text'],metrics=met,semantic=sem,path_error_m=path,heading_error_deg=heading,function_pass=sem['passed'],geometry_pass=not met['geometry_failures'],joint_pass=bool(sem['passed'] and not met['geometry_failures']),priming_receipt=receipt,control=ctl,noise=identity(noise),artifact=str(artifact),accepted_commits=c.controller._accepted_commits)
            write(target,rec);result.append(rec)
    out=dict(rows=result,windows=len(result),function=sum(r['function_pass'] for r in result),joint=sum(r['joint_pass'] for r in result));write(dest/'RESULT.json',out);return out

def evaluate(checkpoint,output,config,sentinel=False,foundation_only=False):
    cfg=load_config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    with Budget(cfg['budget_root']).timed('evaluation:'+str(dest),closing=True):
        started=time.time();m,c,ck=runtime.load(checkpoint,False);p=sampling.NoiseProbe(m);results=[]
        from .acceleration import admitted_backend,stats
        backend=admitted_backend(c,cfg)
        try:
          with torch.inference_mode(),sampling.capture_context() as captured:
            split=cfg.get('evaluation_split','DEV')
            indices=cfg.get('evaluation_source_indices',[10,11] if split=='FINAL' else ([3] if sentinel else [3,4]))
            specs=[] if foundation_only else [s for s in rows(L/'SPLITS.jsonl') if s['split']==split and s['source_index'] in indices]
            for spec in specs:
              for ni,seed in enumerate(spec['seeds']):
                key=f'{spec["group"]}_{spec["source_index"]}_n{ni}';path=dest/(key+'.json')
                if path.exists():results.append(read(path));continue
                runner,offset=initialize(m,c,p,spec,cfg,captured);noises=sampling.fresh_noise(seed,100)
                if spec['group']!='T':
                    set_text(runner,spec,'walk',cfg)
                    for z in noises[:4]:runner.step(z)
                qs=[];phs=[];windows=[];events=[];receipts=[];wi=0;planner_failure=None
                for ei,(action,kind,count) in enumerate(event_plan(spec['group'])):
                    set_text(runner,spec,action,cfg);state={};h=float(runner.heading[0]);startwi=wi
                    for j in range(count):
                        try:ctl,state,intent=control(runner,spec,offset,action,kind,j,state)
                        except ValueError as error:
                            if 'absolute goal exceeds bounded planner range' not in str(error):raise
                            planner_failure=dict(error=str(error),window=wi,event=ei,action=action,kind=kind,relative_window=j,
                                physical_input_hash=identity(runner.physical),interpretation='finite model state outside control planner domain; no fabricated continuation')
                            torch.save(copy_state(sampling.payload(runner,captured,runner.text,{})),dest/f'{key}_planner_failure.pt')
                            break
                        previous=runner.physical
                        item=sampling.payload(runner,captured,runner.text,ctl);z=noises[4+wi];tick=time.time()
                        row,q,dec,world,_=runner.step(z,ctl);w,ph=score_window(m,dec,q,previous,ck['config'],action,ctl)
                        if j==0 or w['metrics']['joint_limit_max_deg']>30:
                            saved=dest/f'{key}_w{wi}_accepted.pt';torch.save(copy_state(item),saved);w['accepted_input']=str(saved)
                        qs.append(q);phs.append(ph);windows.append(w);receipts.append(dict(window=wi,event=ei,relative_window=j,phase=phase(action,kind,j),action=action,kind=kind,control=ctl,intent=intent,noise_hash=identity(z),seconds=time.time()-tick));wi+=1
                    if planner_failure:
                        events.append(dict(action=action,kind=kind,checks={'planner_domain':False},cells={},function_pass=False,geometry_pass=False,all_pass=False,start_window=startwi,end_window=wi,failure=planner_failure))
                        break
                    ev=event_score(np.concatenate(qs[startwi:wi]),np.concatenate(phs[startwi:wi]),windows[startwi:wi],action,kind,h,[r['control'] for r in receipts[startwi:wi]]);ev.update(start_window=startwi,end_window=wi);events.append(ev)
                artifact=dest/(key+'.npz');np.savez_compressed(artifact,qpos=np.concatenate(qs) if qs else np.empty((0,36)),physical=np.concatenate(phs) if phs else np.empty((0,414)),fps=25)
                r=dict(id=key,source=spec,seed=seed,noise_index=ni,windows=windows,events=events,receipts=receipts,artifact=str(artifact),function_pass=bool(not planner_failure and all(v['function_pass'] for v in events)),all_pass=bool(not planner_failure and all(v['all_pass'] for v in events)),joint_peak=max((w['metrics']['joint_limit_max_deg'] for w in windows),default=0.),valid=all(w['valid'] for w in windows),planner_failure=planner_failure,rollout_complete=planner_failure is None);write(path,r);results.append(r);print('EVAL',key,'function',r['function_pass'],'all',r['all_pass'],flush=True)
            base=foundation(m,c,ck,p,cfg,dest/'foundation',sentinel)
            out=dict(complete=True,evaluation_scope='foundation_only' if foundation_only else 'full',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),split=split,sentinel=sentinel,rows=results,foundation=base,seconds=time.time()-started,metric_contract_sha256=sha(L/'METRIC_CONTRACT.json'),actual_ui_test=False,physics=False,imports=runtime.import_sources())
            out.update(execution_backend=backend,kernel_stats=stats(c));write(dest/'RESULT.json',out)
        finally:p.close()
    return out

def compare(parent,current):
    from ._compat.comparison import compare as original
    result=original(parent,current);by={r['id']:r for r in read(Path(current)/'RESULT.json')['rows']}
    def key(x):
        r=by[x['id']];w=r['receipts'][x['window']];return x['source'],r['source']['group'],w['action'],w['kind'],w['phase']
    result['confirmed_severe']=[x for x in result['severe_instances'] if any(key(x)==key(y) and x['noise']!=y['noise'] and set(x['why'])&set(y['why']) for y in result['severe_instances'])]
    result['stop']=bool(result['invalid'] or result['confirmed_severe'] or result['confirmed_old_stages_lost'] or result['basic_confirmed'])
    result['confirmation_identity']='source/task/action/control/phase across noises'
    write(Path(current)/'COMPARISON.json',result);return result

def additional(config,output):
    """Frozen six-family DEV: one absolute future goal rolls from F to G.

    This evaluates control/geometry, not unannotated caption semantics or GT
    pose reconstruction. No future body enters the input or accepted state.
    """
    from ardy.constraints import Root2DConstraintSet
    from .acceleration import admitted_backend,stats
    cfg=load_config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    manifest=read(Path(cfg['budget_root'])/'B/data/MANIFEST.json')
    candidates=read(dest/'CANDIDATES.json')
    write(dest/'PROTOCOL.json',dict(families=manifest['additional_dev'],goal_world_frame=103,
        timing=[dict(G=52,F=196,relative_goal=103),dict(G=52,F=144,relative_goal=51)],
        position_threshold_m=.2,heading_threshold_deg=20,geometry='unchanged score_window',
        semantic_quality='NOT_ANNOTATED; no caption success claim',train_family_disjoint=True,
        thresholds_scope='pre-existing DEV thresholds; new timing evaluation, not sealed test'))
    results=[]
    with Budget(cfg['budget_root']).timed('additional_future_eval',closing=True):
      for name,checkpoint in candidates.items():
        m,c,ck=runtime.load(checkpoint,False);probe=sampling.NoiseProbe(m);backend=admitted_backend(c,cfg)
        try:
          with torch.inference_mode():
            for si,spec in enumerate(manifest['additional_dev']):
              physical=np.load(spec['feature_path'],mmap_mode='r')[spec['crop_start']:spec['crop_start']+248].copy()
              physical[:,[0,2]]-=physical[0,[0,2]].copy();goal=physical[103,[0,2]].copy();heading=float(np.arctan2(physical[103,4],physical[103,3]))
              for ni,seed in enumerate(spec['seeds']):
                key=f'{name}_{si}_n{ni}';rp=dest/(key+'.json')
                if rp.exists():results.append(read(rp));continue
                runner=sampling.Native(m,c,probe,text_for(cfg,spec['caption'],m.device));qs=[];phs=[];checks=[];receipts=[]
                for wi,noise in enumerate(sampling.fresh_noise(seed,2)):
                    frame=103-wi*52;total=248-wi*52
                    constraint=Root2DConstraintSet(m.skeleton,torch.tensor([frame],device=m.device),torch.tensor(goal,device=m.device)[None],torch.tensor([heading],device=m.device))
                    obs,mask=m.motion_rep.create_conditions_from_constraints([constraint],length=total,to_normalize=False,device=m.device)
                    obs=m.motion_rep.normalize(obs)*mask
                    previous=runner.physical
                    measured,q,dec,world,_=runner.step(noise,future_observation=(obs[None],mask[None].bool()))
                    ctl={} if wi==0 else dict(root2d=[dict(frame=51,xz=goal.tolist(),heading=heading)])
                    if previous is None:
                        limits=np.asarray(read(ck['config']['joint_limit_spec'])['bounds'])
                        met,_,_,_=summarize(m,dec,[measured],'walk',limits,None)
                        check=dict(metrics=met,geometry_pass=not met['geometry_failures'],valid=bool(np.isfinite(q).all()),seam_applicable=False)
                        ph=m.motion_rep.unnormalize(dec)[0].cpu().numpy()
                    else:check,ph=score_window(m,dec,q,previous,ck['config'],'walk',ctl)
                    # No walk semantic threshold is applied to arbitrary fusion captions.
                    checks.append(check);qs.append(q);phs.append(ph)
                    receipts.append(dict(window=wi,F=total-52,relative_goal_frame=frame,goal_world=goal.tolist(),heading=heading,mask_hash=identity(mask),observed_hash=identity(obs),visible_scalars=int(mask.sum()),accepted_frames=int(runner.physical.shape[1])))
                actual=phs[-1][51];pos=float(np.linalg.norm(actual[[0,2]]-goal));angle=float(abs(np.rad2deg(np.arctan2(np.sin(np.arctan2(actual[4],actual[3])-heading),np.cos(np.arctan2(actual[4],actual[3])-heading)))))
                geometry=all(x['valid'] and x['geometry_pass'] for x in checks)
                artifact=dest/(key+'.npz');np.savez_compressed(artifact,qpos=np.concatenate(qs),physical=np.concatenate(phs),fps=25)
                row=dict(id=key,candidate=name,checkpoint=checkpoint,family=spec['family'],dataset=spec['dataset'],seed=seed,prompt=spec['caption'],path_error_m=pos,heading_error_deg=angle,control_pass=pos<=.2 and angle<=20,geometry_pass=geometry,joint_pass=geometry and pos<=.2 and angle<=20,semantic_pass=None,receipts=receipts,windows=checks,artifact=str(artifact));write(rp,row);results.append(row)
            write(dest/(name+'_BACKEND.json'),dict(backend=backend,stats=stats(c),checkpoint_sha256=sha(checkpoint)))
        finally:probe.close()
        del probe,c,m,ck;gc.collect();torch.cuda.empty_cache()
    result=dict(rows=results,summary={name:dict(n=sum(x['candidate']==name for x in results),control=sum(x['candidate']==name and x['control_pass'] for x in results),joint=sum(x['candidate']==name and x['joint_pass'] for x in results)) for name in candidates},semantic_not_evaluated=True)
    write(dest/'RESULT.json',result);print('FUTURE_EVAL',result['summary'],flush=True);return result
