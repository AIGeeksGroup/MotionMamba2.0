"""Registered startpoint experiments, using the shared trainer only.

This module owns orchestration/artifacts, never an alternative optimizer loop.
"""
import collections, copy, gc, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
from .io import read, write, rows, sha, identity, Budget

P=Path(__file__).resolve().parents[2]
Z=Path('/data3/zts')
D=Z/'result/g1_startpoint_and_learning_execution_20261004'
F=Z/'result/g1_training_framework_execution_20261003'
M=Z/'result/g1_master_experiment_roadmap_20261003'
R=M/'execution_v1_1/R2'
P0=Z/'result/g1_joint_interaction_training_20261001/joint_interaction/train/checkpoint_additional_002000.pt'
N0=Path('/data4/zts/result/g1_native_fp32_master_20260920/native/checkpoint_step_050000.pt')
LAUNCH=P/'examples/train_motion.sh'

def run(command,args,log,devices='0',ranks=1):
    cmd=['bash',str(LAUNCH),command,*map(str,args)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,MOTION_NPROC=str(ranks))
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    write(log.with_suffix('.command.json'),dict(command=cmd,devices=devices,ranks=ranks,
        formal_source_sha256={name:sha(P/'src/motion_training'/name) for name in ['engine.py','runtime.py','data.py','evaluation.py','startpoint.py']}))
    with log.open('a') as stream:
        subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def prepare():
    from .data import prepare as data_prepare
    D.mkdir(parents=True,exist_ok=True)
    if (D/'PREPARED.json').exists():return read(D/'PREPARED.json')
    old=read(M/'R2_P0_B/EFFECTIVE_CONFIG.json')['recipe']
    cfg=dict(old,version='startpoint_route_v1',max_updates=500,savepoints=[0,8,32,128,250,500],
        optimizer_state_policy='reset',initialization_policy='weights_new_stage',
        betas=[.9,.999],eps=1e-8,weight_decay=.01,
        scheduler=dict(kind='constant',lr=3e-6),budget_root=str(R),
        kernel_gate_path=str(F/'kernel/GATE.json'))
    cfg['text_roots']=[str(D/'text'),*cfg['text_roots']];(D/'text').mkdir(exist_ok=True)
    limits=read(R/'LIMITS.json');write(D/'R2_LIMITS_BEFORE.json',limits)
    limits['caps'].update(B=2000,C=1000);write(R/'LIMITS.json',limits)
    write(D/'fresh_budget/LIMITS.json',dict(caps=dict(B=5000,C=0,teacher=0,checks=0),total_updates=5000,gpu_hours=48))
    generated=data_prepare(cfg,D/'generated_tail')
    prefix=read(M/'R2_P0_B/data/MANIFEST.json');assert len(prefix['plan'])==1280
    assert all(prefix[k]==generated[k] for k in ['index_sha256','exclusions_sha256','dev_families'])
    plan=copy.deepcopy(generated);plan['plan']=prefix['plan']+generated['plan'][1280:]
    rr=plan['plan'];assert len(rr)==5000 and len({r['id'] for r in rr})==5000
    plan.update(unique_families=len({r['source_family_id'] for r in rr}),
        source_counts=dict(collections.Counter(r['dataset'] for r in rr)),
        HGF_counts=dict(collections.Counter(str((r['H'],r['F'])) for r in rr)),
        conditions=dict(collections.Counter(r['condition'] for r in rr)),
        future_nonempty=sum(r['F']>0 and r['condition']!='none' for r in rr),
        future_observed_before_drop=sum(r['out_of_horizon'] for r in rr),
        historical_prefix=dict(path=str(M/'R2_P0_B/data/MANIFEST.json'),sha256=sha(M/'R2_P0_B/data/MANIFEST.json'),exposures=1280,exact=True))
    write(D/'shared_manifest.json',plan)
    cfg['frozen_manifest']=str(D/'shared_manifest.json')
    for name,parent in [('P_reset',P0),('N_reset',N0)]:
        write(D/'configs'/f'{name}.json',dict(cfg,parent=str(parent),experiment_id=name))
    fresh=dict(cfg,optimizer_state_policy='fresh',initialization_policy='fresh_denoiser',
        initialization_seed=2026100401,parent=str(P0),max_updates=5000,
        savepoints=[0,100,500,1000,2000,3500,5000],lr=5e-5,
        scheduler=dict(kind='warmup_cosine',start=1e-6,peak=5e-5,minimum=1e-5,warmup=500,total=50000),
        budget_root=str(D/'fresh_budget'),seed=2026100401)
    fresh.pop('frozen_manifest');write(D/'configs/Fresh.json',fresh)
    write(D/'EVAL_POLICY.json',dict(version='startpoint_v1_frozen',
        original_thresholds=cfg['phase_stop'],contact='predicted contact proxy, not physics truth',
        threshold_change=False,calibration_scope='existing P0/B8/C8, qualified teacher and GT; sensitivity only',
        warm_stop='confirmed harm relative own initialization; P0 protection separately blocks replacement',
        fresh_stop='nonfinite/update-contract error; confirmed loss of previously stable learned capabilities, not initial P0 deficit',
        identity='same source/task/action/control/phase across two noises',
        candidate='same foundation id becomes joint-pass at two adjacent saves; no old joint/function pass lost; no relative harm',
        expansion='same candidate broad retention + interaction joint gain, independent repeat and preregistered new-family test'))
    write(D/'DECISIONS.json',dict(version='startpoint_route_v1',
        branches=['P_reset<=500','N_reset<=500','eligible warm->current history<=492','no eligible warm->fresh<=5000'],
        initial_state_compare='P/N reset moments; frozen masters; P vs historical B exact prefix and reference backend single GPU',
        candidate_selection='persistent new foundation joint IDs with no old joint/function IDs lost, own-stop false; P tie priority',
        policy_frozen_before_training=True,production_replacement=False,formal_50k=False))
    write(D/'EXPERIMENT_LEDGER.json',dict(version='startpoint_route_v1',
        reused_done=['migration','FP32/codec/statistics','four-rank equivalence','81 labels31 stages','H52/H196 AB','kernel no-grad equivalence','15 Adam attribution probes'],
        experiments={n:dict(status='PREPARED',checkpoint=str(p),variable='initial weights and reset Adam; matched P0 comparison resets only moments',config=str(D/'configs'/f'{n}.json')) for n,p in [('P_reset',P0),('N_reset',N0)]}))
    result=dict(status='PREPARED',prefix_matched=True,exposures=5000,
        historical_single_rank=True,warm_training_ranks=1,warm_reason='preserve exact old B reduction order; two independent GPUs concurrently',
        fresh_training_ranks=4,production_unchanged=True)
    write(D/'PREPARED.json',result);return result

def verify():
    """Two policies × (continuous2 vs1+1) = exactly8 real updates."""
    if (D/'verification/RESULT.json').exists():return read(D/'verification/RESULT.json')
    results=[]
    for policy in ['P_reset','Fresh']:
        cfg=read(D/'configs'/f'{policy}.json')
        cfg.update(max_updates=2,savepoints=[0,1,2],budget_root=str(R),frozen_manifest=str(D/'shared_manifest.json'))
        path=D/'verification'/policy/'config.json';write(path,cfg)
        for name,until,resume in [('continuous',2,None),('split',1,None),('split',2,True)]:
            out=path.parent/name;cp=out/f'checkpoint{until}.pt'
            if cp.exists():continue
            args=['--config',path,'--output',out,'--until',until,'--category','B']
            if resume:args+=['--resume',out/'checkpoint1.pt']
            run('train',args,path.parent/f'{name}{until}.log')
        a=torch.load(path.parent/'continuous/checkpoint2.pt',map_location='cpu',weights_only=False,mmap=True)
        b=torch.load(path.parent/'split/checkpoint2.pt',map_location='cpu',weights_only=False,mmap=True)
        checks={k:identity(a[k])==identity(b[k]) for k in ['root_model','body_model','denoiser_aux','optimizer','rng_by_rank','step']}
        checks['scheduler']=a['framework']['scheduler']==b['framework']['scheduler']
        zero=torch.load(path.parent/'continuous/checkpoint0.pt',map_location='cpu',weights_only=False,mmap=True)
        parent=torch.load(cfg['parent'],map_location='cpu',weights_only=False,mmap=True)
        if policy=='P_reset':
            checks['compute_unchanged']=all(identity(zero[k])==identity(parent[k]) for k in ['root_model','body_model','denoiser_aux'])
            checks['masters_unchanged']=identity(zero['optimizer']['master_parameters'])==identity(parent['optimizer']['master_parameters'])
            checks['moments_empty']=not zero['optimizer']['state']
        else:
            checks['fresh_declared']=zero['initialization']['official_denoiser_loaded'] is False
            checks['fresh_root_body_aux_differ']=all(identity(zero[k])!=identity(parent[k]) for k in ['root_model','body_model','denoiser_aux'])
            checks['fresh_moments_empty']=not zero['optimizer']['state']
            write(D/'INITIALIZATION_CONTRACT.json',dict(reset='preserve compute and FP32 masters, clear Adam state',
                fresh=zero['initialization'],resume='full state exact',carry='new recipe preserves masters moments scheduler',
                scheduler=zero['framework']['scheduler'],fresh_parameter_count=len(zero['initialization']['parameter_sources'])))
        result=dict(policy=policy,checks=checks,passed=all(checks.values()))
        results.append(result);write(path.parent/'RESULT.json',result)
        assert result['passed'],result
        del a,b,zero,parent;gc.collect()
    out=dict(passed=True,actual_updates=8,results=results);write(D/'verification/RESULT.json',out);return out

def baseline(name):
    if name=='P_reset':return F/'P0/development'
    candidates=[F/'native_base/evaluation/RESULT.json']
    for path in candidates:
        obj=read(path)
        if obj.get('checkpoint')==str(N0) and obj.get('complete') and not obj.get('sentinel'):return path.parent
    raise RuntimeError('Existing full native baseline not found; locate evidence, do not regenerate silently')

def calibrate():
    """One CPU-only sensitivity pass, never changes quality thresholds."""
    target=D/'calibration/RESULT.json'
    if target.exists():return read(target)
    from ardy.skeleton import G1Skeleton34
    from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
    sk=G1Skeleton34();rep=ArdyMotionRep(sk,fps=25)
    sources={'P0':F/'P0/development','B8':F/'B/eval8','C8':R/'closed_loop/eval8'}
    out=[]
    def measured(ph):
        motion=rep.inverse(torch.from_numpy(ph.copy())[None],is_normalized=False)
        feet=motion['posed_joints'][0,:,list(sk.foot_joint_idx)]
        speed=torch.diff(feet,dim=0)[...,[0,2]].norm(dim=-1)*25
        return feet,speed,motion['foot_contacts'][0,1:]
    for noise in [0,1]:
        inputs={name:read(folder/f'S_3_n{noise}.json') for name,folder in sources.items()}
        gtrow=inputs['P0']['source'];gt=np.load(gtrow['feature_path'],mmap_mode='r')[gtrow['crop_start']:gtrow['crop_start']+104].copy()
        gtfeet,_,_=measured(gt);floor=float(torch.quantile(gtfeet[...,1],.1))
        arrays={name:measured(np.load(row['artifact'])['physical'][312:364]) for name,row in inputs.items()}
        for name,(feet,speed,contact) in arrays.items():
            original=inputs[name]['windows'][6]['metrics']['whole']
            slip=float((speed[contact]>.15).float().mean()*100)
            assert abs(slip-original['contact_slip_percent'])<1e-4
            near=feet[1:,:,1]<floor+.04
            out.append(dict(noise=noise,trajectory=name,source=inputs[name]['artifact'],
                original_contact_slip=slip,original_contact_speed=float(speed[contact].mean()),
                shared_gt_floor=floor,geometric_contact_fraction=float(near.float().mean()),
                geometric_low_speed=float(speed[near].mean()) if near.any() else None,
                cross_masks={other:dict(slip_percent=float((speed[mask]>.15).float().mean()*100),speed=float(speed[mask].mean())) for other,(_,_,mask) in arrays.items()}))
    pool=read(R/'w1_completion/POOL.json')
    teacher=[dict(id=r['id'],label=r['label'],label_sha256=r['label_sha256'],stage=r['stage_key'],checks=r['checks']) for r in pool['labels'][:4]]
    result=dict(rows=out,qualified_teacher_reference=teacher,GT_floor_source='common initial physical104 before any model generation',
        threshold_changed=False,conclusion='Trajectory and predicted contacts both change; neither proxy nor cross-mask is physical contact truth. Keep conservative historical thresholds, explicit uncertainty.',
        previously_severe=[str(F/'B/eval32/COMPARISON.json'),str(F/'B/eval64/COMPARISON.json')],new_generation=0,optimizer_updates=0)
    write(target,result);return result

def warm(name):
    from .evaluation import compare
    cfgpath=D/'configs'/f'{name}.json';dest=D/name;train=dest/'train';cfg=read(cfgpath)
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    base=baseline(name);done=len(rows(train/'train.jsonl'));device='0' if name=='P_reset' else '4'
    comparisons=[]
    for step in [8,32,128,250,500]:
        current=dest/f'eval{step}'
        if not (train/f'checkpoint{step}.pt').exists():
            assert not (train/'STOP.json').exists()
            args=['--config',cfgpath,'--output',train,'--until',step]
            if done:args+=['--resume',train/f'checkpoint{done}.pt']
            run('train',args,dest/f'train{step}.log',device);done=step
        if not (current/'RESULT.json').exists():run('evaluate',['--config',cfgpath,'--checkpoint',train/f'checkpoint{step}.pt','--output',current],dest/f'eval{step}.log',device)
        own=compare(base,current);write(current/'OWN_COMPARISON.json',own)
        protected=compare(F/'P0/development',current);write(current/'P0_COMPARISON.json',protected)
        comparisons.append(dict(step=step,own_stop=own['stop'],P0_stop=protected['stop'],evaluation=str(current)))
        if own['stop']:
            write(train/'STOP.json',dict(step=step,reason='own_parent_confirmed_harm',evidence=str(current/'OWN_COMPARISON.json')))
            break
    result=dict(status='STOPPED_FOR_HARM' if comparisons[-1]['own_stop'] else 'BOUNDED_TRAINING_COMPLETE',
        checkpoint=str(train/f'checkpoint{comparisons[-1]["step"]}.pt'),comparisons=comparisons,
        baseline=str(base),production_eligible=False)
    write(dest/'RESULT.json',result);return result

def scores(folder):
    result=read(Path(folder)/'RESULT.json');base=result['foundation']['rows']
    return dict(function={r['id'] for r in base if r['function_pass']},joint={r['id'] for r in base if r['joint_pass']},
        events=sum(e['all_pass'] for r in result['rows'] for e in r['events']),
        chains=sum(r['all_pass'] for r in result['rows']))

def eligible_warm(name):
    obj=read(D/name/'RESULT.json');base=scores(obj['baseline']);previous=None;eligible=[]
    for row in obj['comparisons']:
        s=scores(row['evaluation']);new=s['joint']-base['joint']
        retained=base['joint']<=s['joint'] and base['function']<=s['function'] and not row['own_stop']
        if retained and previous and previous['retained'] and new&previous['new']:
            eligible.append(dict(arm=name,step=row['step'],checkpoint=str(D/name/'train'/f'checkpoint{row["step"]}.pt'),
                persistent_new_joint=sorted(new&previous['new']),joint_count=len(s['joint']),function_count=len(s['function']),
                P0_stop=row['P0_stop'],evaluation=row['evaluation']))
        previous=dict(new=new,retained=retained)
    # A later confirmed harmful update does not erase a saved earlier eligible state.
    return eligible

def fresh():
    from .evaluation import compare
    cfgpath=D/'configs/Fresh.json';dest=D/'Fresh';train=dest/'train'
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    if not (dest/'TEXT_READY.json').exists():run('encode',['--config',cfgpath,'--output',dest],dest/'encode.log','1')
    cfg=read(cfgpath);done=len(rows(train/'train.jsonl'));evaluations=[];prior=None;prior2=None
    for step in [0,100,500,1000,2000,3500,5000]:
        cp=train/f'checkpoint{step}.pt';ev=dest/f'eval{step}'
        if not cp.exists():
            args=['--config',cfgpath,'--output',train,'--until',step]
            if done:args+=['--resume',train/f'checkpoint{done}.pt']
            run('train',args,dest/f'train{step}.log','0,1,2,3',4);done=step
        # Uniform full frozen panel enables actual retention checks, not a
        # different set of cases at each savepoint. Kernel no-grad only.
        if not (ev/'RESULT.json').exists():
            args=['--config',cfgpath,'--checkpoint',cp,'--output',ev]
            if step<2000:args+=['--foundation-only']
            run('evaluate',args,dest/f'eval{step}.log','4')
        score=scores(ev);stable=set() if prior is None or prior2 is None else prior['score']['joint']&prior2['score']['joint']
        harm=False
        if stable:
            # Early foundation-only and later full panels differ in event rows.
            # Confirm acquired foundation abilities explicitly below.
            comparison=dict(invalid=[],basic_confirmed=[])
            # Only previously stable foundation abilities have a retention
            # baseline for a fresh learner. Initially poor actions aren't harm.
            lost=stable-score['joint']
            harm=bool(comparison['invalid'] or any(x['id'] in stable for x in comparison['basic_confirmed']))
            oldrows={r['id']:r for r in read(Path(prior['evaluation'])/'RESULT.json')['foundation']['rows']}
            newrows={r['id']:r for r in read(ev/'RESULT.json')['foundation']['rows']}
            severe=[]
            for key in stable:
                a,b=oldrows[key],newrows[key];am,bm=a['metrics'],b['metrics'];why=[]
                if a['function_pass'] and not b['function_pass']:why.append('semantic_loss')
                ps,cs=am['whole']['contact_slip_percent'],bm['whole']['contact_slip_percent']
                av,bv=am['whole']['contact_speed_m_s'],bm['whole']['contact_speed_m_s']
                if None not in (ps,cs,av,bv) and cs-ps>20 and bv>1.5*av:why.append('contact_proxy_harm')
                if bm['joint_limit_max_deg']>30 and bm['joint_limit_max_deg']-am['joint_limit_max_deg']>5:why.append('joint_harm')
                if why:severe.append(dict(id=key,action=b['action'],source=b['source_index'],cell=b['cell'],noise=b['noise_index'],why=why))
            confirmed=[r for r in severe if any((r['action'],r['source'],r['cell'])==(s['action'],s['source'],s['cell']) and r['noise']!=s['noise'] and set(r['why'])&set(s['why']) for s in severe)]
            harm=harm or bool(confirmed)
            write(ev/'STABLE_CAPABILITY_COMPARISON.json',dict(stable_ids=sorted(stable),severe=severe,confirmed=confirmed,stop=harm))
        else:lost=set()
        entry=dict(step=step,evaluation=str(ev),function=len(score['function']),joint=len(score['joint']),events=score['events'],chains=score['chains'],
            confirmed_harm_of_stable_capability=harm,lost_previously_stable_joint=sorted(lost))
        evaluations.append(entry);write(dest/'PROGRESS.json',dict(evaluations=evaluations))
        if harm:
            write(train/'STOP.json',dict(step=step,reason='fresh_previously_stable_capability_harm',evidence=str(ev/'COMPARISON.json')));break
        prior2=prior;prior=dict(score=score,evaluation=str(ev))
    last=evaluations[-1];earlier=evaluations[-2] if len(evaluations)>1 else last
    lastscore=scores(last['evaluation']);earlierscore=scores(earlier['evaluation']);zeroscore=scores(evaluations[0]['evaluation'])
    persistent=(lastscore['joint']&earlierscore['joint'])-zeroscore['joint']
    learned=bool(persistent) and last['joint']>evaluations[0]['joint'] and earlier['joint']>evaluations[0]['joint']
    # Random motion can pass amplitude-only semantics. Never require increasing
    # that misleading random baseline, or count it as learned action quality.
    cold=sum('_cold_text' in key for key in persistent);history=sum('_gt_accepted_text' in key for key in persistent)
    usable=learned and last['function']>=48 and last['joint']>=24 and cold>=2 and history>=2 and not last['confirmed_harm_of_stable_capability']
    result=dict(status='STOPPED_FOR_HARM' if last['confirmed_harm_of_stable_capability'] else 'BOUNDED_TRAINING_COMPLETE',
        evaluations=evaluations,checkpoint=str(train/f'checkpoint{last["step"]}.pt'),evaluation=last['evaluation'],
        learning_established=learned,persistent_new_joint_ids=sorted(persistent),cold_joint=cold,history_joint=history,closed_loop_admissible=usable,threshold_scope='project screening frozen before first updated candidate evaluation; not final qualification',
        independent_repeat=False,model_qualified=False,expansion_ready=False)
    write(dest/'RESULT.json',result);return result

def closed(selected):
    dest=D/'closed_loop';path=D/'configs/closed_loop.json'
    if (dest/'PIPELINE_STATE.json').exists() and read(dest/'PIPELINE_STATE.json')['status'] in ['STOPPED_FOR_HARM','BOUNDED_EXECUTION_COMPLETE','INCOMPLETE_REFRESH_COVERAGE']:return read(dest/'PIPELINE_STATE.json')
    source='Fresh' if selected.get('arm')=='Fresh' else selected['arm']
    cfg=read(D/'configs'/f'{source}.json');cfg.pop('frozen_manifest',None)
    cfg.update(parent=selected['checkpoint'],name='closed_loop_v1',initialization_policy='carry_state_new_stage',optimizer_state_policy='inherit',
        max_updates=492,gt_slots=8,budget_root=str(R),savepoints=[0,8,32,64,128,250,492],refresh_steps=[0,250],
        teacher_history_frames=52,teacher_candidates_per_generation=200,baseline_evaluation=selected['evaluation'])
    # Own-parent harm stops research. Cumulative P0 deficits separately block
    # replacement; existing native deficits must not masquerade as new damage.
    write(path,cfg)
    pool=dest/'pool0/POOL.json'
    if not pool.exists():run('collect',['--config',path,'--checkpoint',selected['checkpoint'],'--output',pool.parent,'--generation',1,'--teacher-history',52,'--max-candidates',200],dest/'collect0.log','4')
    run('pipeline',['--config',path,'--output',dest,'--pool',pool,'--training-ranks',4],dest/'pipeline.log','0,1,2,3')
    from .evaluation import compare
    for ev in dest.glob('eval*/RESULT.json'):
        comp=compare(F/'P0/development',ev.parent);write(ev.parent/'P0_COMPARISON.json',comp)
    return read(dest/'PIPELINE_STATE.json')

def execute():
    """One continuous registered route. Restart reuses completed artifacts."""
    import concurrent.futures
    assert read(D/'verification/RESULT.json')['passed']
    assert (D/'shared_text/TEXT_READY.json').exists() and (D/'calibration/RESULT.json').exists()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        jobs=[executor.submit(run,'startpoint',['--mode','warm','--arm',name],D/f'{name}.log','0' if name=='P_reset' else '4') for name in ['P_reset','N_reset']]
        for job in jobs:job.result()
    qualified=eligible_warm('P_reset')+eligible_warm('N_reset')
    write(D/'WARM_SELECTION.json',dict(eligible=qualified,rule='stable new joint IDs with zero loss of own old joint/function IDs; rank joint then function; P_reset tie priority'))
    if qualified:
        selected=max(qualified,key=lambda r:(r['joint_count'],r['function_count'],r['arm']=='P_reset',-r['step']))
    else:
        result=fresh()
        selected=dict(arm='Fresh',checkpoint=result['checkpoint'],evaluation=result['evaluation']) if result['closed_loop_admissible'] else None
    write(D/'SELECTED.json',dict(candidate=selected,production_replaced=False))
    if selected:closed(selected)
    write(D/'ROUTE_STATE.json',dict(status='PRIMARY_BRANCHES_FINISHED_REQUIRES_DELIVERY',selected=selected,model_qualified=False,expansion_ready=False))

def summarize():
    result={};oldrows=rows(M/'R2_P0_B/train.jsonl')
    for name in ['P_reset','N_reset','Fresh']:
        dest=D/name;rr=rows(dest/'train/train.jsonl')
        if not rr:continue
        exposures=[x for r in rr for x in r['receipts']]
        samples=dict(updates=len(rr),samples=len(exposures),families=len({r['source_family_id'] for r in exposures}),
            sources=dict(collections.Counter(r['dataset'] for r in exposures)),HGF=dict(collections.Counter(str((r['H'],r['G'],r['F'])) for r in exposures)),
            conditions=dict(collections.Counter(r['condition'] for r in exposures if 'condition' in r)),
            F_visible=sum(r.get('F_visible_after_drop',False) for r in exposures),
            median_seconds=float(np.median([r['seconds'] for r in rr[1:]])) if len(rr)>1 else rr[0]['seconds'],
            optimizer_updates_nonzero=all(any(x>0 for x in r['master_delta'].values()) for r in rr))
        if name in ['P_reset','N_reset']:
            same=[]
            for old,new in zip(oldrows,rr):
                same.extend(all(a[k]==b[k] for k in ['id','input_hash','target_hash','noise_hash','physical_G_hash']) for a,b in zip(old['receipts'],new['receipts']))
            samples['historical_frozen_inputs_exact']=all(same);samples['matched_sample_count']=len(same)
        if (dest/'RESULT.json').exists():samples['result']=read(dest/'RESULT.json')
        result[name]=samples
    write(D/'EXPOSURE_AND_PAIRING.json',result)
    ledger=read(D/'EXPERIMENT_LEDGER.json')
    for name,obj in result.items():
        ledger['experiments'].setdefault(name,{})
        ledger['experiments'][name].update(status=obj.get('result',{}).get('status','RUNNING'),
            actual_updates=obj['updates'],actual_samples=obj['samples'],families=obj['families'],
            result_path=str(D/name/'RESULT.json'),limitations='DEV only; no deployment qualification')
    ledger['engineering_checks']=dict(status='DONE',updates=8,evidence=str(D/'verification/RESULT.json'))
    write(D/'EXPERIMENT_LEDGER.json',ledger)
    write(D/'CODE_HASHES.json',{str(p.relative_to(P)):sha(p) for p in (P/'src/motion_training').rglob('*.py')})
    import difflib
    patch=[]
    for old in (D/'code_before').glob('*.py'):
        current=P/'src/motion_training'/old.name
        if old.name=='load_model.py':current=P/'src/motion_training/_runtime/ardy/model/load_model.py'
        if current.exists():patch.extend(difflib.unified_diff(old.read_text().splitlines(True),current.read_text().splitlines(True),fromfile='before/'+str(current.relative_to(P)),tofile='after/'+str(current.relative_to(P))))
    (D/'CORE_CHANGES.diff').write_text(''.join(patch))
    snapshots=D/'executed_code';snapshots.mkdir(exist_ok=True)
    import shutil
    for name in ['startpoint.py','engine.py','runtime.py','data.py','evaluation.py','visualization.py','pipeline.py','__main__.py']:
        shutil.copy2(P/'src/motion_training'/name,snapshots/name)
    accounting={}
    for name,root in [('R2_including_previous',R),('fresh_foundation',D/'fresh_budget')]:
        updates=rows(root/'UPDATES.jsonl');times=rows(root/'GPU_TIME.jsonl')
        accounting[name]=dict(counts=dict(collections.Counter(r['category'] for r in updates)),
            optimizer_updates=sum(r['category']!='teacher' for r in updates),
            completed_gpu_hours=sum(r['seconds']*r.get('devices',1) for r in times)/3600,
            active=read(root/'ACTIVE.json') if (root/'ACTIVE.json').exists() else {},limits=read(root/'LIMITS.json'))
    write(D/'RESOURCE_ACCOUNTING.json',accounting)
    return result

def delivery():
    """Reuse saved motions for complete comparisons and exact duration crops."""
    summarize();dest=D/'videos';dest.mkdir(exist_ok=True)
    candidates=[('P0 reference',F/'P0/development'),('Native0920 reference',F/'native_base/evaluation')]
    for name in ['P_reset','N_reset','Fresh']:
        rp=D/name/'RESULT.json'
        if not rp.exists():continue
        r=read(rp);ev=r.get('evaluation') or r['comparisons'][-1]['evaluation']
        candidates.append((name,Path(ev)))
    cases=[]
    for key in ['W_3_n0','W_3_n1','S_3_n0','S_3_n1','T_3_n0','X_3_n0']:
        cols=[]
        for name,folder in candidates:
            row=read(folder/(key+'.json'))
            cols.append(dict(name=name,artifact=row['artifact'],row_source=str(folder/(key+'.json'))))
        cases.append(dict(id=key,cols=cols))
    write(dest/'CASES.json',cases)
    cfgpath=D/'configs/P_reset.json'
    if not (dest/'index.html').exists():
        with Budget(R).timed('raw_complete_videos',closing=True):
            run('render',['--config',cfgpath,'--output',dest],D/'render.log','5')
    from ardy.skeleton import G1Skeleton34
    from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
    sk=G1Skeleton34();rep=ArdyMotionRep(sk,fps=25)
    limits=np.asarray(read(read(F/'native_base_compat/LOAD.json')['config']['joint_limit_spec'])['bounds'])
    durations=[]
    for name,folder in candidates:
      for key in ['W_3_n0','W_3_n1','W_4_n0','W_4_n1']:
        row=read(folder/(key+'.json'));arr=np.load(row['artifact'])
        for frames in [52,225,500,1000]:
            q=arr['qpos'][:frames];ph=arr['physical'][:frames]
            assert len(q)==frames
            decoded=rep.inverse(torch.from_numpy(ph.copy())[None],is_normalized=False)
            feet=decoded['posed_joints'][0,:,list(sk.foot_joint_idx)]
            speed=torch.diff(feet,dim=0)[...,[0,2]].norm(dim=-1)*25;contact=decoded['foot_contacts'][0,1:]
            violation=np.maximum(np.maximum(limits[:,0]-q[:,7:],q[:,7:]-limits[:,1]),0)
            durations.append(dict(candidate=name,case=key,frames=frames,seconds=frames/25,
                initial_GT_prefix_frames=104,student_preroll_frames=208,free_walk_after_preroll=True,
                artifact=row['artifact'],artifact_slice=[0,frames],
                contact_fraction=float(contact.float().mean()),contact_slip_percent=float((speed[contact]>.15).float().mean()*100) if contact.any() else None,
                contact_speed_m_s=float(speed[contact].mean()) if contact.any() else None,
                joint_limit_max_deg=float(np.rad2deg(violation).max()),
                root_path_m=float(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=-1).sum()),
                root_endpoint_displacement_m=float(np.linalg.norm(q[-1,:2]-q[0,:2])),
                foot_contact_is_prediction=True,unique_long_GT=False))
    write(D/'DURATION_RESULTS.json',dict(rows=durations,protocol='Exact crops from saved SAME long rollout; not independent cold52 or independent 9/20/40s draws. Dedicated cold52 is in foundation panel. No GT ADE claim.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=D/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,2,figsize=(13,9),constrained_layout=True)
    for ni in [0,1]:
      for name,folder in candidates:
        row=read(folder/f'W_3_n{ni}.json');ws=row['windows'];x=np.arange(len(ws))*52/25
        for ax,key in zip(axes[:,ni],['contact_slip_percent','root_speed_m_s','joint_limit_max_deg']):
            values=[w['metrics'][key] if key=='joint_limit_max_deg' else w['metrics']['whole'][key] for w in ws]
            ax.plot(x,values,label=name,linewidth=1.3)
        axes[0,ni].set_title(f'Same walk / backwalk source; noise {ni}')
        axes[-1,ni].set_xlabel('Generated seconds after common preroll')
        for ax in axes[:,ni]:ax.grid(alpha=.2);ax.axvline(20*52/25,color='grey',linestyle='--')
    for ax,label in zip(axes[:,0],['Predicted-contact slip (%)','Root speed (m/s)','Maximum joint excess (deg)']):ax.set_ylabel(label)
    axes[0,0].legend(fontsize=8);fig.suptitle('Raw autonomous motion: quality and movement shown together')
    fig.savefig(figures/'walk_quality.png',dpi=160);fig.savefig(figures/'walk_quality.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for name in ['P_reset','N_reset','Fresh']:
        rr=rows(D/name/'train/train.jsonl')
        if not rr:continue
        block=25 if name=='Fresh' else 1
        groups=[rr[i:i+block] for i in range(0,len(rr),block)]
        axes[0].plot([g[-1]['step'] for g in groups],[np.mean([r['loss'] for r in g]) for g in groups],label=name)
        saves=sorted((p for p in (D/name).glob('eval*/RESULT.json')),key=lambda p:int(p.parent.name[4:]))
        axes[1].plot([int(p.parent.name[4:]) for p in saves],[read(p)['foundation']['joint'] for p in saves],marker='o',label=name)
    for ax in axes:ax.set_xlabel('New stage optimizer updates');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Training loss (Fresh 25-step means)');axes[1].set_ylabel('Foundation function AND geometry /96')
    fig.savefig(figures/'learning.png',dpi=160);fig.savefig(figures/'learning.pdf');plt.close(fig)
    write(D/'RESULTS.json',dict(candidates=[dict(name=n,path=str(p),**{k:(sorted(v) if isinstance(v,set) else v) for k,v in scores(p).items()}) for n,p in candidates],
        durations=str(D/'DURATION_RESULTS.json'),engineering=read(D/'verification/RESULT.json'),
        production_replaced=False,independent_repeat=False,model_qualified=False,expansion_ready=False))
    report=['# 起点解耦与有界学习：实际执行结果','',
        '工程修改与8次真实更新验证完成；模型是否改善以以下同一保存点的完整结果为准。生产权重未更换。','',
        '|候选|基础功能/96|基础功能与几何/96|事件联合/48|完整链/16|',
        '|---|---:|---:|---:|---:|']
    for name,folder in candidates:
        s=scores(folder);report.append(f'|{name}|{len(s["function"])}|{len(s["joint"])}|{s["events"]}|{s["chains"]}|')
    report+=['','## 执行与结论边界','',
        '- P_reset和N_reset使用相同真实GT输入、编码目标、条件与噪声；与旧B前缀的逐项匹配见EXPOSURE_AND_PAIRING.json。',
        '- 清Adam保留同一FP32 master；从零初始化覆盖root/body/条件辅助，公共codec/统计冻结。恢复检查见verification/RESULT.json。',
        '- 温启动按相对各自0点的保护门槛停止；相对P0的差距另外记录，Native原有缺陷不算新增伤害。',
        '- Fresh不同学习率与初始化，是独立学习方案，不能把差异只归因旧权重。5000步不是正式50k或收敛保证。',
        '- 所有接触滑动指标均为运动学代理，非SONIC物理跟随结果；视频为完整原始qpos，无平滑。',
        '- 本轮实际修正JSON数值解析及旧Native缺失Python/NumPy RNG的新阶段兼容；失败前0更新，原耗时与日志保留。','']
    for name in ['P_reset','N_reset','Fresh']:
        path=D/name/'RESULT.json'
        if path.exists():
            rr=read(path);report+= [f'### {name}','',f'状态：{rr["status"]}。权重：`{rr["checkpoint"]}`。',
                f'逐项结果：[{name}/RESULT.json]({name}/RESULT.json)。','']
    selected=read(D/'SELECTED.json') if (D/'SELECTED.json').exists() else dict(candidate=None)
    closed_state=read(D/'closed_loop/PIPELINE_STATE.json') if (D/'closed_loop/PIPELINE_STATE.json').exists() else dict(status='NOT_RUN_NO_ELIGIBLE_FOUNDATION')
    report+=['## 后续阶段','',f'闭环状态：`{closed_state["status"]}`。未执行的采集、刷新、独立复训不得当作验证失败。',
        '当前不自动扩到50k，不以基础loss下降或工程验收替代同一checkpoint的联合能力。独立复训及新来源资格未满足时，EXPANSION_READY=false。','',
        '[完整对照视频](videos/index.html)；[实际曝光与配对](EXPOSURE_AND_PAIRING.json)；[精确时长切片指标](DURATION_RESULTS.json)。']
    (D/'REPORT.md').write_text('\n'.join(report)+'\n')
    write(D/'DELIVERY_STATE.json',dict(engineering_complete=True,closed_loop_state=closed_state,
        independent_repeat=False,model_qualified=False,expansion_ready=False,report=str(D/'REPORT.md'),selected=selected))
    return read(D/'RESULTS.json')

def render_warm():
    dest=D/'warm_videos';candidates=[('P0',F/'P0/development'),('P_reset8',D/'P_reset/eval8'),
        ('Native0920',F/'native_base/evaluation'),('N_reset32',D/'N_reset/eval32')]
    cases=[]
    for key in ['W_3_n0','W_3_n1','X_3_n0','X_3_n1']:
        cases.append(dict(id=key,cols=[dict(name=n,artifact=read(f/(key+'.json'))['artifact'],row_source=str(f/(key+'.json'))) for n,f in candidates]))
    write(dest/'CASES.json',cases)
    with Budget(R).timed('warm_failure_videos',closing=True):
        run('render',['--config',D/'configs/P_reset.json','--output',dest],D/'render_warm.log','5')

def complete_route():
    """Apply frozen final eligibility to stored evidence; never regenerate it."""
    path=D/'Fresh/RESULT.json'
    if path.exists():
        r=read(path);ev=r['evaluations'];a=scores(ev[-1]['evaluation']);b=scores(ev[-2]['evaluation']);z=scores(ev[0]['evaluation'])
        persistent=(a['joint']&b['joint'])-z['joint']
        learned=bool(persistent) and len(a['joint'])>len(z['joint']) and len(b['joint'])>len(z['joint'])
        cold=sum('_cold_text' in key for key in persistent);history=sum('_gt_accepted_text' in key for key in persistent)
        admitted=learned and len(a['function'])>=48 and len(a['joint'])>=24 and cold>=2 and history>=2 and r['status']!='STOPPED_FOR_HARM'
        write(D/'Fresh/INITIAL_SELECTION_RECORD.json',r) if not (D/'Fresh/INITIAL_SELECTION_RECORD.json').exists() else None
        r.update(learning_established=learned,persistent_new_joint_ids=sorted(persistent),cold_joint=cold,history_joint=history,closed_loop_admissible=admitted,
            selection_policy=str(D/'DECISIONS.json'),selection_uses_saved_motions_only=True)
        write(path,r)
        if admitted:
            selected=dict(arm='Fresh',checkpoint=r['checkpoint'],evaluation=r['evaluation'])
            write(D/'SELECTED.json',dict(candidate=selected,production_replaced=False));closed(selected)
    return delivery()

def main(mode,arm=None):
    if mode=='prepare':return prepare()
    if mode=='verify':return verify()
    if mode=='warm':return warm(arm)
    if mode=='calibrate':return calibrate()
    if mode=='execute':return execute()
    if mode=='fresh':return fresh()
    if mode=='summarize':return summarize()
    if mode=='delivery':return delivery()
    if mode=='render-warm':return render_warm()
    if mode=='complete-route':return complete_route()
    raise ValueError(mode)
