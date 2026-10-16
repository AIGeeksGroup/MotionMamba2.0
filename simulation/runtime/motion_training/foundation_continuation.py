"""Fixed-recipe foundation progression after the completed startpoint study.

Owns registration, gates and subprocesses only; engine.py owns every update.
Historical budgets, STOPs and DEV results are never overwritten.
"""
import collections
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from .io import read, write, rows, sha, Budget

Z = Path('/data3/zts')
P = Path(__file__).resolve().parents[2]
D = Z / 'result/g1_startpoint_and_learning_execution_20261004'
OUT = D / 'continuation'
R2 = Z / 'result/g1_master_experiment_roadmap_20261003/execution_v1_1/R2'
LAUNCH = P / 'examples/train_motion.sh'


def run(command, args, log, devices='0', ranks=1):
    command = ['bash',str(LAUNCH),command,*map(str,args)]
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    write(log.with_suffix('.command.json'),dict(command=command,devices=devices,ranks=ranks,
        source_hashes={n:sha(P/'src/motion_training'/n) for n in
                      ['engine.py','runtime.py','data.py','sampling.py','losses.py','evaluation.py','foundation_continuation.py']}))
    with log.open('a') as stream:
        subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,MOTION_NPROC=str(ranks)),
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'REGISTRATION.json').exists():
        registered=read(OUT/'REGISTRATION.json')
        actual=sha(Path(__file__))
        if registered['experiment_source_hashes']['foundation_continuation.py']!=actual:
            assert not rows(OUT/'train/train.jsonl'),'Do not silently change gates during training'
            registered['prelaunch_revision']='Before first optimizer update: add full event/tail protection and five targeted regression checks; reuse prepared text cache.'
            registered['experiment_source_hashes']['foundation_continuation.py']=actual
            write(OUT/'REGISTRATION.json',registered)
            policy=read(OUT/'EVAL_POLICY.json')
            policy['event_protection']='Full W/S/T/X windows and same-stage two-noise confirmation; acquired contact and stop/recovery checks protected independently of total pass count.'
            write(OUT/'EVAL_POLICY.json',policy)
            shutil.copy2(Path(__file__),OUT/'executed_code/foundation_continuation.py')
        return registered
    transfer=read(OUT/'transfer/RESULT.json')
    assert transfer['status']=='DONE' and transfer['optimizer_updates']==0
    # No diagnosed implementation defect in the production CFG/training path.
    # The noCFG bool-mask repair affects only the diagnostic's missing arm.
    cfg=read(D/'configs/Fresh.json')
    cfg.update(version='foundation_continuation_v1',parent=str(D/'Fresh/train/checkpoint5000.pt'),
        initialization_policy='carry_state_new_stage',optimizer_state_policy='inherit',
        max_updates=20000,savepoints=[0,1000,5000,10000,15000,20000],seed=2026110401,
        budget_root=str(OUT/'budget'),text_roots=[str(OUT/'text'),*cfg['text_roots']],
        execution_authorization='USER_CONTINUE_20261004',
        stage_note='new TRAIN plan; unchanged objective, distribution, optimizer and inherited 50k scheduler; not exact resume of old stage')
    cfg.pop('initialization_seed',None)
    write(OUT/'config.json',cfg)
    write(OUT/'budget/LIMITS.json',dict(caps=dict(B=20000,C=0,teacher=0,checks=0),total_updates=20000,gpu_hours=72))
    checkpoint=Path(cfg['parent'])
    available=shutil.disk_usage(OUT).free
    assert available>8*checkpoint.stat().st_size+15*1024**3,'Insufficient checkpoint/artifact reserve'
    policy=dict(
        version='foundation_continuation_gates_v1',frozen_before_new_optimizer_updates=True,
        original_thresholds_unchanged=True,previous_STOPs_unchanged=True,
        numerical_error='stop affected branch immediately; retain failed receipt',
        stable_capability='same case/check passes at two preceding saved evaluations; protect per metric, not only whole joint passes',
        severe_geometry=dict(joint_peak_deg=30,joint_delta_deg=5,slip_delta_pp=20,contact_speed_ratio=1.5,
                             seam_root_m=.5,seam_root_delta_m=.1,seam_joint_deg=45,seam_joint_delta_deg=10),
        confirmation='same action/source/cell or event phase; both registered noises',
        functional_loss='all losses reported and block qualification; stop on previously stable function lost plus >=50% action-relevant activity collapse or locomotion direction reversal, in both noises',
        reasonable_fluctuation='near-threshold semantic changes alone do not stop foundational research',
        learning='report continuous same-case changes separately from quality pass counts; no claim convergence from random training loss',
        no_progress='two consecutive 5000-update intervals with no supported joint-count or paired geometry+function improvement: save and stop continuation, no automatic grid',
        closed_entry='two adjacent saves with >=2 persistent cold_text and >=2 persistent gt_accepted_text joint-pass cases, no confirmed protected harm; no 40s-perfect requirement',
        production='no replacement; qualification also requires broad retention, complete interaction, independent repeat and sealed evaluation')
    write(OUT/'EVAL_POLICY.json',policy)
    prior=read(D/'EXPOSURE_AND_PAIRING.json')['Fresh']
    sec=prior['median_seconds']
    reg=dict(status='REGISTERED',question='Does the fixed foundation recipe establish stable raw motion with systematic additional learning?',
        prior_completed=dict(updates=5000,parent=str(checkpoint),sha256=sha(checkpoint),result=str(D/'Fresh/RESULT.json')),
        transfer_result=str(OUT/'transfer/RESULT.json'),transfer_conclusion='24/24 total paired losses worsened; hybrid improves on average but decoded body/FK worsen; noCFG does not rescue quality',
        authorized_scope='Continue the agreed finite diagnostic then systematic foundation and conditional current-student loop; no production replacement',
        optimizer='inherit full FP32 master+Adam+scheduler at5000; no new warmup; stage0 predictions unchanged',
        changes=['new disjoint-from-DEV frozen TRAIN exposure plan','additional updates','prospectively separated learning/research/deployment gates'],
        unchanged=['losses','weights','source/task ratios','architecture','batch10','CFG2 evaluation','frozen frontend'],
        initial_full_foundation_target=50000,executed_before_this_stage=5000,
        maximum_new_updates=20000,maximum_global_step=25000,resource_cap_gpu_hours=72,
        budget_reason='Four-rank throughput would exceed the old72GPUh planning cap for45000 more updates; do not silently raise it. This executable tranche is <=20000, with >=25% time reserve. A50000 completion is NOT claimed.',
        training_devices='0,1,2,3',training_ranks=4,
        estimated_seconds_per_update=sec,estimated_first5000_training_hours=5000*sec/3600,
        estimated_max_training_gpu_hours=20000*sec*4/3600,
        stage0_evaluation_reuse=str(D/'Fresh/eval5000'),
        refresh='If foundation entrance is usable, switch at a safe saved boundary to <=492 current-history updates, generations0/250; existing R2 budget, <=200 teacher attempts each generation',
        final_independent='Only a jointly improved candidate triggers a separately budgeted independent training repeat; no sealed-set selection',
        experiment_source_hashes={n:sha(P/'src/motion_training'/n) for n in ['engine.py','runtime.py','data.py','losses.py','foundation_continuation.py']})
    assert reg['estimated_max_training_gpu_hours']<72*.75
    write(OUT/'REGISTRATION.json',reg)
    for name in ['engine.py','runtime.py','data.py','losses.py','sampling.py','evaluation.py','foundation_continuation.py','learning_transfer.py','__main__.py']:
        target=OUT/'executed_code'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(P/'src/motion_training'/name,target)
    write(OUT/'EXPERIMENT_LEDGER.json',dict(reused_DONE=['startpoint warm8/32','Fresh5000','paired transfer48forwards16windows','precision/codec/resume/fourGPU gates'],
        experiments=dict(foundation_continuation=dict(status='REGISTERED',parent=str(checkpoint),config=str(OUT/'config.json'),
            criterion=str(OUT/'EVAL_POLICY.json'),result=str(OUT/'STATE.json'),limitation='DEV panel, not deployment qualification'))))
    return reg


def foundation_rows(folder):
    return {r['id']:r for r in read(Path(folder)/'RESULT.json')['foundation']['rows']}


def event_protection(previous2,previous,current):
    older,old,new=[read(Path(p)/'RESULT.json') for p in [previous2,previous,current]]
    if not new.get('rows'):
        return dict(scope='foundation-only savepoint; full event check at5k',severe=[],confirmed=[],lost=[],invalid=[])
    aa={r['id']:r for r in older['rows']};bb={r['id']:r for r in old['rows']}
    severe=[];lost=[];invalid=[]
    for row in new['rows']:
        before=bb[row['id']];earlier=aa[row['id']]
        if not row['valid'] or (before.get('rollout_complete') and not row.get('rollout_complete')):
            invalid.append(row['id'])
        for wi,(p,n) in enumerate(zip(before['windows'],row['windows'])):
            pm,nm=p['metrics'],n['metrics'];why=[]
            if nm['joint_limit_max_deg']>30 and nm['joint_limit_max_deg']>pm['joint_limit_max_deg']+5:why.append('severe_joint')
            for key,absolute,delta in [('seam_root_m',.5,.1),('seam_joint_rms_deg',45,10)]:
                if nm[key]>absolute and nm[key]>pm[key]+delta:why.append(key)
            em=earlier['windows'][wi]['metrics'] if wi<len(earlier['windows']) else None
            ps,ns=pm['whole']['contact_slip_percent'],nm['whole']['contact_slip_percent']
            es=em['whole']['contact_slip_percent'] if em else None
            pv,nv=pm['whole']['contact_speed_m_s'],nm['whole']['contact_speed_m_s']
            if None not in [ps,ns,es,pv,nv] and max(ps,es)<=10 and ns-ps>20 and nv>1.5*pv:why.append('acquired_contact_collapse')
            receipt=row['receipts'][wi]
            if why:severe.append(dict(id=row['id'],source=row['source']['source_family_id'],group=row['source']['group'],
                action=receipt['action'],kind=receipt['kind'],phase=receipt['phase'],noise=row['noise_index'],window=wi,why=why))
        for ei,(a,b,c) in enumerate(zip(earlier['events'],before['events'],row['events'])):
            for key in ['progress_and_legs_by_deadline','sustained_after_deadline','holds_after_buffer','stopped_by_end_window1']:
                if a['checks'].get(key) and b['checks'].get(key) and not c['checks'].get(key):
                    lost.append(dict(id=row['id'],source=row['source']['source_family_id'],group=row['source']['group'],
                        action=c['action'],kind=c['kind'],phase='complete_event',noise=row['noise_index'],event=ei,why=[key]))
    def confirmed(items):
        keys=['source','group','action','kind','phase']
        return [x for x in items if any(all(x[k]==y[k] for k in keys) and x['noise']!=y['noise'] and set(x['why'])&set(y['why']) for y in items)]
    return dict(scope='full event phase and tail-window protection, not pooled averages',severe=severe,
                confirmed=confirmed(severe)+confirmed(lost),lost=lost,invalid=invalid)


def gate(previous2,previous,current,include_events=True):
    a,b,c=map(foundation_rows,[previous2,previous,current])
    failures=[];lost=[];better=[]
    for key,row in c.items():
        old=b[key];older=a[key];om,nm=old['metrics'],row['metrics'];why=[]
        if not np.isfinite([nm['joint_limit_max_deg'],nm['whole']['jitter_m_per_frame2']]).all():why.append('nonfinite')
        if nm['joint_limit_max_deg']>30 and nm['joint_limit_max_deg']>om['joint_limit_max_deg']+5:why.append('severe_joint')
        for name,absolute,delta in [('seam_root_m',.5,.1),('seam_joint_rms_deg',45,10)]:
            if nm.get(name,0)>absolute and nm.get(name,0)>om.get(name,0)+delta:why.append(name)
        # Preserve actual acquired contact quality even if another joint gate failed.
        stable_contact=all('contact_slip' not in x['metrics']['geometry_failures'] for x in [older,old])
        os,ns=om['whole']['contact_slip_percent'],nm['whole']['contact_slip_percent']
        ov,nv=om['whole']['contact_speed_m_s'],nm['whole']['contact_speed_m_s']
        if stable_contact and None not in [os,ns,ov,nv] and ns-os>20 and nv>1.5*ov:why.append('acquired_contact_collapse')
        if older['function_pass'] and old['function_pass'] and not row['function_pass']:
            lost.append(key);s0,s1=old['semantic'],row['semantic'];action=row['action']
            feature={'walk':'signed_displacement_m','backwalk':'signed_displacement_m','turn':'heading_change_deg',
                     'wave':'arm_range_deg','rhythm':'arm_range_deg','squat':'height_range_m','leg_raise':'leg_range_deg','bend':'leg_range_deg'}[action]
            if abs(s1[feature])<.5*abs(s0[feature]) or (action in ('walk','backwalk') and s1[feature]*s0[feature]<=0):why.append('acquired_function_collapse')
        if why:failures.append(dict(id=key,action=row['action'],source=row['source_index'],cell=row['cell'],noise=row['noise_index'],why=why))
        # A same-case quality gain needs retained function, lower contact speed,
        # no larger joint violations, and no more jitter. No stationary wins.
        if row['function_pass'] and old['function_pass'] and None not in [ov,nv] and nv<=.9*ov and nm['joint_limit_max_deg']<=max(2,om['joint_limit_max_deg']) and nm['whole']['jitter_m_per_frame2']<=1.1*om['whole']['jitter_m_per_frame2']:
            better.append(key)
    confirmed=[x for x in failures if 'nonfinite' in x['why'] or any(
        (x['action'],x['source'],x['cell'])==(y['action'],y['source'],y['cell']) and x['noise']!=y['noise'] and set(x['why'])&set(y['why']) for y in failures)]
    persistent={k for k,r in c.items() if r['joint_pass'] and b[k]['joint_pass']}
    cold=[k for k in persistent if k.endswith('_cold_text')]
    history=[k for k in persistent if k.endswith('_gt_accepted_text')]
    paired_better=[k for k in better if any(c[k]['action']==c[j]['action'] and c[k]['source_index']==c[j]['source_index'] and c[k]['cell']==c[j]['cell'] and c[k]['noise_index']!=c[j]['noise_index'] for j in better)]
    events=event_protection(previous2,previous,current) if include_events else dict(confirmed=[],invalid=[])
    stop=bool(confirmed or events['confirmed'] or events['invalid'])
    result=dict(function=sum(r['function_pass'] for r in c.values()),joint=sum(r['joint_pass'] for r in c.values()),
        previous_joint=sum(r['joint_pass'] for r in b.values()),lost_stable_function=lost,
        severe_instances=failures,confirmed_severe=confirmed,event_protection=events,stop=stop,
        paired_quality_improved=paired_better,learning_progress=bool(paired_better or sum(r['joint_pass'] for r in c.values())>sum(r['joint_pass'] for r in b.values())),
        closed_eligible=len(cold)>=2 and len(history)>=2 and not stop and not lost,
        persistent_cold=cold,persistent_history=history,model_qualified=False)
    write(Path(current)/'CONTINUATION_GATE.json',result);return result


def render(current,step):
    dest=OUT/'videos'/f'global{5000+step}'
    if (dest/'index.html').exists():return
    cases=[]
    for key in ['W_3_n0','S_3_n0','T_3_n0','X_3_n0']:
        cols=[]
        for name,folder in [('Fresh3500',D/'Fresh/eval3500'),('Fresh5000',D/'Fresh/eval5000'),(f'Fresh{5000+step}',current)]:
            path=folder/(key+'.json');r=read(path)
            cols.append(dict(name=name,artifact=r['artifact'],row_source=str(path)))
        cases.append(dict(id=key,cols=cols))
    write(dest/'CASES.json',cases)
    with Budget(OUT/'budget').timed('raw_videos',closing=True):
        run('render',['--config',OUT/'config.json','--output',dest],OUT/f'render{step}.log','4')


def closed(checkpoint,evaluation):
    cfg=read(OUT/'config.json')
    cfg.update(name='closed_loop_v1',parent=str(checkpoint),max_updates=492,gt_slots=8,
        budget_root=str(R2),savepoints=[0,8,32,64,128,250,492],refresh_steps=[0,250],
        teacher_history_frames=52,teacher_candidates_per_generation=200,baseline_evaluation=str(evaluation))
    cfg.pop('frozen_manifest',None)
    target=OUT/'closed';config=OUT/'closed_config.json';write(config,cfg)
    run('encode',['--config',config,'--output',target/'train'],target/'encode.log','4')
    pool=target/'pool0/POOL.json'
    if not pool.exists():run('collect',['--config',config,'--checkpoint',checkpoint,'--output',pool.parent,
        '--generation',1,'--teacher-history',52,'--max-candidates',200],target/'collect0.log','4')
    run('pipeline',['--config',config,'--output',target,'--pool',pool,'--training-ranks',4],target/'pipeline.log','0,1,2,3',1)
    return read(target/'PIPELINE_STATE.json')


def execute():
    reg=prepare();statefile=OUT/'STATE.json'
    state=read(statefile) if statefile.exists() else dict(status='PREPARING',completed_updates=0,global_step=5000,model_qualified=False)
    if state['status'] in ['STOPPED_FOR_HARM','STOPPED_TWO_SEGMENTS_NO_PROGRESS','BUDGET_TRANCHE_COMPLETE','CLOSED_LOOP_FINISHED']:return state
    cfg=OUT/'config.json';train=OUT/'train'
    if not (train/'TEXT_READY.json').exists():
        write(statefile,dict(state,status='PREPARING_FROZEN_TRAIN_DATA_AND_TEXT'))
        run('encode',['--config',cfg,'--output',train],OUT/'encode.log','4')
    evaluations=state.get('evaluations',[])
    last=D/'Fresh/eval5000';previous=D/'Fresh/eval3500';no_progress=0
    for step in [1000,5000,10000,15000,20000]:
        ev=OUT/f'eval{step}';cp=train/f'checkpoint{step}.pt'
        if not cp.exists():
            done=len(rows(train/'train.jsonl'))
            assert done<step
            elapsed=sum(x['seconds']*x.get('devices',1) for x in rows(OUT/'budget/GPU_TIME.jsonl'))/3600
            projected=(step-done)*reg['estimated_seconds_per_update']*4/3600
            if elapsed+projected>72*.75:
                state.update(status='BUDGET_TRANCHE_COMPLETE',reason='preserve25percent_eval_reserve');write(statefile,state);return state
            args=['--config',cfg,'--output',train,'--until',step]
            if done:args+=['--resume',train/f'checkpoint{done}.pt']
            state.update(status='TRAINING',target_updates=step,completed_updates=done,global_step=5000+done,
                estimated_training_end_epoch=time.time()+(step-done)*reg['estimated_seconds_per_update'],evaluations=evaluations)
            write(statefile,state);run('train',args,OUT/f'train{step}.log','0,1,2,3',4)
        state.update(status='EVALUATING',completed_updates=step,global_step=5000+step);write(statefile,state)
        if not (ev/'RESULT.json').exists():
            args=['--config',cfg,'--checkpoint',cp,'--output',ev]
            if step==1000:args+=['--foundation-only']
            run('evaluate',args,OUT/f'eval{step}.log','4')
        decision=gate(previous,last,ev)
        if step>=5000:
            render(ev,step)
            no_progress=0 if decision['learning_progress'] else no_progress+1
        entry=dict(step=step,global_step=5000+step,evaluation=str(ev),**decision)
        evaluations=[x for x in evaluations if x['step']!=step]+[entry]
        state.update(evaluations=evaluations,no_progress_segments=no_progress,selected_checkpoint=None)
        if decision['stop']:
            write(train/'STOP.json',dict(step=step,reason='confirmed_protected_harm',evidence=str(ev/'CONTINUATION_GATE.json')))
            state.update(status='STOPPED_FOR_HARM');write(statefile,state);return state
        if decision['closed_eligible'] and step>=5000:
            state.update(status='COLLECTING_CURRENT_STUDENT_HISTORY',selected_checkpoint=str(cp));write(statefile,state)
            result=closed(cp,ev);state.update(status='CLOSED_LOOP_FINISHED',closed_loop=result);write(statefile,state);return state
        if no_progress>=2:
            state.update(status='STOPPED_TWO_SEGMENTS_NO_PROGRESS');write(statefile,state);return state
        # The extra1000 point protects early severe harm; trend intervals stay5k.
        if step>=5000:previous,last=last,ev
        state.update(status='SAVEPOINT_COMPLETE');write(statefile,state)
    state.update(status='BUDGET_TRANCHE_COMPLETE',reason='20000additional_update_cap');write(statefile,state);return state


def main(mode):
    if mode=='prepare':result=prepare()
    elif mode=='status':result=read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else dict(status='NOT_STARTED')
    else:
        try:
            result=execute()
        except Exception as error:
            import traceback
            state=read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}
            state.update(status='EXECUTION_ERROR',error=str(error),traceback=traceback.format_exc())
            write(OUT/'STATE.json',state)
            deliver(state)
            raise
        deliver(result)
    print('FOUNDATION_CONTINUATION',result.get('status'),flush=True)
    return result


def deliver(state):
    """Terminal delivery, derived from receipts; completion never implies quality."""
    ledger=read(OUT/'EXPERIMENT_LEDGER.json')
    receipts=rows(OUT/'train/train.jsonl')
    ledger['experiments']['foundation_continuation'].update(status=state['status'],
        actual_updates=len(receipts),last_global_step=5000+len(receipts),model_qualified=False)
    write(OUT/'EXPERIMENT_LEDGER.json',ledger)
    times=rows(OUT/'budget/GPU_TIME.jsonl')
    used=sum(r['seconds']*r.get('devices',1) for r in times)/3600
    exposure=collections.Counter(x['dataset'] for r in receipts for x in r['receipts'])
    write(OUT/'EXECUTION_SUMMARY.json',dict(state=state,actual_updates=len(receipts),source_exposures=dict(exposure),
        gpu_hours=used,budget=read(OUT/'budget/LIMITS.json'),production_changed=False))
    lines=['# 固定配方基础学习接续','',f'实际状态：{state["status"]}。新增更新：{len(receipts)}；累计本次 fresh 基础更新：{5000+len(receipts)}。',
        '', '配对检查：3500→5000 的24项总预测损失全部变差；latent重建均值略改善，decoded-body与FK一致性变差。真正无CFG没有恢复功能与接触联合质量。此结论限定于固定DEV小面板。',
        '', '本阶段保留全部模型/master/Adam/调度，从5000接续。loss、数据比例、架构、有效batch与CFG未改。新阶段有独立TRAIN清单，不能冒称原阶段exact resume；旧5000结果与STOP均保留。',
        '', '|全局步数|功能/96|联合/96|跨噪声质量改善条目|确认严重退化|','|---|---:|---:|---:|---|']
    for r in state.get('evaluations',[]):
        lines.append(f'|{r["global_step"]}|{r["function"]}|{r["joint"]}|{len(r["paired_quality_improved"])}|{r["stop"]}|')
    lines+=['',f'本阶段已记GPU小时：{used:.3f}。完整阶段结果及失败见STATE.json、各eval目录/CONTINUATION_GATE.json；逐步更新与样本回执见train/train.jsonl。',
        '', '模型合格与扩大部署：尚不能宣称。仅达到训练步数不构成验收。若已执行闭环，其独立状态保存在closed/PIPELINE_STATE.json。未执行的刷新或复训不算方法失败。',
        '', '50k为总路线目标，本次最多接续20000步且受72GPU小时限制；不能把本阶段写成50k完成。生产权重未替换。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    links=['<!doctype html><meta charset="utf-8"><title>Foundation continuation</title><h1>基础学习接续</h1>',
           '<p>状态以 STATE.json 为准；以下为原始动作，无SONIC稳定化。</p><a href="REPORT.md">报告</a> · <a href="STATE.json">状态</a>']
    for index in sorted((OUT/'videos').glob('*/index.html')):
        links.append(f'<p><a href="{index.relative_to(OUT)}">{index.parent.name} 完整对照</a></p>')
    (OUT/'index.html').write_text('\n'.join(links))
