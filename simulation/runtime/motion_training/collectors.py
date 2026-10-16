"""Official ARDY continuations from the PRE-query accepted student boundary.

No GT splice or teacher privilege. Rejected full trajectories are retained.
"""
from pathlib import Path
from collections import defaultdict,Counter
import math
import numpy as np
import torch
from ardy.model import load_model
from . import runtime,sampling
from .engine import config as load_config
from .evaluation import initialize,set_text,event_plan,control,text_for,L
from ._compat.event_scores import phase,event_score
from ._compat.window_score import score_window
from .io import Budget,read,write,rows,sha,identity,copy_state,append
from .schema import VERSION,validate_record

def plans(group):
    if group=='T':return [('turn','turn',3),('walk','continue',4)]
    if group=='S':return [('walk','continue',2)]+event_plan(group)
    return event_plan(group)
def stage_key(group,action,kind,j):return '|'.join([group,action,kind,phase(action,kind,j)])
def required_stages():return sorted({stage_key(g,a,k,j) for g in ['W','S','T','X'] for a,k,n in plans(g) for j in range(n)})

def schedule_pool(pool,n,start=0):
    assert pool['status']=='COMPLETE_QUALIFIED_COVERAGE',pool['status']
    buckets=defaultdict(list)
    for r in pool['labels']:buckets[r['stage_key']].append(r)
    required=pool['required_stages'];assert set(required)<=set(buckets)
    # Interleave task groups in every prefix; sorting all S phases before W/X
    # would starve entire tasks during the first safety gate.
    groups={g:[key for key in required if key.startswith(g+'|')] for g in ['W','S','T','X']}
    required=[groups[g][i] for i in range(max(map(len,groups.values()))) for g in groups if i<len(groups[g])]
    out=[]
    for i in range(start,start+n):
        pair=[]
        for j in range(2):
            slot=2*i+j;key=required[slot%len(required)];records=buckets[key];r=dict(records[(slot//len(required))%len(records)])
            r.update(id=f'{r["id"]}:exposure{slot}',seed=202610035000+slot);pair.append(validate_record(r))
        out.append(pair)
    return out

def collect(checkpoint,output,config,generation=1,supplement_pool=None,groups=None,max_candidates=220,teacher_history=196,merge_pool=None,source_indices=None):
    cfg=load_config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'INCOMPATIBLE_OLD_COLLECTOR.json').exists()
    if (dest/'POOL.json').exists():return read(dest/'POOL.json')
    specs=[s for s in rows(L/'SPLITS.jsonl') if s['split']=='TRAIN']
    if groups:specs=[s for s in specs if s['group'] in groups]
    if source_indices is not None:specs=[s for s in specs if s['source_index'] in source_indices]
    assert 0<max_candidates<=220
    dev={s['source_family_id'] for s in rows(L/'SPLITS.jsonl') if s['split']!='TRAIN'}
    assert not dev&{s['source_family_id'] for s in specs}
    required=required_stages();write(dest/'PROTOCOL.json',dict(version='same_state_official_v2',required_stages=required,minimum_updates_for_two_slots=math.ceil(len(required)/2),generation=generation,max_candidates=max_candidates,groups=groups,teacher='ARDY-G1-RP-25FPS-Horizon52 public weights',extra_conditions=False,qualification='complete event semantics and every window geometry/contact/control/target roundtrip'))
    budget=Budget(cfg['budget_root']);cp=Path(checkpoint);cpsha=sha(cp);accepted=[];attempts=[];qcount=0
    if merge_pool:
        prior=read(merge_pool);assert prior['checkpoint_sha256']==cpsha
        accepted=list(prior['labels'])
    assert teacher_history in [52,196]
    write(dest/'TEACHER_HISTORY.json',dict(frames=teacher_history,merge_pool=merge_pool,extra_conditions=False))
    if supplement_pool:
        prior=read(supplement_pool);assert prior['checkpoint_sha256']==cpsha
        accepted=list(prior['labels']);specs=[s for s in specs if s['group'] in ['W','X']]
        write(dest/'SUPPLEMENT.json',dict(parent_pool=str(supplement_pool),difference='Explicit persistent walking intent, same new text visible to student and teacher; only missing W/X groups; no hidden spatial restart.',prompt='A person continuously walks straight ahead with a natural gait.',generation_is_same_parent_not_updated_weights=True))
    with budget.timed('collect:'+str(dest)):
        m,c,ck=runtime.load(cp,False);probe=sampling.NoiseProbe(m)
        from .acceleration import admitted_backend,stats
        backend=admitted_backend(c,cfg)
        official=load_model('g1',device=str(m.device),text_encoder=False).eval()
        assert not hasattr(official,'_sequence_kernel_post_window_hook'),'Teacher unexpectedly carries student adapter'
        teacher=sampling.OfficialSampler(official)
        teacher_id=identity(official.denoiser.model.state_dict());write(dest/'TEACHER.json',dict(model_hash=teacher_id,model_type=type(official.denoiser.model).__name__,student_hash=cpsha,different_from_student=teacher_id!=identity(m.denoiser.model.state_dict())))
        try:
          with torch.inference_mode(),sampling.capture_context() as captured:
            specs=sorted(specs,key=lambda s:(s['source_index'],s['group']))
            for spec in specs:
                if qcount+sum(n for _,_,n in plans(spec['group']))>max_candidates:break
                if all(key in {r['stage_key'] for r in accepted} for key in required if key.startswith(spec['group']+'|')):continue
                seed=int(spec['seeds'][0])+generation*100003;noise=sampling.fresh_noise(seed,100)
                runner,offset=initialize(m,c,probe,spec,cfg,captured);set_text(runner,spec,'walk',cfg);burn=[]
                if supplement_pool:runner.text=text_for(cfg,'A person continuously walks straight ahead with a natural gait.',m.device)
                for z in noise[:4]:
                    _,q,_,_,_=runner.step(z);burn.append(q)
                entry=dest/f'{spec["group"]}_{spec["source_index"]}_entry.pt';torch.save(copy_state(sampling.payload(runner,captured,runner.text)),entry)
                np.savez_compressed(entry.with_suffix('.npz'),qpos=np.concatenate(burn),fps=25)
                chain=[];allq=[];allph=[];wi=0
                for ei,(action,kind,count) in enumerate(plans(spec['group'])):
                    set_text(runner,spec,action,cfg);state={};heading=float(runner.heading[0]);event_records=[];eq=[];ep=[];ew=[];controls=[]
                    if supplement_pool and action=='walk':runner.text=text_for(cfg,'A person continuously walks straight ahead with a natural gait.',m.device)
                    for j in range(count):
                        ctl,state,intent=control(runner,spec,offset,action,kind,j,state)
                        item=sampling.payload(runner,captured,runner.text,ctl);before=identity([item['physical'],item['translation'],item['previous']])
                        ticket=budget.ticket('teacher',generation=generation,source=spec['source_family_id'],window=wi);qcount+=1
                        _,q,dec,world,_=teacher.step(runner,noise[4+wi],history_frames=teacher_history,control=ctl)
                        target,prefix,contract=sampling.target_check(m,world,item['previous'],item['translation'],dec)
                        checks,ph=score_window(m,dec,q,item['physical'],ck['config'],action,ctl)
                        item['target'].update(target_latent=target.detach(),decoder_prefix=prefix.detach(),normalized=dec.detach(),noise=noise[4+wi].to(m.device),intent=intent)
                        key=f'g{generation}:{spec["group"]}:{spec["source_index"]}:w{wi}';path=dest/(key.replace(':','_')+'.pt');torch.save(copy_state(item),path)
                        r=dict(schema=VERSION,id=key,source_family_id=spec['source_family_id'],dataset=spec['dataset'],split='TRAIN',crop_start=spec['crop_start'],crop_end=spec['crop_start']+104,feature_path=spec['feature_path'],fps=25,skeleton='g1',representation='physical414',codec_version='g1_history_codec_104_v1',normalizer_version=m.motion_contract,H=0,G=52,F=0,supervision='same_state_teacher',label=str(path),label_sha256=sha(path),collector_checkpoint_sha256=cpsha,teacher_model_hash=teacher_id,teacher_extra_conditions=[],generation=generation,qualified=False,stage_key=stage_key(spec['group'],action,kind,j),group=spec['group'],action=action,kind=kind,phase=phase(action,kind,j),relative_event_window=j,seed=seed+wi,checks=checks,target_contract=contract,pre_query_boundary_hash=before,pre_query_history_frames=int(item['physical'].shape[1]),teacher_history_frames=min(196,item['physical'].shape[1]),teacher_query=ticket)
                        r['teacher_history_frames']=min(teacher_history,item['physical'].shape[1])
                        event_records.append(r);ew.append(checks);eq.append(q);ep.append(ph);controls.append(ctl);wi+=1
                    score=event_score(np.concatenate(eq),np.concatenate(ep),ew,action,kind,heading,controls)
                    good=score['all_pass'] and all(r['target_contract']['passed'] for r in event_records)
                    for r in event_records:
                        r.update(qualified=bool(good),event_score=score,rejection_reasons=[] if good else dict(function=[k for k,v in score['checks'].items() if not v],geometry=not score['geometry_pass'],target_contract=not r['target_contract']['passed']))
                        attempts.append(r);append(dest/'attempts.jsonl',r)
                        if good:accepted.append(validate_record(r))
                    chain.append(dict(event=ei,action=action,kind=kind,score=score,qualified=good));allq+=eq;allph+=ep
                artifact=dest/f'{spec["group"]}_{spec["source_index"]}_teacher_chain.npz';np.savez_compressed(artifact,qpos=np.concatenate(allq),physical=np.concatenate(allph),fps=25)
                write(artifact.with_suffix('.json'),dict(source=spec,seed=seed,events=chain,student_entry=str(entry),artifact=str(artifact)))
                print('COLLECT',spec['group'],spec['source_index'],'candidates',qcount,'qualified',len(accepted),flush=True)
                if set(required)<={r['stage_key'] for r in accepted}:break
        finally:
            backend_stats=stats(c);teacher.close();probe.close()
    coverage=Counter(r['stage_key'] for r in accepted);missing=sorted(set(required)-set(coverage))
    out=dict(schema='g1_training_pool_v2',generation=generation,checkpoint=str(cp),checkpoint_sha256=cpsha,teacher_model_hash=teacher_id,labels=accepted,candidate_windows=qcount,qualified=len(accepted),required_stages=required,coverage=dict(coverage),missing_stages=missing,minimum_updates_for_two_slots=math.ceil(len(required)/2),status='INCOMPLETE_MISSING_QUALIFIED_STAGES' if missing else 'COMPLETE_QUALIFIED_COVERAGE',attempts_path=str(dest/'attempts.jsonl'),extra_conditions=False)
    out.update(execution_backend=backend,kernel_stats=backend_stats,supplement_parent=supplement_pool);write(dest/'POOL.json',out);return out

def history_ab(config,output):
    """Targeted same-boundary teacher history intervention; no resampled input."""
    from .history import HistoryContext
    cfg=load_config(config);dest=Path(output);protocol=read(dest/'PROTOCOL.json')
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    pool=read(protocol['parent_pool']);attempts=rows(pool['attempts_path']);source=Path(protocol['parent_pool']).parent
    group=protocol['source_group'];index=protocol['source_index']
    chain=read(source/f'{group}_{index}_teacher_chain.json');baseline=np.load(source/f'{group}_{index}_teacher_chain.npz')['qpos']
    entries={int(r['id'].split(':w')[-1]):r for r in attempts if r['group']==group and f':{index}:w' in r['id']}
    budget=Budget(cfg['budget_root']);results=[];labels=[]
    with budget.timed('teacher_history_ab'):
        m,c,ck=runtime.load(pool['checkpoint'],False);probe=sampling.NoiseProbe(m)
        from .acceleration import admitted_backend
        backend=admitted_backend(c,cfg)
        official=load_model('g1',device=str(m.device),text_encoder=False).eval();teacher=sampling.OfficialSampler(official)
        teacher_id=identity(official.denoiser.model.state_dict());assert teacher_id==pool['teacher_model_hash']
        try:
          with torch.inference_mode():
            for event,start in zip(protocol['events'],protocol['start_windows']):
                initial=copy_state(torch.load(entries[start]['label'],map_location='cpu',weights_only=False),m.device)
                def restore():
                    runner=sampling.Native(m,c,probe,initial['target']['text'])
                    HistoryContext(initial['chunks'],initial['physical'],pool['checkpoint_sha256']).prime(m,c,runner.text)
                    for key in ['physical','previous','heading','translation']:setattr(runner,key,initial[key].clone())
                    runner.i=initial['meta']['depth'];return runner
                runner=restore();budget.ticket('teacher',experiment='history_ab_replay',event=event)
                _,q,_,_,_=teacher.step(runner,initial['target']['noise'],history_frames=196,control={})
                delta=float(np.abs(q-baseline[start*52:(start+1)*52]).max())
                write(dest/f'event{event}_REPLAY.json',dict(max_qpos_difference=delta,passed=delta<=1e-6))
                assert delta<=1e-6,('saved boundary does not replay',event,delta)
                with sampling.capture_context() as captured:
                    runner=restore();captured.extend(copy_state(initial['chunks'],m.device));qs=[];phs=[];windows=[];records=[];heading=float(runner.heading[0])
                    for j in range(4):
                        old=entries[start+j];noise=torch.load(old['label'],map_location='cpu',weights_only=False)['target']['noise'].to(m.device)
                        item=sampling.payload(runner,captured,runner.text,{});ticket=budget.ticket('teacher',experiment='history_ab52',event=event,window=j)
                        _,q,dec,world,_=teacher.step(runner,noise,history_frames=52,control={})
                        target,prefix,contract=sampling.target_check(m,world,item['previous'],item['translation'],dec)
                        check,ph=score_window(m,dec,q,item['physical'],ck['config'],old['action'],{})
                        item['target'].update(target_latent=target.detach(),decoder_prefix=prefix.detach(),normalized=dec.detach(),noise=noise,intent=dict(kind=old['kind']))
                        path=dest/f'event{event}_w{j}.pt';torch.save(copy_state(item),path)
                        record=dict(old,id=old['id']+':teacherH52',label=str(path),label_sha256=sha(path),teacher_history_frames=52,checks=check,target_contract=contract,teacher_query=ticket,pre_query_boundary_hash=identity([item['physical'],item['translation'],item['previous']]),pre_query_history_frames=int(item['physical'].shape[1]),qualified=False)
                        records.append(record);qs.append(q);phs.append(ph);windows.append(check)
                score=event_score(np.concatenate(qs),np.concatenate(phs),windows,entries[start]['action'],entries[start]['kind'],heading,[{}]*4)
                good=score['all_pass'] and all(r['target_contract']['passed'] for r in records)
                for r in records:
                    r.update(qualified=good,event_score=score,rejection_reasons=[] if good else dict(function=[k for k,v in score['checks'].items() if not v],geometry=not score['geometry_pass'],target_contract=not r['target_contract']['passed']))
                    append(dest/'attempts.jsonl',r)
                    if good:labels.append(validate_record(r))
                artifact=dest/f'event{event}_H52.npz';np.savez_compressed(artifact,qpos=np.concatenate(qs),physical=np.concatenate(phs),fps=25)
                result=dict(event=event,start_window=start,baseline_score=chain['events'][event]['score'],H52_score=score,qualified=good,artifact=str(artifact),records=records)
                write(dest/f'event{event}_RESULT.json',result);results.append(result);print('HISTORY_AB',event,'qualified',good,score['checks'],flush=True)
        finally:teacher.close();probe.close()
    result=dict(protocol=protocol,checkpoint=pool['checkpoint'],checkpoint_sha256=pool['checkpoint_sha256'],results=results,labels=labels,teacher_candidates=5*len(protocol['events']),backend=backend,model_qualified=False)
    write(dest/'RESULT.json',result);return result

def repair_ab_prefix(config,output):
    """Restore omitted serialized prefix; no output regeneration or updates.

    First AB generation restored runtime via HistoryContext, whose direct
    canonical function reference bypassed the capture hook. The runtime was
    correct (exact baseline replay); saved new labels omitted the old chunks.
    Preserve those files and write corrected payloads with the saved prefix.
    """
    dest=Path(output)
    if (dest/'PREFIX_REPAIR.json').exists():return read(dest/'PREFIX_REPAIR.json')
    result=read(dest/'RESULT.json');protocol=result['protocol'];pool=read(protocol['parent_pool']);attempts=rows(pool['attempts_path']);records=[]
    for event,start in zip(protocol['events'],protocol['start_windows']):
        origin=next(r for r in attempts if r['group']=='X' and r['id'].endswith(f':0:w{start}'))
        initial=torch.load(origin['label'],map_location='cpu',weights_only=False)
        part=read(dest/f'event{event}_RESULT.json');newrows=[]
        for j,r in enumerate(part['records']):
            oldpath=Path(r['label']);item=torch.load(oldpath,map_location='cpu',weights_only=False)
            assert len(item['chunks'])==j,('unexpected saved capture',j,len(item['chunks']))
            assert item['physical'].shape[1]==initial['physical'].shape[1]+52*j
            before=identity([item['physical'],item['previous'],item['target']])
            item['chunks']=copy_state(initial['chunks'])+item['chunks']
            assert len(item['chunks'])==item['meta']['depth']
            assert before==identity([item['physical'],item['previous'],item['target']])
            path=oldpath.with_name(oldpath.stem+'_with_prefix.pt');torch.save(item,path)
            fixed=dict(r,label=str(path),label_sha256=sha(path),superseded_label=str(oldpath),prefix_fixed_without_regeneration=True)
            records.append(fixed);newrows.append(fixed)
        part['records']=newrows;write(dest/f'event{event}_RESULT_WITH_PREFIX.json',part)
    qualified=[r for r in records if r['qualified']];merged=list(pool['labels'])+qualified;coverage=Counter(r['stage_key'] for r in merged);missing=sorted(set(required_stages())-set(coverage))
    newpool=dict(pool,labels=merged,coverage=dict(coverage),missing_stages=missing,qualified=len(merged),status='INCOMPLETE_MISSING_QUALIFIED_STAGES' if missing else 'COMPLETE_QUALIFIED_COVERAGE',parent_pool=protocol['parent_pool'],new_history_ab_candidates=10,corrected_payloads=True)
    write(dest/'POOL.json',newpool);write(dest/'PREFIX_REPAIR.json',dict(status='CORRECTED_WITHOUT_REGENERATION',new_qualified_labels=len(qualified),runtime_replay_exact=True,original_payloads_excluded_from_training=True,model_updates=0,affected_records=records,merged_pool=str(dest/'POOL.json')))
    print('PREFIX_REPAIRED',len(qualified),'missing',missing,flush=True)

def audit_pool(config,output,parent=None):
    """Admission from actual payloads, not only a COMPLETE string."""
    cfg=load_config(config);dest=Path(output);pool=read(dest/'POOL.json')
    assert pool['checkpoint_sha256']==sha(parent or cfg['parent'])
    assert pool['status']=='COMPLETE_QUALIFIED_COVERAGE'
    dev={r['source_family_id'] for r in rows(L/'SPLITS.jsonl') if r['split']!='TRAIN'}
    audited=[]
    for r in pool['labels']:
        validate_record(r);assert r['source_family_id'] not in dev and not r['teacher_extra_conditions']
        assert r['qualified'] and r['event_score']['all_pass'] and r['target_contract']['passed']
        assert r['collector_checkpoint_sha256']==pool['checkpoint_sha256'] and r['label_sha256']==sha(r['label'])
        item=torch.load(r['label'],map_location='cpu',weights_only=False);chunks=item['chunks'];n=item['physical'].shape[1]
        assert len(chunks)==item['meta']['depth'] and sum(t.shape[1]*4 for t,_ in chunks)==n
        assert all(not kw['include_synthetic_generation'] for _,kw in chunks)
        assert len({kw['chunk_id'] for _,kw in chunks})==len(chunks)
        assert r['pre_query_history_frames']==n and item['target']['target_latent'].shape==(1,13,148)
        assert torch.isfinite(item['target']['target_latent']).all()
        if r['kind'] in ['text','continue','free_resume','release_hold']:assert not item['target']['control']
        audited.append(dict(id=r['id'],history_frames=n,prime_chunks=len(chunks),teacher_history_frames=r['teacher_history_frames'],label_sha256=r['label_sha256'],stage=r['stage_key']))
    schedule=schedule_pool(pool,16);stages={r['stage_key'] for pair in schedule for r in pair}
    assert set(required_stages())<=stages
    early={r['group'] for pair in schedule[:2] for r in pair};assert early==set('WSTX')
    result=dict(passed=True,pool_sha256=sha(dest/'POOL.json'),labels=len(audited),all_stages=len(stages),minimum_updates=16,first_two_updates_cover_groups=sorted(early),sample_ids_unique=len({r['id'] for r in pool['labels']})==len(pool['labels']),all_payload_prefixes_complete=True,dev_overlap=False,extra_teacher_condition=False,parent_sha256=pool['checkpoint_sha256'],audited=audited)
    assert result['sample_ids_unique'];write(dest/'ADMISSION.json',result);print('POOL_ADMISSION_PASS',result['labels'],result['all_stages'],flush=True);return result
