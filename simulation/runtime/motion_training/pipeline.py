"""Durable bounded closed-loop progression, never bypassing qualification.

Pass a reviewed first pool explicitly. This entry does not hide missing stages,
clear STOP, renew budgets, or start a different loss recipe.
"""
from pathlib import Path
import subprocess,os
from .engine import config as load_config
from .io import read,write,rows,sha

def pipeline(config,output,pool,parent=None,status_only=False,training_ranks=1):
    cfg=load_config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    assert cfg['name']=='closed_loop_v1'
    pool=Path(pool);state=read(pool);start=parent or cfg['parent']
    status=dict(parent=str(start),pool=str(pool),pool_sha256=sha(pool),pool_status=state['status'],missing_stages=state['missing_stages'],status='READY' if state['status']=='COMPLETE_QUALIFIED_COVERAGE' else 'INCOMPLETE_MISSING_QUALIFIED_STAGES',model_qualified=False)
    if (dest/'train/STOP.json').exists():
        stopped=read(dest/'train/STOP.json');status.update(status='STOPPED_FOR_HARM',step=stopped['step'],evidence=stopped['evidence'])
    write(dest/'PIPELINE_STATE.json',status)
    if status_only or status['status']!='READY':print('PIPELINE',status,flush=True);return status
    launcher=Path(__file__).resolve().parents[2]/'examples/train_motion.sh'
    def run(command,args):
        cmd=['bash',str(launcher),command,*map(str,args)]
        env=dict(os.environ,MOTION_NPROC=str(training_ranks if command=='train' else 1))
        subprocess.run(cmd,check=True,env=env)
    assert state['checkpoint_sha256']==sha(start),'Pool and initial student checkpoint differ'
    from .collectors import audit_pool
    audit_pool(config,str(pool.parent),parent=str(start))
    base=Path(cfg.get('baseline_evaluation',str(dest/'baseline')))
    if not (base/'RESULT.json').exists():run('evaluate',['--config',config,'--checkpoint',start,'--output',base])
    train=dest/'train';done=len(rows(train/'train.jsonl'));last=None
    gates=[s for s in cfg['savepoints'] if s>0];refresh_steps=[s for s in cfg.get('refresh_steps',[64]) if s>0]
    assert set(refresh_steps)<=set(gates)
    for step in gates:
        if done<step:
            if done in refresh_steps:
                from .collectors import collect
                refreshed=dest/f'pool{done}';state=collect(str(train/f'checkpoint{done}.pt'),str(refreshed),config,generation=2+refresh_steps.index(done),teacher_history=cfg.get('teacher_history_frames',196),max_candidates=cfg.get('teacher_candidates_per_generation',220));pool=refreshed/'POOL.json'
                if state['status']!='COMPLETE_QUALIFIED_COVERAGE':
                    status.update(status='INCOMPLETE_REFRESH_COVERAGE',missing_stages=state['missing_stages'],step=done);write(dest/'PIPELINE_STATE.json',status);return status
                audit_pool(config,str(refreshed),parent=str(train/f'checkpoint{done}.pt'))
            args=['--config',config,'--output',train,'--until',step,'--category','C','--pool',pool]
            args+=['--resume',train/f'checkpoint{done}.pt'] if done else ['--parent',start]
            status.update(status='TRAINING_SEGMENT',completed_updates=done,target_step=step,active_pool=str(pool));write(dest/'PIPELINE_STATE.json',status)
            run('train',args);done=step
        current=dest/f'eval{step}'
        status.update(status='EVALUATING_SAVEPOINT',completed_updates=done,target_step=step);write(dest/'PIPELINE_STATE.json',status)
        if not (current/'RESULT.json').exists():run('evaluate',['--config',config,'--checkpoint',train/f'checkpoint{step}.pt','--output',current])
        from .evaluation import compare
        comparison=compare(base,current)
        if cfg.get('protected_evaluation'):
            write(current/'PARENT_COMPARISON.json',comparison)
            protected=compare(cfg['protected_evaluation'],current);write(current/'PROTECTED_COMPARISON.json',protected)
            comparison['protected_reference']=cfg['protected_evaluation'];comparison['protected_stop']=protected['stop'];comparison['stop']=comparison['stop'] or protected['stop']
            write(current/'COMPARISON.json',comparison)
        if comparison['stop']:
            write(train/'STOP.json',dict(status='STOPPED_FOR_HARM',step=step,evidence=str(current/'COMPARISON.json')))
            status.update(status='STOPPED_FOR_HARM',step=step);write(dest/'PIPELINE_STATE.json',status);return status
        last=str(current/'RESULT.json')
    status.update(status='BOUNDED_EXECUTION_COMPLETE',step=done,last_evaluation=last,model_qualified=False,expansion_ready=False)
    write(dest/'PIPELINE_STATE.json',status);return status
