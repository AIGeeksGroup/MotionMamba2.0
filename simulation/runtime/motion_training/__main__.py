"""Explicit CLI; importing the package never performs work."""
import argparse
from pathlib import Path
from .io import read,write
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    z=sub.add_parser('startpoint');z.add_argument('--mode',required=True,choices=['prepare','verify','warm','calibrate','execute','fresh','summarize','delivery','render-warm','complete-route']);z.add_argument('--arm',choices=['P_reset','N_reset'])
    z=sub.add_parser('learning-transfer');z.add_argument('--config',required=True);z.add_argument('--output',required=True)
    z=sub.add_parser('foundation-continue');z.add_argument('--mode',choices=['prepare','run','status'],required=True)
    tr=sub.add_parser('train');tr.add_argument('--config',required=True);tr.add_argument('--output',required=True);tr.add_argument('--until',type=int);tr.add_argument('--resume');tr.add_argument('--parent');tr.add_argument('--pool');tr.add_argument('--category',choices=['B','C','checks'])
    for name in ['prepare','encode','contracts','migration','pipeline','kernel-gate','kernel-performance','kernel-cold-suite','additional-eval','gradient-summary','distributed-summary','graph-contract','teacher-history-ab','repair-ab-prefix','pool-audit','paired-check','retention-plot','render']:
        z=sub.add_parser(name);z.add_argument('--config',required=True);z.add_argument('--output',required=True)
        if name=='pipeline':z.add_argument('--pool',required=True);z.add_argument('--parent');z.add_argument('--status-only',action='store_true');z.add_argument('--training-ranks',type=int,choices=[1,4,8],default=1)
    z=sub.add_parser('evaluate');z.add_argument('--checkpoint',required=True);z.add_argument('--output',required=True);z.add_argument('--config',required=True);z.add_argument('--sentinel',action='store_true');z.add_argument('--foundation-only',action='store_true')
    z=sub.add_parser('collect');z.add_argument('--checkpoint',required=True);z.add_argument('--output',required=True);z.add_argument('--config',required=True);z.add_argument('--generation',type=int,required=True);z.add_argument('--supplement-pool');z.add_argument('--groups',nargs='+',choices=['W','S','T','X']);z.add_argument('--max-candidates',type=int,default=220);z.add_argument('--teacher-history',type=int,choices=[52,196],default=196);z.add_argument('--merge-pool')
    z=sub.add_parser('compare');z.add_argument('--parent',required=True);z.add_argument('--current',required=True)
    z=sub.add_parser('kernel-cold-worker');z.add_argument('--config',required=True);z.add_argument('--output',required=True);z.add_argument('--backend',choices=['reference','optimized'],required=True);z.add_argument('--seed',type=int,required=True)
    sub.choices['collect'].add_argument('--source-indices',nargs='+',type=int)
    z=sub.add_parser('probe-load');z.add_argument('--checkpoint',required=True);z.add_argument('--output',required=True)
    z=sub.add_parser('verify-worker');z.add_argument('--config',required=True);z.add_argument('--output',required=True);z.add_argument('--mode',required=True);z.add_argument('--steps',type=int,required=True);z.add_argument('--resume');z.add_argument('--pool')
    a=p.parse_args();kw=vars(a).copy();kw.pop('command')
    from .distributed import setup,world
    setup()
    if world()>1 and a.command not in ['train','verify-worker']:
        raise ValueError('Distributed entry is only implemented for training and real-update verification')
    if a.command=='foundation-continue':
        from .foundation_continuation import main as continuation
        continuation(**kw)
    elif a.command=='learning-transfer':
        from .learning_transfer import execute
        execute(**kw)
    elif a.command=='startpoint':
        from .startpoint import main as startpoint
        startpoint(**kw)
    elif a.command=='train':
        from .engine import train
        train(kw.pop('config'),kw.pop('output'),**kw)
    elif a.command=='prepare':
        from .engine import config
        from .data import prepare
        d=prepare(config(a.config),a.output);print('PREPARED',d['planned_exposures'],d['unique_families'])
    elif a.command=='probe-load':
        from . import runtime
        from .io import Budget,sha
        with Budget(a.output).timed('probe_load'):
            m,c,ck=runtime.load(a.checkpoint,True);opt=runtime.optimizer(m,ck)
            write(Path(a.output)/'LOAD.json',dict(checkpoint_sha256=sha(a.checkpoint),imports=runtime.import_sources(),dims=[m.denoiser.model.nframe_root_dim,m.denoiser.model.latent_embedding_dim],config=ck['config'],optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups]))
        print('LOAD_PASS')
    elif a.command=='verify-worker':
        from .validation import verify_worker
        verify_worker(**kw)
    elif a.command=='kernel-gate':
        from .acceleration import gate
        gate(**kw)
    elif a.command=='kernel-performance':
        from .acceleration import performance
        performance(**kw)
    elif a.command=='kernel-cold-suite':
        from .acceleration import cold_suite
        cold_suite(**kw)
    elif a.command=='kernel-cold-worker':
        from .acceleration import cold_worker
        cold_worker(**kw)
    elif a.command=='additional-eval':
        from .evaluation import additional
        additional(**kw)
    elif a.command=='gradient-summary':
        from .validation import gradient_summary
        gradient_summary(**kw)
    elif a.command=='distributed-summary':
        from .validation import distributed_summary
        distributed_summary(**kw)
    elif a.command=='graph-contract':
        from .validation import graph_contract
        graph_contract(**kw)
    elif a.command=='teacher-history-ab':
        from .collectors import history_ab
        history_ab(**kw)
    elif a.command=='repair-ab-prefix':
        from .collectors import repair_ab_prefix
        repair_ab_prefix(**kw)
    elif a.command=='pool-audit':
        from .collectors import audit_pool
        audit_pool(**kw)
    elif a.command=='paired-check':
        from .validation import paired_check
        paired_check(**kw)
    elif a.command=='render':
        from .visualization import render
        render(**kw)
    elif a.command=='retention-plot':
        from .retention_plot import render
        render(**kw)
    elif a.command=='compare':
        from .evaluation import compare
        result=compare(**kw);print({k:len(v) if isinstance(v,list) else v for k,v in result.items()})
    else:
        from importlib import import_module
        mod=import_module('.'+{'evaluate':'evaluation','collect':'collectors','contracts':'validation','migration':'validation','encode':'preparation','pipeline':'pipeline'}[a.command],__package__)
        getattr(mod,a.command)(**kw)
if __name__=='__main__':
    try:main()
    finally:
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
