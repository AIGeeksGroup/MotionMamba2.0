"""Opt-in exact inference optimization; backward always has an eager route."""
import copy,time
from pathlib import Path
import numpy as np
import torch
from .io import read,write,identity,sha,Budget,append

def configure(client,backend='reference'):
    assert backend in ['reference','optimized'];enabled=backend=='optimized'
    for b in [client.controller.root_block,client.controller.body_block]:
        b.inference_no_diagnostics=enabled;b.inference_no_pending=enabled;b.inference_cfg_parallel=False
        b.inference_no_commit_diagnostics=enabled;b.inference_serial_cfg_graph=enabled
        stack=b.sequence_stack;stack.inference_graph=enabled;stack.inference_graph_shared_kv=enabled;stack.inference_fast_graph_checks=enabled
        for layer in stack.layers:
            layer.mixer._mamba_wrapper.inference_fused_adt=enabled
            layer.mixer.attention.inference_cache=enabled
            layer.mixer.inference_prepared_attention=enabled;layer.mixer.inference_dual_attention=False
            layer.inference_fused_adaln=False;layer.inference_fused_pointwise=enabled
    return backend
def stats(client):
    return {name:dict(replays=getattr(getattr(b.sequence_stack,'_inference_graphs',None),'replays',0),fallbacks=getattr(getattr(b.sequence_stack,'_inference_graphs',None),'fallbacks',0),build_events=getattr(getattr(b.sequence_stack,'_inference_graphs',None),'events',[])) for name,b in [('root',client.controller.root_block),('body',client.controller.body_block)]}
def admitted_backend(client,cfg):
    gatefile=Path(cfg.get('kernel_gate_path',str(Path(cfg['budget_root'])/'kernel/GATE.json')))
    passed=gatefile.exists() and read(gatefile).get('passed',False)
    backend='optimized' if passed else 'reference'
    configure(client,backend)
    return dict(backend=backend,gate=str(gatefile) if passed else None,gradient_path='reference')
def state_tree(x):
    if torch.is_tensor(x) or isinstance(x,np.ndarray):return x
    if isinstance(x,dict):return {k:state_tree(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [state_tree(v) for v in x]
    if hasattr(x,'__dict__'):return state_tree(vars(x))
    return x
def snapshot(runner):
    ctl=runner.c
    return copy.deepcopy(dict(root=ctl.root_block._sequence_state,body=ctl.body_block._sequence_state,runner={k:getattr(runner,k) for k in ['translation','heading','previous','physical','i','text','mask']},ctl={k:getattr(ctl,k) for k in ['_completed_windows','_accepted_commits','_next_chunk_id','_last_text_feat','_last_text_pad_mask']}))
def restore(runner,s):
    runner.c.restore_training_snapshot((s['root'],s['body']))
    for k,v in s['runner'].items():setattr(runner,k,copy.deepcopy(v))
    for k,v in s['ctl'].items():setattr(runner.c,k,copy.deepcopy(v))
def gate(config,output):
    from . import runtime,sampling
    from .engine import config as load_config
    from .evaluation import text_for,R
    cfg=load_config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'GATE.json').exists():return read(dest/'GATE.json')
    with Budget(cfg['budget_root']).timed('kernel_gate'):
        m,c,ck=runtime.load(cfg['parent'],False);p=sampling.NoiseProbe(m);text=text_for(cfg,read(R/'TEXTS.json')['walk_0']['prompt'],m.device);wave=text_for(cfg,read(R/'TEXTS.json')['wave_0']['prompt'],m.device)
        checks=[];timing=[];reference=[];states=[]
        try:
          with torch.inference_mode():
            for seed in [202610034801,202610034802]:
              for backend in ['reference','optimized']:
                configure(c,backend);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);runner=sampling.Native(m,c,p,text);noise=sampling.fresh_noise(seed,20);planner={}
                for i,z in enumerate(noise):
                    runner.text=wave if i in [6,7,8] else text;ctl={}
                    if i in [10,11,12]:ctl,planner,_=sampling.command(runner,'stop','stop',planner,i-10)
                    if i in [15,16,17]:ctl,planner,_=sampling.command(runner,'restart','restart',planner,i-15)
                    if i==19:ctl=dict(heading=float(runner.heading[0])+.3)
                    torch.cuda.synchronize();t=time.perf_counter();_,q,dec,_,_=runner.step(z,ctl);torch.cuda.synchronize()
                    hashes=dict(qpos=identity(q),motion=identity(dec),state=identity(state_tree(snapshot(runner))),rng=identity(runtime.rng()),commits=runner.c._accepted_commits)
                    timing.append(dict(seed=seed,window=i,backend=backend,seconds=time.perf_counter()-t))
                    if backend=='reference':reference.append(hashes)
                    else:
                        parent=reference[(0 if seed==202610034801 else 20)+i];same={k:parent[k]==v for k,v in hashes.items()};checks.append(dict(seed=seed,window=i,checks=same,passed=all(same.values())))
                        if not all(same.values()):
                            write(dest/'GATE.json',dict(passed=False,checks=checks,timing=timing,stats=stats(c)));raise AssertionError(checks[-1])
                    if i==4 and backend=='optimized':states.append((snapshot(runner),runtime.rng(),noise[5]))
                print('KERNEL',seed,backend,'20 windows',flush=True)
            # Unpublished rollback: same state and RNG regenerate the same full output.
            s,rng,z=states[-1];restore(runner,s);runtime.restore_rng(rng);_,q,dec,_,_=runner.step(z);first=identity([q,dec,state_tree(snapshot(runner)),runtime.rng()])
            restore(runner,s);runtime.restore_rng(rng);_,q,dec,_,_=runner.step(z);assert first==identity([q,dec,state_tree(snapshot(runner)),runtime.rng()])
            graph=stats(c);assert sum(v['replays'] for v in graph.values())>0
            result=dict(passed=True,checkpoint=str(cfg['parent']),checkpoint_sha256=sha(cfg['parent']),checks=checks,rollback_exact=True,reset_two_seeds=True,timing=timing,stats=graph,scope='two seeds x20 recursive windows incl W4/text/stop/release/restart/heading; qpos/motion/state/RNG/commit exact',gradient_backend='reference',performance_scope='generation/decode/state hash measured here; final timing separately excludes hash instrumentation')
            write(dest/'GATE.json',result)
        finally:p.close()
    return result

def performance(config,output):
    from . import runtime,sampling
    from .engine import config as load_config
    from .evaluation import text_for,R
    cfg=load_config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    with Budget(cfg['budget_root']).timed('paired_performance'):
        load_start=time.perf_counter();m,c,ck=runtime.load(cfg['parent'],False);torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
        probe=sampling.NoiseProbe(m);text=text_for(cfg,read(R/'TEXTS.json')['walk_0']['prompt'],m.device);timings=[];fallbacks=[];activation=[]
        try:
          with torch.inference_mode():
            runner=sampling.Native(m,c,probe,text);noises=sampling.fresh_noise(202610037107,8);configure(c,'reference')
            for z in noises[:5]:runner.step(z)
            boundary=snapshot(runner);rng=runtime.rng();expected=None
            for name in ['reference','optimized']:
                configure(c,name);restore(runner,boundary);runtime.restore_rng(rng);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                _,q,d,_,_=runner.step(noises[5]);torch.cuda.synchronize();activation.append(dict(backend=name,ms=1000*(time.perf_counter()-tick),peak_allocated=torch.cuda.max_memory_allocated(),scope='first backend activation at same W5 boundary; optimized includes graph build, reference has completed history warmup; not symmetric process-cold comparison'))
                value=identity([q,d,state_tree(snapshot(runner)),runtime.rng()])
                if expected is None:expected=value
                else:assert expected==value
            for pair in range(12):
              for name in (['reference','optimized'] if pair%2==0 else ['optimized','reference']):
                configure(c,name);restore(runner,boundary);runtime.restore_rng(rng);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                runner.step(noises[5]);torch.cuda.synchronize();elapsed=time.perf_counter()-t
                row=dict(pair=pair,backend=name,ms=elapsed*1000,peak_allocated=torch.cuda.max_memory_allocated());timings.append(row);append(dest/'timings.jsonl',row)
            # Real masked stack input exercises the eager fallback, not a mocked flag.
            block=c.controller.root_block;stack=block.sequence_stack;state=block._sequence_state;d=stack.layers[0].d_model
            x=torch.randn(1,13,d,device=m.device);cond=torch.randn(1,d,device=m.device);mask=torch.ones(1,13,device=m.device,dtype=torch.bool);mask[:,-1]=False
            outputs=[]
            for name in ['reference','optimized']:
                configure(c,name)
                with torch.autocast('cuda',dtype=torch.bfloat16):y=stack(x,cond,state=state,commit=False,compute_pending=False,collect_diagnostics=False,current_mask=None,attention_mask=mask)[0]
                outputs.append(y.clone())
            assert torch.equal(*outputs);fallbacks.append(dict(masked_fallback_exact=True))
            # Weight-change invalidation: same accepted physical entry, changed weight,
            # and exact ref/opt result. Restore the original parameter afterward.
            param=next(stack.parameters());original=param.clone();param.add_(.001);hashes=[]
            for name in ['reference','optimized']:
                configure(c,name);restore(runner,boundary);runtime.restore_rng(rng);_,q,d,_,_=runner.step(noises[5]);hashes.append(identity([q,d,state_tree(snapshot(runner)),runtime.rng()]))
            param.copy_(original);assert hashes[0]==hashes[1];fallbacks.append(dict(weight_change_invalidates_exact=True))
            result=dict(checkpoint=str(cfg['parent']),checkpoint_sha256=sha(cfg['parent']),paired=12,equivalent=True,device=torch.cuda.get_device_name(),torch=torch.__version__,timings=timings,checks=fallbacks,kernel_stats=stats(c),scope='same weight/environment alternating warm window generation+decode; snapshot copy,text/network/buffer/SONIC excluded; one process exploratory, not paper multi-process final')
            result['summary']={name:{f'p{p}_ms':float(np.percentile([r['ms'] for r in timings if r['backend']==name],p)) for p in [50,95,99]} for name in ['reference','optimized']};write(dest/'RESULT.json',result);print('PERFORMANCE_PASS',result['summary'],flush=True)
            result.update(model_load_seconds=load_seconds,backend_first_activation=activation,service_latency='NOT_MEASURED; no queue or controller in this harness');write(dest/'RESULT.json',result)
        finally:probe.close()
    return result

def cold_worker(config,output,backend,seed):
    """First generated window in an independent process; no history warmup."""
    from . import runtime,sampling
    from .engine import config as load_config
    from .evaluation import text_for,R
    import random
    cfg=load_config(config);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'RESULT.json').exists():raise RuntimeError('Cold worker must be a fresh process/output, not replay old timing')
    with Budget(cfg['budget_root']).timed('cold_backend:'+backend,closing=True):
        begin=time.perf_counter();m,c,ck=runtime.load(cfg['parent'],False);torch.cuda.synchronize();load=time.perf_counter()-begin
        text=text_for(cfg,read(R/'TEXTS.json')['walk_0']['prompt'],m.device);probe=sampling.NoiseProbe(m)
        try:
          with torch.inference_mode():
            configure(c,backend);random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            runner=sampling.Native(m,c,probe,text);noise=sampling.fresh_noise(seed,1)[0]
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            _,q,dec,_,_=runner.step(noise);torch.cuda.synchronize();first=time.perf_counter()-begin
            state=state_tree(snapshot(runner));owners={id(module):name for name,module in m.named_modules()};checked=[]
            def portable(value,path='state'):
                if isinstance(value,dict):
                    out={}
                    for key,v in value.items():
                        if key=='source_kernel_id':
                            assert v in owners,('Foreign state owner',path,v)
                            out[key]=owners[v];checked.append(dict(path=path,owner=owners[v]))
                        else:out[key]=portable(v,path+'.'+str(key))
                    return out
                if isinstance(value,(list,tuple)):return [portable(v,path+'.'+str(i)) for i,v in enumerate(value)]
                return value
            numerical_state=portable(state);assert checked
            result=dict(backend=backend,seed=seed,checkpoint=cfg['parent'],checkpoint_sha256=sha(cfg['parent']),device=torch.cuda.get_device_name(),torch=torch.__version__,model_load_seconds=load,first_generation_seconds=first,peak_allocated_bytes=torch.cuda.max_memory_allocated(),hashes=dict(qpos=identity(q),motion=identity(dec),state=identity(numerical_state),rng=identity(runtime.rng())),state_process_local_owner_checks=checked,kernel_stats=stats(c),scope='first cold generation+decode+commit after model load in new process; text cached; OS/disk/driver cache not cleared; excludes networking, queue, playback and SONIC',comparison_contract='all RNGs seeded; process-local source_kernel_id validated then compared as stable module name; all numerical state retained')
            write(dest/'RESULT.json',result);print('COLD',backend,round(first,3),flush=True)
        finally:probe.close()

def cold_suite(config,output):
    """Registered missing cold-start measurement, not another quality panel."""
    import subprocess,os
    dest=Path(output);protocol=read(dest/'PROTOCOL.json')
    if (dest/'RESULT.json').exists():return read(dest/'RESULT.json')
    launcher=Path(__file__).resolve().parents[2]/'examples/train_motion.sh';results=[]
    for pair,seed in enumerate(protocol['seeds']):
      for backend in (['reference','optimized'] if pair%2==0 else ['optimized','reference']):
        folder=dest/f'pair{pair}_{backend}';path=folder/'RESULT.json'
        if not path.exists():
            begin=time.perf_counter()
            with (dest/f'pair{pair}_{backend}.log').open('w') as log:
                subprocess.run(['bash',str(launcher),'kernel-cold-worker','--config',config,'--output',str(folder),'--backend',backend,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,MOTION_NPROC='1'))
            write(folder/'PROCESS.json',dict(total_process_seconds=time.perf_counter()-begin,includes='imports,model_load,generation,hashing,serialization; not pure user request latency'))
        results.append(read(path))
    pairs=[]
    for seed in protocol['seeds']:
        a,b=[next(r for r in results if r['seed']==seed and r['backend']==name) for name in ['reference','optimized']]
        checks={k:a['hashes'][k]==b['hashes'][k] for k in a['hashes']};assert all(checks.values()),checks
        pairs.append(dict(seed=seed,checks=checks,reference_ms=1000*a['first_generation_seconds'],optimized_ms=1000*b['first_generation_seconds']))
    result=dict(passed=True,pairs=pairs,processes=len(results),rows=results,distribution_limit='two fresh processes per backend; raw pair timings only, too few cold observations for reliable p95/p99',hot_performance_reused=protocol['hot_results'],service_latency='NOT_MEASURED; no production replacement or valid new-model UI qualification')
    write(dest/'RESULT.json',result);print('COLD_SUITE',pairs,flush=True);return result
