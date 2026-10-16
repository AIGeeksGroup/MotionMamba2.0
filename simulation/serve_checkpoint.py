"""Generic Native/Minimal service with the existing official viewer transport and controls."""
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

SIM=Path(__file__).resolve().parent
sys.path.insert(0,str(SIM))
from checkpoint_profile import resolve,file_sha
from paths import activate, LIVE
path=sys.argv[sys.argv.index('--checkpoint')+1]
profile=resolve(path)
assert profile['kind'] in ('native','minimal')
SOURCE=Path(profile['source']);OUT=Path(profile['output'])
activate()
from model_loader import load_selected
from motion_training import acceleration
import main3_runtime
import g1_online_runtime

for name in ('g1_online_controls','g1_control_intents','g1_online_runtime'):
    p=LIVE/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,p)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
g1_online_runtime=sys.modules['g1_online_runtime']

def frozen_load(checkpoint,mode,stats_dir,device='cuda:0'):
    m,c,ck,actual=load_selected(checkpoint,device)
    assert actual['checkpoint_sha256']==profile['checkpoint_sha256']
    requested=os.environ.get('MM2_INFERENCE_BACKEND','optimized')
    assert requested in ('auto','reference','optimized')
    backend='reference'
    gatefile=None
    if requested!='reference':
        from inference_acceleration import enable_sequence
        gatefile=enable_sequence(m,c,actual)
        backend='optimized'
    else:
        from inference_acceleration import configure
        configure(c,False)
    m.online_inference_backend=backend
    m.online_control_contract='g1_bounded_future_smooth_control_v1'
    m.online_masked_graphs=backend=='optimized'
    m.simulation_architecture=actual['kind']
    record=json.loads((OUT/'MODEL_LOAD.json').read_text())
    record.update(inference_backend=backend,acceleration_gate=gatefile,
        control_runtime=str(LIVE/'g1_online_runtime.py'),
        control_source_sha256={str(p):file_sha(p) for p in [
            *(LIVE/f'{n}.py' for n in ('g1_online_controls','g1_control_intents','g1_online_runtime')),
            LIVE/'g1_live_replanning.py']})
    (OUT/'MODEL_LOAD.json').write_text(json.dumps(record,indent=2)+'\n')
    print('SELECTED_CHECKPOINT_LOADED',actual['label'],actual['checkpoint_sha256'],flush=True)
    return m,c

main3_runtime.load_generator=frozen_load
sys.path.append(str(LIVE))
g1_online_runtime.OriginalOnlineG1=g1_online_runtime.OnlineG1
def runtime_factory(*args,**kwargs):
    from g1_live_replanning import ReplanningOnlineG1
    return ReplanningOnlineG1(*args,**kwargs)
g1_online_runtime.OnlineG1=runtime_factory
runpy.run_path(str(SIM/'serve_main.py'),run_name='__main__')
