#!/usr/bin/env python3
"""Source-only six-service launcher with checkout-scoped process ownership."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
CONFIGS=ROOT.parent/'configs'
sys.path.insert(0,str(ROOT.parent))
from motion_mamba2.model.configuration import load_environment,read_config
PREFIX='mm2_'+hashlib.sha256(str(ROOT).encode()).hexdigest()[:10]+'_'
REQUIRED=['MM2_CHECKPOINT','MM2_FRONTEND_DIR','MM2_VIEWER_ASSETS','MM2_TEXT_ENCODER_BASE','MM2_TEXT_ENCODER_ADAPTER','MM2_SONIC_ROOT','MM2_PYTHON','MM2_SIM_PYTHON','MM2_TENSORRT_ROOT','MM2_ONNXRUNTIME_ROOT']

def read_env(path):
    result={}
    for n,line in enumerate(Path(path).read_text().splitlines(),1):
        line=line.strip()
        if not line or line.startswith('#'):continue
        if '=' not in line:raise ValueError(f'{path}:{n}: expected KEY=value')
        key,value=line.split('=',1)
        tokens=shlex.split(value,comments=True)
        if len(tokens)>1:raise ValueError(f'{path}:{n}: quote values containing spaces')
        result[key.strip()]=tokens[0] if tokens else ''
    return result

def environment(path):
    values=load_environment(CONFIGS/'paths.env')
    values.update(read_env(CONFIGS/'simulation.env'))
    if Path(path)!=CONFIGS/'simulation.env':values.update(read_env(path))
    values.update({k:v for k,v in os.environ.items() if k in values})
    env=dict(os.environ,**values)
    if not env.get('MM2_SKELETON_PATH') and env.get('MM2_VIEWER_ASSETS'):env['MM2_SKELETON_PATH']=str(Path(env['MM2_VIEWER_ASSETS'])/'skeletons/g1skel34')
    for key in REQUIRED:
        if env.get(key):env[key]=str(Path(env[key]).expanduser())
    env['MM2_RUNTIME_DIR']=values.get('MM2_RUNTIME_DIR') or str(Path.home()/'.cache/motion_mamba2/simulation'/PREFIX)
    env.update(MM2_MODEL_LABEL='MotionMamba2',TEXT_ENCODER_MODE='api',TEXT_ENCODER_URL=f'http://127.0.0.1:{env["MM2_TEXT_PORT"]}/',
        ARDY_GENERATOR_BACKEND='motion_mamba',MM2_GENERATOR_URL=f'http://127.0.0.1:{env["MM2_GENERATOR_PORT"]}',
        ARDY_DEFAULT_MODEL='MotionMamba2',ARDY_VIEWER_PORT=env['MM2_VIEWER_PORT'],
        ARDY_SONIC_ENABLE='1',ARDY_SONIC_WEB_URL=f'ws://127.0.0.1:{env["MM2_WEB_PORT"]}/ws/producer',
        ARDY_SONIC_ZMQ_BIND='tcp://127.0.0.1:5556',ARDY_SONIC_CONTROL_BIND='tcp://127.0.0.1:5559',
        ARDY_SONIC_ZMQ_TOPIC='pose',ARDY_SONIC_FPS='50',ARDY_SONIC_DEBUG_MODE='ardy_streaming',ARDY_SONIC_TIME_STRETCH='1.0',
        SONIC_TELEMETRY_ENDPOINT='tcp://127.0.0.1:5560',SONIC_REFERENCE_TELEMETRY_ENDPOINT='tcp://127.0.0.1:5557',
        SONIC_CONTROL_ENDPOINT='tcp://127.0.0.1:5558',ARDY_BRIDGE_CONTROL_ENDPOINT='tcp://127.0.0.1:5559',
        SONIC_PHYSICS_STEP_MODE='fixed_lowcmd',SONIC_PHYSICS_STEPS_PER_LOWCMD='4',SONIC_SIM_YAML_PD_JOINTS='',
        SONIC_SAFETY_HARNESS_SCALE='0',SONIC_SYNC_TO_PHYSICS='1',SONIC_LOWCMD_HZ='50',
        CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LOCAL_CACHE='true',
        NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
    env['MM2_VIEWER_OUTPUT']=str(Path(env['MM2_RUNTIME_DIR'])/'viewer')
    env['PYTHONPATH']=os.pathsep.join([str(ROOT.parent),str(ROOT/'vendor/viser'),str(ROOT/'inference'),str(ROOT/'viewer'),str(ROOT/'web/backend'),env.get('MM2_SONIC_ROOT',''),str(Path(env.get('MM2_SONIC_ROOT',''))/'external_dependencies/unitree_sdk2_python'),env.get('PYTHONPATH','')])
    env['LD_LIBRARY_PATH']=str(Path(env.get('MM2_TENSORRT_ROOT',''))/'lib')+os.pathsep+str(Path(env.get('MM2_ONNXRUNTIME_ROOT',''))/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    env['MUJOCO_GL']='egl'
    for k in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','DISPLAY','CYCLONEDDS_URI','RMW_IMPLEMENTATION','ROS_DOMAIN_ID'):env.pop(k,None)
    return env

def check(env):
    errors=[]
    for key in REQUIRED:
        value=env.get(key,'')
        if not value:errors.append(f'{key}: required deployment path is empty')
        elif not Path(value).expanduser().is_absolute():errors.append(f'{key}: use an absolute path')
        elif not Path(value).expanduser().exists():errors.append(f'{key}: path does not exist: {value}')
    sonic=Path(env.get('MM2_SONIC_ROOT',''))
    for name in ['gear_sonic_deploy/target/release/g1_deploy_onnx_ref','gear_sonic_deploy/policy/release/model_decoder.onnx','gear_sonic_deploy/policy/release/model_encoder.onnx','gear_sonic_deploy/policy/release/observation_config.yaml','gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx']:
        if env.get('MM2_SONIC_ROOT') and not (sonic/name).is_file():errors.append('Missing SONIC artifact: '+name)
    if not (ROOT/'web/node_modules/three').is_dir():errors.append('Run npm ci --prefix visualization/web (or npm install before the lockfile exists)')
    if errors:raise ValueError('\n'.join(errors))

def prepare(env):
    run=Path(env['MM2_RUNTIME_DIR']);run.mkdir(parents=True,exist_ok=True)
    config=read_config(CONFIGS/'model.yaml')
    spec=dict(format='motion_mamba2_inference',sequence_kernel=config['sequence'],representation=config['representation'],normalization=config['normalization'],execution=config['execution'],skeleton_assets=env.get('MM2_SKELETON_PATH',''))
    spec.update(weights=str(Path(env['MM2_CHECKPOINT']).expanduser().resolve()),frontend_assets=str(Path(env['MM2_FRONTEND_DIR']).expanduser().resolve()))
    target=run/'inference.json';target.write_text(json.dumps(spec,indent=2)+'\n');return target

def plans(env,spec):
    p=env.get('MM2_PYTHON') or sys.executable
    s=env.get('MM2_SIM_PYTHON') or sys.executable
    entry=str(ROOT/'run_component.py');sonic=Path(env.get('MM2_SONIC_ROOT') or '/path/to/SONIC')/'gear_sonic_deploy'
    return [
        ('text',int(env['MM2_TEXT_PORT']),[p,entry,'text','--host','127.0.0.1','--port',env['MM2_TEXT_PORT'],'--device','cuda:0'],ROOT,env['MM2_GPU_TEXT']),
        ('generator',int(env['MM2_GENERATOR_PORT']),[p,entry,'generator','--checkpoint',str(spec),'--output',str(Path(env['MM2_RUNTIME_DIR'])/'generator'),'--port',env['MM2_GENERATOR_PORT'],'--text-url',env['TEXT_ENCODER_URL']],ROOT,env['MM2_GPU_GENERATOR']),
        ('web',int(env['MM2_WEB_PORT']),[p,entry,'web'],ROOT,None),
        ('physics',5560,[s,entry,'physics','--debug-mode','ardy_streaming','--telemetry-port','5560','--control-port','5558'],Path(env.get('MM2_SONIC_ROOT') or '/path/to/SONIC'),env['MM2_GPU_SIM']),
        ('policy',5557,[str(sonic/'target/release/g1_deploy_onnx_ref'),'lo','policy/release/model_decoder.onnx','reference/example/','--obs-config','policy/release/observation_config.yaml','--encoder-file','policy/release/model_encoder.onnx','--planner-file','planner/target_vel/V2/planner_sonic.onnx','--input-type','zmq_manager','--output-type','zmq','--zmq-host','127.0.0.1','--zmq-port','5556','--zmq-topic','pose','--zmq-conflate','--zmq-out-port','5557','--zmq-out-topic','g1_debug','--disable-crc-check'],sonic,env['MM2_GPU_SIM']),
        ('viewer',int(env['MM2_VIEWER_PORT']),[p,entry,'viewer'],ROOT,env['MM2_GPU_VIEWER'])]

def tmux(*args,check=True):return subprocess.run(['tmux',*args],check=check,capture_output=True,text=True)
def alive(name):return tmux('has-session','-t','='+PREFIX+name,check=False).returncode==0

def port_open(port):
    with socket.socket() as s:s.settimeout(.5);return s.connect_ex(('127.0.0.1',port))==0

def wait(name,port):
    end=time.monotonic()+1800
    while time.monotonic()<end:
        if not alive(name):raise RuntimeError(f'{name} exited; inspect its runtime log')
        if port_open(port):return
        time.sleep(1)
    raise TimeoutError(f'{name} did not become ready')

def start(env):
    check(env)
    queue=plans(env,Path(env['MM2_RUNTIME_DIR'])/'inference.json')
    if any(alive(name) for name,*_ in queue):raise RuntimeError('This checkout already has services. Use status or stop before starting.')
    for name,port,*_ in queue:
        if port_open(port):raise RuntimeError(f'Port {port} is occupied. Existing services were preserved.')
    spec=prepare(env)
    logs=Path(env['MM2_RUNTIME_DIR'])/'logs';logs.mkdir(parents=True,exist_ok=True)
    for name,port,command,cwd,gpu in queue:
        child=dict(env)
        if gpu is not None:child['CUDA_VISIBLE_DEVICES']=gpu
        else:child['CUDA_VISIBLE_DEVICES']=''
        if name=='physics':
            for k in ('CYCLONEDDS_URI','RMW_IMPLEMENTATION','ROS_DOMAIN_ID'):child.pop(k,None)
        # Do not copy credentials or the caller's full environment into tmux.
        allowed={k:v for k,v in child.items() if k.startswith(('MM2_','ARDY_','SONIC_','TEXT_ENCODER','HF_HUB_','TRANSFORMERS_','CUDA_')) or k in {'PATH','HOME','LD_LIBRARY_PATH','PYTHONPATH','LOCAL_CACHE','MUJOCO_GL','NO_PROXY','no_proxy','HF_HOME','TMPDIR'}}
        shell=shlex.join(['env','-u','HF_TOKEN','-u','HUGGING_FACE_HUB_TOKEN',*[f'{k}={v}' for k,v in allowed.items()],*command])+' > '+shlex.quote(str(logs/(name+'.log')))+' 2>&1'
        tmux('new-session','-d','-s',PREFIX+name,'-c',str(cwd),shell)
        print(f'Starting {name}: {logs/(name+".log")}',flush=True);wait(name,port)
        if name=='generator':
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=30) as response:health=json.load(response)
            expected=hashlib.file_digest(open(env['MM2_CHECKPOINT'],'rb'),'sha256').hexdigest()
            if health.get('checkpoint_sha256')!=expected:raise RuntimeError('Generator checkpoint identity does not match deployment configuration')
    print(f'Viser http://127.0.0.1:{env["MM2_VIEWER_PORT"]}\nSONIC http://127.0.0.1:{env["MM2_WEB_PORT"]}')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check','plan','start','status','stop']);p.add_argument('--env',type=Path,default=CONFIGS/'simulation.env');a=p.parse_args()
    if a.action in ('status','stop'):
        for name in ['viewer','policy','physics','web','generator','text']:
            running=alive(name)
            if a.action=='stop' and running:tmux('kill-session','-t','='+PREFIX+name)
            print(name,('stopped' if a.action=='stop' else 'running') if running else 'not running')
        return
    env=environment(a.env if a.env.exists() else CONFIGS/'simulation.env')
    if a.action=='check':check(env);print('Deployment paths and required files exist; no service started.')
    elif a.action=='plan':
        for name,port,command,cwd,gpu in plans(env,Path(env['MM2_RUNTIME_DIR'])/'inference.json'):print(json.dumps(dict(service=name,port=port,command=command,cwd=str(cwd),gpu=gpu)))
    else:start(env)
if __name__=='__main__':main()
