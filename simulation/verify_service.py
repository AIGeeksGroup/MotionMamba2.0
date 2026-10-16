"""Exercise warmed RPC without publishing test motion to the simulator."""
import json
import argparse
from pathlib import Path
import time
import urllib.request

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8766')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'runtime/SERVICE_ACCEPTANCE.json')
args=parser.parse_args()

opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def call(path,body=None):
    request=urllib.request.Request(args.url+path,
        data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    tick=time.monotonic()
    with opener.open(request,timeout=60) as response:r=json.load(response)
    return r,time.monotonic()-tick

health,_=call('/health');assert health['ready'] and health['inference_backend']=='optimized'
assert health['masked_future_graphs']
s,_=call('/session',dict(seed=2026101215));sid=s['session'];rows=[]
try:
    for i in range(6):
        r,seconds=call('/next',dict(session=sid,window=i,revision=0,
            prompt='A person continuously walks straight ahead with a natural gait',request_version=i))
        assert len(r['qpos'])==52 and r['accepted_commits']==i+1
        assert sum(r['inference_graph_builds'].values())==0
        rows.append(dict(kind='next',rpc_seconds=seconds,generation_seconds=r['generation_seconds']))
        call('/publish',dict(session=sid,boundary=i+1,revision=0))
    for frame in (22,74,75,180,247):
        r,seconds=call('/replan/prepare',dict(session=sid,source_frame=264,revision=0,
            prompt='A person walks backwards.',request_version=20+frame,
            control={'root2d':[{'frame':frame,'xz':[.8,1.5]}]}))
        assert len(r['qpos'])==52 and r['frozen_prefix_frames']==4
        assert sum(r['inference_graph_builds'].values())==0
        rows.append(dict(kind='replan',frame=frame,rpc_seconds=seconds,generation_seconds=r['generation_seconds']))
        call('/replan/cancel',dict(session=sid))
    out=Path(health['checkpoint']).name
    record=dict(passed=True,health=health,timings=rows,
                scope='Loopback generation and text/future replan; test motion not published to physics')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print('WARMED_SERVICE_PASS',json.dumps(rows),flush=True)
finally:call('/close',dict(session=sid))
