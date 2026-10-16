"""Loopback-only model service, isolated from the simulation's ARDY package."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from paths import activate, warm_embedding
activate()
from main3_runtime import load_generator
from g1_online_runtime import OnlineG1
from ardy.model.text_encoder_api import TextEncoderAPI


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,default=8766)
    p.add_argument('--text-url',default='http://127.0.0.1:9550/')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--mamba-forward-launch', choices=['mamba3_siso_angle_w4s1_v1', 'legacy_autotune'],
                   default='mamba3_siso_angle_w4s1_v1')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    model,client=load_generator(args.checkpoint,'hybrid',None)
    if args.mamba_forward_launch != 'legacy_autotune':
        from g1_inference_launch import install_forward_launch_contract
        install_forward_launch_contract(args.mamba_forward_launch)
    stats_dir=Path(model.motion_rep.stats.folder)
    stats_hash=hashlib.sha256()
    for name in ('mean.npy','std.npy'):
        stats_hash.update(name.encode());stats_hash.update((stats_dir/name).read_bytes())
    metadata=dict(stats_dir=str(stats_dir),stats_sha256=stats_hash.hexdigest(),
                  motion_contract=getattr(model,'motion_contract','legacy'),
                  generation_contract=getattr(model,'generation_contract','legacy'),
                  padding_contract=getattr(model.denoiser.model,'g1_padding_contract','legacy'),
                  mamba_forward_launch=args.mamba_forward_launch,
                  model_label=getattr(model,'simulation_model_label',Path(args.checkpoint).parent.name),
                  architecture=getattr(model,'simulation_architecture','native'),
                  decoder_history_frames=getattr(model,'decoder_history_frames',0),
                  protocol='g1_interaction_v2', control_schema='world_ardy_xz_v1',
                  continuous_sessions=True, max_control_points=52)
    metadata['inference_backend'] = getattr(model, 'online_inference_backend', 'reference')
    metadata['control_contract'] = getattr(model, 'online_control_contract', 'legacy_window_controls')
    metadata['masked_future_graphs'] = getattr(model,'online_masked_graphs',False)
    if metadata['control_contract'] == 'g1_bounded_future_smooth_control_v1':
        metadata.update(max_control_points=248,max_future_condition_frames=196,
                        heading_convention='atan2(vx,vz); zero=+Z',keyboard_transition_frames=52)
    checkpoint_hash=hashlib.sha256()
    with Path(args.checkpoint).open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            checkpoint_hash.update(block)
    metadata.update(checkpoint_sha256=checkpoint_hash.hexdigest(),
                    training_codec_contract=getattr(model,'training_codec_contract','unknown'),
                    training_manifest=getattr(model,'checkpoint_training_manifest',None),
                    mode=('minimal' if getattr(model,'simulation_architecture','native')=='minimal' else 'hybrid'), kv_capacity_chunks=4, diffusion_steps=10, cfg=[2.,2.])
    runtime=OnlineG1(model,client)
    if metadata['inference_backend'] == 'optimized':
        from g1_inference_graph_warmup import configure_graph_capacity
        configure_graph_capacity(client)
        metadata['inference_graph_capacity_per_branch'] = 16
    metadata['live_replanning'] = hasattr(runtime, 'prepare_replan')
    from validate_g1_online_runtime import validate
    validate(runtime,args.output/'runtime_validation.json')
    if metadata['live_replanning']:
        # Warm the replacement path itself, including its guard window and
        # archive restoration, without publishing anything to the simulator.
        import numpy as np
        warm_text=warm_embedding(model.device)
        warm_mask=torch.ones(1,1,dtype=torch.bool,device=model.device)
        warm_session=runtime.new_session(202610083)
        for i in range(2):
            runtime.step(warm_session,warm_text,warm_mask,'warm',i,0)
            runtime.publish(warm_session,i+1,0)
        warm_start=time.perf_counter()
        runtime.prepare_replan(warm_session,warm_text,warm_mask,'warm',44,0,{},1,1)
        metadata['live_replan_prepare_s']=time.perf_counter()-warm_start
        metadata['replanning_contract']='g1_fixed_prefix_4_frame_transaction_v1'
    # Startup validation warms graph shapes through the FIFO boundary before
    # accepting requests; graph compilation must not stall the first Play.
    if metadata['inference_backend'] == 'optimized':
        from g1_inference_graph_warmup import warm_ordinary_session
        if not metadata['live_replanning']:
            import numpy as np
            warm_text=warm_embedding(model.device)
            warm_mask=torch.ones(1,1,dtype=torch.bool,device=model.device)
        warm_ordinary_session(runtime, warm_text, warm_mask)
        if metadata['masked_future_graphs']:
            from g1_masked_inference import warm_future_graphs
            metadata['future_graph_warmup'] = warm_future_graphs(runtime,warm_text,warm_mask)
            metadata['inference_graph_capacity_per_branch'] = 256
        from motion_training.acceleration import stats
        metadata['inference_graph_warmup'] = stats(client)
    encoder=TextEncoderAPI(args.text_url).to(device='cuda:0')
    sessions={}; cache=OrderedDict(); lock=threading.Lock()

    def process_start(pid):
        try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        except (OSError,IndexError):return None

    def reap_orphan_sessions():
        for sid,s in list(sessions.items()):
            pid=getattr(s,'owner_pid',0)
            if pid and process_start(pid)!=s.owner_start:
                if hasattr(s,'archive'):shutil.rmtree(s.archive,ignore_errors=True)
                del sessions[sid]

    def trim_archive(s):
        if not hasattr(s,'archive'):return
        # Planning is bounded to 1500 future frames (<29 windows). Retain 64
        # accepted windows, covering that future plus the active consumed one.
        minimum=max(0,s.published-64)
        for mapping in (s.boundaries,s.records):
            for boundary in list(mapping):
                if boundary<minimum:del mapping[boundary]
        keep=set(s.boundaries.values())|set(s.records.values())
        if s.pending_replan:
            trial=s.pending_replan[2]
            keep.update(trial.boundaries.values());keep.update(trial.records.values())
        for path in s.archive.glob('*.pt'):
            if path not in keep:path.unlink(missing_ok=True)

    def encode(prompt):
        started=time.perf_counter()
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>2000:
            raise ValueError('prompt must contain 1–2000 characters')
        hit=prompt in cache
        if not hit:
            feature,lengths=encoder([prompt])
            mask=torch.arange(feature.shape[1],device=feature.device)[None,:]<torch.tensor(lengths,device=feature.device)[:,None]
            cache[prompt]=(feature,mask)
            if len(cache)>64:cache.popitem(last=False)
        cache.move_to_end(prompt)
        return *cache[prompt],time.perf_counter()-started,hit

    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,value):
            data=json.dumps(value).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            if self.path=='/health':
                self.reply(200,dict(ready=True,checkpoint=args.checkpoint,frames_per_window=52,fps=25,
                                    session_count=len(sessions),max_sessions=4,text_url=args.text_url,
                                    threads=torch.get_num_threads(),**metadata))
            else:self.reply(404,dict(error='not found'))

        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<65536:raise ValueError('request exceeds 64KiB sparse-control limit')
                body=json.loads(self.rfile.read(size))
                requested_at=time.time()
                queued_at=time.perf_counter()
                with lock:
                    queue_seconds=time.perf_counter()-queued_at
                    if self.path=='/session':
                        reap_orphan_sessions()
                        if len(sessions)>=4:raise ValueError('session limit reached; close an unused session')
                        sid=uuid.uuid4().hex
                        s=runtime.new_session(int(body.get('seed',20260917)),int(body.get('windows',0)),
                            body.get('translation'),body.get('heading',0.0))
                        sessions[sid]=s
                        s.owner_pid=int(body.get('owner_pid',0))
                        s.owner_start=process_start(s.owner_pid) if s.owner_pid else None
                        result=dict(session=sid,revision=0,windows=0)
                    elif self.path=='/encode':
                        text,mask,elapsed,hit=encode(body['prompt'])
                        result=dict(text_seconds=elapsed,cache_hit=hit,shape=list(text.shape))
                    else:
                        sid=body['session'];s=sessions[sid]
                        if self.path=='/next':
                            text,mask,elapsed,hit=encode(body['prompt'])
                            graphs_before = stats(client) if metadata['inference_backend'] == 'optimized' else {}
                            build_before = {k:len(v['build_events']) for k,v in graphs_before.items()}
                            result=dict(runtime.step(s,text,mask,body['prompt'],int(body['window']),int(body['revision']),
                                body.get('control'),body.get('request_version',0)),text_seconds=elapsed,text_cache_hit=hit,
                                requested_at=requested_at,queue_seconds=queue_seconds)
                            if metadata['inference_backend'] == 'optimized':
                                after=stats(client)
                                result['inference_graph_builds'] = {k:len(v['build_events'])-build_before[k] for k,v in after.items()}
                                result['inference_graph_build_seconds'] = sum(e['prepare_seconds'] for k,v in after.items() for e in v['build_events'][build_before[k]:])
                            if not body.get('include_motion', False):
                                result.pop('normalized_motion', None)
                            d=args.output/sid;d.mkdir(exist_ok=True)
                            (d/f"r{s.revision}_w{result['window']:05d}.json").write_text(json.dumps(result)+'\n')
                        elif self.path=='/publish':
                            runtime.publish(s,int(body['boundary']),int(body['revision']))
                            trim_archive(s)
                            result=dict(windows=s.windows,published=s.published,revision=s.revision)
                        elif self.path=='/rewind':
                            result=runtime.rewind(s,int(body['boundary']),int(body['revision']))
                        elif self.path=='/replan/prepare':
                            text,mask,elapsed,hit=encode(body['prompt'])
                            graphs_before=stats(client) if metadata['inference_backend']=='optimized' else {}
                            counts_before={k:dict(builds=len(v['build_events']),replays=v['replays'],fallbacks=v['fallbacks']) for k,v in graphs_before.items()}
                            prepare_started=time.perf_counter()
                            result=runtime.prepare_replan(s,text,mask,body['prompt'],
                                int(body['source_frame']),int(body['revision']),
                                body.get('control'),body.get('request_version',0),
                                int(body.get('lookahead_windows',0)),body.get('lookahead_control'))
                            result.update(text_seconds=elapsed,text_cache_hit=hit,queue_seconds=queue_seconds)
                            result['prepare_seconds']=time.perf_counter()-prepare_started
                            if metadata['inference_backend']=='optimized':
                                after=stats(client)
                                result['inference_graph_builds']={k:len(v['build_events'])-counts_before[k]['builds'] for k,v in after.items()}
                                result['inference_graph_replays']={k:v['replays']-counts_before[k]['replays'] for k,v in after.items()}
                                result['inference_graph_build_seconds']=sum(e['prepare_seconds'] for k,v in after.items() for e in v['build_events'][counts_before[k]['builds']:])
                            d=args.output/sid;d.mkdir(exist_ok=True)
                            receipt={k:result[k] for k in ('replacement_source_frame','condition_shape','future_condition_nonzero',
                                'generation_seconds','prepare_seconds','text_seconds','queue_seconds',
                                'inference_graph_builds','inference_graph_replays','inference_graph_build_seconds') if k in result}
                            receipt.update(request_version=body.get('request_version'),requested_at=requested_at)
                            with (d/'replan_timings.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
                        elif self.path=='/replan/accept':
                            result=runtime.accept_replan(s,body['token'],int(body['revision']))
                        elif self.path=='/replan/cancel':
                            result=runtime.cancel_replan(s)
                            trim_archive(s)
                        elif self.path=='/close':
                            if hasattr(s,'archive'):shutil.rmtree(s.archive,ignore_errors=True)
                            del sessions[sid];result=dict(closed=True)
                        else:raise ValueError('unknown endpoint')
                self.reply(200,result)
            except (ValueError,KeyError) as exc:self.reply(409,dict(error=str(exc)))
            except Exception as exc:
                import traceback;traceback.print_exc()
                self.reply(500,dict(error=f'{type(exc).__name__}: {exc}'))

    print(f'MODEL_SERVICE_READY http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
