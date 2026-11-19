"""Evaluate Motion Mamba 2.0 on BONES, HumanML3D and HiPHI geometric controls."""
import argparse
import importlib
import json
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmarks.metrics import measure
from benchmarks.results import file_hash,write_results
from motion_mamba2.model.configuration import ROOT,load_environment,read_config

def assets_for(domain,config,folder):
    spec=config['domains'][domain];folder=Path(folder)
    frozen=json.loads((ROOT/'configs/benchmark_assets.json').read_text())
    for key in ['inputs','cases','text_features','text_index']:
        path=folder/spec[key]
        if file_hash(path)!=frozen[spec[key]]:raise ValueError(f'Benchmark asset changed: {path}')
    data=dict(np.load(folder/spec['inputs'],allow_pickle=False))
    cases=[json.loads(line) for line in (folder/spec['cases']).read_text().splitlines()]
    names=json.loads((folder/spec['text_index']).read_text())['prompts']
    features=dict(zip(names,np.load(folder/spec['text_features'],allow_pickle=False)))
    return spec,data,cases,features

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=['all','bones','humanml3d','hiphi'],default='all')
    p.add_argument('--paths',default=str(ROOT/'configs/paths.env'))
    p.add_argument('--task',help='For example frames_52, frames_208, frames_468 or frames_988')
    p.add_argument('--check',action='store_true',help='Validate frozen inputs without loading weights or using a GPU')
    p.add_argument('--score-only',action='store_true',help='Score existing geometry files without loading the model')
    p.add_argument('--limit',type=int,help='Development check only; limited results are marked incomplete')
    a=p.parse_args(argv)
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    values=load_environment(a.paths);settings=read_config('configs/benchmarks.yaml');inference=read_config('configs/inference.yaml')
    domains=list(settings['domains']) if a.domain=='all' else [a.domain]
    folder=Path(values.get('MM2_BENCHMARK_DATA_PATH') or ROOT/'benchmarks'/'benchmark_assets')
    if not a.check and not values.get('MM2_OUTPUT_PATH'):p.error('Fill MM2_OUTPUT_PATH in configs/paths.env')
    output=Path(values.get('MM2_OUTPUT_PATH') or '.')
    model=client=None
    for domain in domains:
        spec,data,cases,features=assets_for(domain,settings,folder)
        tasks=[a.task] if a.task else list(spec['tasks'])
        for task in tasks:
            if task not in spec['tasks']:p.error(f'{domain} does not define {task}')
        if a.check:
            print(json.dumps(dict(domain=domain,tasks=tasks,test_families=sum(c['split']=='locked_test' for c in cases),inputs_verified=True)))
            continue
        if not a.score_only and model is None:
            from motion_mamba2.model.checkpoint import load_model
            from motion_mamba2.model.inference_runtime import configure_acceleration
            model,client,_=load_model(paths=a.paths,device=inference['device'])
            configure_acceleration(client,inference)
        schedule=importlib.import_module('benchmarks.'+domain).schedule
        for task in tasks:
            dest=output/domain/task;dest.mkdir(parents=True,exist_ok=True)
            selected=[(i,c) for i,c in enumerate(cases) if c['split']=='locked_test']
            if a.limit:selected=selected[:a.limit]
            rows=[]
            identity=dict(checkpoint_sha256=file_hash(values['MM2_CHECKPOINT']) if values.get('MM2_CHECKPOINT') else None,
                task=task,domain=domain,steps=inference['denoising_steps'],cfg=[inference['cfg_text'],inference['cfg_constraint']],
                input_assets_sha256=file_hash(folder/spec['inputs']),model_config_sha256=file_hash(ROOT/'configs/model.yaml'),
                limited=a.limit is not None,selected_cases=[c['case_id'] for _,c in selected],
                inference_config_sha256=file_hash(ROOT/'configs/inference.yaml'),
                protocol_config_sha256=file_hash(ROOT/'configs/benchmarks.yaml'),metrics_code_sha256=file_hash(ROOT/'benchmarks/metrics.py'))
            binding=dest/'inputs.json'
            if binding.exists():
                previous=json.loads(binding.read_text())
                if a.score_only:
                    if any(previous.get(k)!=identity[k] for k in ['input_assets_sha256','selected_cases','model_config_sha256','protocol_config_sha256','inference_config_sha256','metrics_code_sha256']):raise ValueError('Saved inputs or protocol differ')
                    identity=previous
                elif previous!=identity:raise ValueError('Output directory belongs to a different model/configuration')
            elif a.score_only:raise FileNotFoundError(binding)
            else:binding.write_text(json.dumps(identity,indent=2)+'\n')
            for index,case in selected:
                commands,texts=schedule(spec,task,case)
                for seed in case['noise_seeds']:
                    path=dest/f'{case["case_id"]}_seed_{seed}.npz'
                    event=path.with_suffix('.json')
                    if not a.score_only and not path.exists():
                        from benchmarks.rollout import generate
                        geometry,events=generate(model,client,data['initial_physical'][index],data['goal_coordinate_indices'][index],
                            data['goal_physical_values'][index],texts,features,seed,steps=inference['denoising_steps'],
                            cfg=(inference['cfg_text'],inference['cfg_constraint']))
                        temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**geometry);temp.replace(path)
                        event.write_text(json.dumps(dict(case_id=case['case_id'],seed=seed,geometry_sha256=file_hash(path),events=events),indent=2)+'\n')
                    saved=json.loads(event.read_text())
                    if saved['geometry_sha256']!=file_hash(path) or saved['seed']!=seed or saved['case_id']!=case['case_id']:raise ValueError(f'Invalid trial binding: {path}')
                    if len(saved['events'])!=len(commands) or any(e['prompt']!=texts[i] or e['visible_scalars']!=8 for i,e in enumerate(saved['events'])):
                        raise ValueError('Saved conditions differ')
                    with np.load(path,allow_pickle=False) as z:
                        metrics=measure(z['root'],z['joints'],case,commands,data['initial_physical'][index,-1,:3],
                            settings['window_frames'],settings['fps'],settings['command_speed_threshold'])
                    rows.append(dict(case_id=case['case_id'],source_family=case['source_family'],seed=seed,**metrics))
            result=write_results(dest,rows,settings);result['complete_protocol']=not identity['limited']
            (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(domain,task,json.dumps(result['metrics']),flush=True)
if __name__=='__main__':main()
