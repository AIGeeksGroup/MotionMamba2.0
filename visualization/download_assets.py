#!/usr/bin/env python3
"""Download deployment artifacts outside the source checkout; never an environment."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--sonic-root',type=Path);a=p.parse_args()
    from huggingface_hub import snapshot_download
    from huggingface_hub import hf_hub_download
    root=a.root.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    from motion_mamba2.model.configuration import read_config
    config=read_config('configs/downloads.yaml')
    snapshots=[(x['repository'],x['revision'],x['directory'],x['patterns']) for x in config['snapshots']]
    for repo,revision,dest,patterns in snapshots:
        snapshot_download(repo,revision=revision,local_dir=root/dest,allow_patterns=patterns,ignore_patterns=['*.engine','*.trt','original/*'])
    # LLM2Vec's base config refers to the gated Llama model; cache its pinned snapshot.
    llama_revision=config['llama_revision']
    llama=Path(snapshot_download(config['llama_repository'],revision=llama_revision,ignore_patterns=['original/*']))
    # PEFT resolves the base model by its repository name in offline mode.
    refs=llama.parent.parent/'refs';refs.mkdir(exist_ok=True)
    (refs/'main').write_text(llama_revision)
    prefix=config['skeleton_prefix'];source_rev=config['skeleton_revision'];repository=config['skeleton_repository']
    url=f'https://api.github.com/repos/{repository}/git/trees/{source_rev}?recursive=1'
    request=urllib.request.Request(url,headers={'User-Agent':'MotionMamba2-asset-downloader'})
    with urllib.request.urlopen(request,timeout=60) as r:tree=json.load(r)['tree']
    entries=[x for x in tree if x['type']=='blob' and x['path'].startswith(prefix+'skeletons/g1skel34/')]
    if not entries:raise ValueError('Pinned G1 skeleton source is unavailable')
    def fetch(entry):
        relative=entry['path'][len(prefix):];target=root/'viewer_assets'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/{repository}/{source_rev}/{entry["path"]}',timeout=120) as r:data=r.read()
        if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=entry['sha']:raise ValueError('Git blob mismatch: '+relative)
        target.write_bytes(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,entries))
    if a.sonic_root:
        deploy=a.sonic_root.expanduser().resolve()/'gear_sonic_deploy'
        import shutil
        paths={'model_encoder.onnx':'policy/release/model_encoder.onnx','model_decoder.onnx':'policy/release/model_decoder.onnx','observation_config.yaml':'policy/release/observation_config.yaml','planner_sonic.onnx':'planner/target_vel/V2/planner_sonic.onnx'}
        for filename,relative in paths.items():
            source=hf_hub_download(config['sonic_repository'],filename,revision=config['sonic_revision'])
            target=deploy/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    print('Downloads ready. Fill configs/paths.env with these absolute paths:',root)
if __name__=='__main__':main()
