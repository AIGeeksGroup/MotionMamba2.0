"""Resolve supported G1 checkpoints before any simulation services are launched."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path

from paths import SIM, SOURCE, STATS, OUTPUT
SOURCES = {
    'native': SOURCE,
    'minimal': SOURCE,
    'ardy': SOURCE,
}

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for chunk in iter(lambda: fp.read(16*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def resolve(value):
    import torch
    path = Path(value).expanduser().resolve(strict=True)
    if path.is_dir():
        if not (path/'config.yaml').is_file():
            raise ValueError('Official ARDY model directory must contain config.yaml')
        weights = sorted([*path.glob('*.safetensors'), *path.glob('*.ckpt'), *path.glob('*.pt')])
        if not weights:
            raise ValueError('Official ARDY directory has no model weights')
        digest = hashlib.sha256()
        for asset in [path/'config.yaml',*weights]:
            digest.update(asset.name.encode())
            digest.update(file_sha(asset).encode())
        fingerprint = digest.hexdigest()
        kind, step, source = 'official_ardy', None, SOURCES['ardy']
        fmt = 'released_directory';content_memory=False
    else:
        ck = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        cfg = ck.get('config',{})
        if cfg.get('model',ck.get('model','g1')) != 'g1':
            raise ValueError('This simulator supports G1 checkpoints only')
        mode = cfg.get('mode',ck.get('mode'))
        kind = {'hybrid':'native','minimal':'minimal','ardy':'ardy','original':'ardy'}.get(mode)
        if kind is None or not all(k in ck for k in ('root_model','body_model')):
            raise ValueError('Unknown checkpoint architecture or incomplete model weights')
        if not cfg.get('stats_dir') or not STATS.is_dir():
            raise ValueError('Checkpoint normalization directory is missing')
        step = int(ck.get('step',0));fingerprint = file_sha(path);source=SOURCES[kind]
        content_memory=bool(cfg.get('content_memory_enabled',False))
        imports = ck.get('framework',{}).get('source_imports',{})
        # Historical source_imports are provenance, never executable paths.
        if not (source/'motion_training/_runtime').is_dir():
            raise ValueError('Matching inference source directory is missing')
        fmt = 'framework_file'
        if kind == 'ardy' and not any('seqTransEncoder' in k for k in ck['root_model']):
            raise ValueError('ARDY metadata disagrees with saved Transformer weights')
        if kind != 'ardy' and not any('sequence_stack' in k for k in ck['root_model']):
            raise ValueError('Sequence-model metadata disagrees with saved weights')
    label = {'native':'Native','minimal':'Minimal','ardy':'ARDY','official_ardy':'ARDY official'}[kind]
    if step is not None:label += f' {step//1000}k' if step%1000==0 else f' step {step}'
    output = OUTPUT/f'{kind}_{step or "official"}_{fingerprint[:12]}'
    stage = SIM
    # Fine-tuning saves a global step but evaluates by local stage step.
    # Admit evidence only by the actual checkpoint hash, never a directory name.
    candidates = [stage/'evaluation'/str(step)/'kernel/GATE.json']
    candidates.extend(sorted((stage/'evaluation').glob('*/kernel/GATE.json')))
    gate = None
    for candidate in candidates:
        if candidate.is_file():
            try:
                evidence = json.loads(candidate.read_text())
            except (OSError, ValueError):
                continue
            if evidence.get('passed') and evidence.get('checkpoint_sha256') == fingerprint:
                gate = candidate
                break
    return dict(checkpoint=str(path),checkpoint_sha256=fingerprint,kind=kind,step=step,
        format=fmt,source=str(source),stage_root=str(stage),label=label,output=str(output),
        backend='ardy' if kind in ('ardy','official_ardy') else 'motion_mamba',
        content_memory_enabled=content_memory,
        evaluation_gate=str(gate) if gate else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--shell',action='store_true')
    a=p.parse_args();profile=resolve(a.checkpoint)
    out=Path(profile['output']);out.mkdir(parents=True,exist_ok=True)
    (out/'PROFILE.json').write_text(json.dumps(profile,indent=2)+'\n')
    if a.shell:
        values=dict(MM2_CHECKPOINT=profile['checkpoint'],MM2_MODEL_LABEL=profile['label'],
            MM2_OUT=profile['output'],MM2_SERVER_SCRIPT=str(SIM/'serve_checkpoint.py'),
            MM2_PROFILE=str(out/'PROFILE.json'),MM2_RESOLVED_GENERATOR=profile['backend'])
        for key,value in values.items():print(f'export {key}={shlex.quote(value)}')
    else:print(json.dumps(profile,indent=2))
if __name__=='__main__':main()
