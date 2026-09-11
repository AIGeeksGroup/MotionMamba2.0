"""Convert accepted-window tensor traces into a compact cross-run donor bank."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--trace',required=True); p.add_argument('--panel',required=True); p.add_argument('--output',required=True); p.add_argument('--manifest',required=True)
    a=p.parse_args(); trace_path=Path(a.trace); rows=torch.load(trace_path,map_location='cpu')
    panel=[json.loads(x) for x in Path(a.panel).read_text().splitlines() if x.strip()]
    allowed={str(x['id']) for x in panel}; bank={}; counts={}
    for rec in rows:
        ids=[str(x) for x in rec.get('sample_ids',[])]
        if rec.get('event') not in {'initial_history','generated_window'}: continue
        window=int(rec['completed_windows']); block=str(rec['block']); layer=int(rec['layer']); slots=rec['slots']
        if slots.ndim != 3 or slots.shape[0] != len(ids): raise ValueError('trace slot shape mismatch')
        for i,sample_id in enumerate(ids):
            if sample_id not in allowed: continue
            bank[(sample_id,window,block,layer)] = slots[i].detach().float().contiguous()
            counts[sample_id]=counts.get(sample_id,0)+1
    if not bank: raise RuntimeError('trace produced an empty donor bank')
    windows=sorted({k[1] for k in bank}); blocks=sorted({k[2] for k in bank}); layers=sorted({k[3] for k in bank})
    expected=len(allowed)*len(windows)*len(blocks)*len(layers)
    if len(bank) != expected: raise RuntimeError(f'incomplete bank: {len(bank)} keys, expected {expected}')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); torch.save(bank,out)
    manifest={'schema_version':'d3b_donor_memory_bank_v1','trace':str(trace_path.resolve()),'trace_sha256':sha(trace_path),'panel':str(Path(a.panel).resolve()),'panel_sha256':sha(Path(a.panel)),'output':str(out.resolve()),'output_sha256':sha(out),'sample_count':len(allowed),'sample_ids':sorted(allowed),'window_indices':windows,'blocks':blocks,'layers':layers,'slots_shape':list(next(iter(bank.values())).shape),'key_count':len(bank),'per_sample_key_counts':counts,'alignment':'donor committed memory at identical completed_window index; initial_history=0; post-W4 reads use indices 4..max'}
    Path(a.manifest).write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
