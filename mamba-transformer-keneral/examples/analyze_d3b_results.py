"""Analyze D3b paired transplant outputs and donor-bank specificity diagnostics."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np, torch
import sys; sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from long_memory_protocol import paired_cluster_bootstrap
METRICS=('root_post_w5_auc_cm_windows','constraint_post_w5_auc_cm_windows','root_post_w5_slope_cm_per_window','constraint_post_w5_slope_cm_per_window','root_final_window_cm','constraint_final_window_cm')
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def pair(base,cand,seed):
 a,b=rows(base),rows(cand)
 if [x['panel_id'] for x in a]!=[x['panel_id'] for x in b]: raise ValueError('panel order mismatch')
 return {m:paired_cluster_bootstrap([{'actor_uid':x['actor_uid'],'id':x['motion_id'],'difference':float(y[m])-float(x[m])} for x,y in zip(a,b)],value_key='difference',resamples=20000,seed=seed) for m in METRICS}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--output',required=True); p.add_argument('--bank',required=True); p.add_argument('--panel',required=True); p.add_argument('--pairing',required=True); p.add_argument('--normal-name',default='20s_normal'); a=p.parse_args(); root=Path(a.root); normal=root/a.normal_name/'per_sample.jsonl'; pairings=json.loads(Path(a.pairing).read_text())['mappings']; comps={}
 for i,name in enumerate(('own','nearest','random','farthest')):
  path=(root/name/'per_sample.jsonl' if name!='own' else root/'20s_own_replay/per_sample.jsonl')
  if path.exists() and name!='own': comps[name]=pair(normal,path,20261000+i)
  elif path.exists(): comps[name]=pair(normal,path,20261000+i)
 bank=torch.load(a.bank,map_location='cpu'); panel=rows(a.panel); ids=[str(x['id']) for x in panel]; windows=sorted({k[1] for k in bank}); blocks=sorted({k[2] for k in bank}); layers=sorted({k[3] for k in bank});
 # Actor-balanced common/residual diagnostics at each aligned index.
 diag=[]
 for block in blocks:
  for layer in layers:
   for w in windows:
    values=[bank[(sid,w,block,layer)] for sid in ids if (sid,w,block,layer) in bank]; stack=torch.stack(values).float(); common=stack.mean(0); norms=stack.flatten(1).norm(dim=1); cos=torch.nn.functional.cosine_similarity(stack.flatten(1),common.flatten().unsqueeze(0),dim=1); diag.append({'block':block,'layer':layer,'completed_windows':w,'trajectory_cosine_to_common_mean':float(cos.mean()),'trajectory_variance':float(stack.var(dim=0).mean()),'common_norm':float(common.norm()),'mean_memory_norm':float(norms.mean()),'common_relative_norm':float(common.norm()/norms.mean())})
 result={'schema_version':'d3b_results_v1','root':str(root.resolve()),'panel':str(Path(a.panel).resolve()),'panel_sha256':hashlib.sha256(Path(a.panel).read_bytes()).hexdigest(),'bank':str(Path(a.bank).resolve()),'pairing':str(Path(a.pairing).resolve()),'comparisons':comps,'common_specific_diagnostics':diag,'provenance':{'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'checkpoint_sha256':'f197e8d7c12fca6c82db0d36a1e31ccf34163074ec8ec9aa847f0a25ca696020','bootstrap_resamples':20000}}
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 Path(Path(a.output).with_name('D3B_LAYERWISE_SPECIFICITY_ANALYSIS.md')).write_text(layerwise_md(diag))
 Path(Path(a.output).with_name('D3B_COMMON_SPECIFIC_DECOMPOSITION_RESULTS.json')).write_text(json.dumps({'schema_version':'d3b_common_specific_diagnostic_v1','interpretation':'offline actor-balanced common/residual geometry; averaged-vector generation interventions were not used as confirmatory evidence','diagnostics':diag},indent=2)+'\n')
def layerwise_md(diag):
 lines=['# D3b Layerwise Specificity Diagnostic','','This is a descriptive bank analysis, not a post-hoc selection of a main layer. Values are pooled over aligned windows.','', '| Block | Layer | Mean trajectory cosine | Mean trajectory variance |', '|---|---:|---:|---:|']
 by={}
 for d in diag: by.setdefault((d['block'],d['layer']),[]).append(d)
 for (b,l),v in sorted(by.items()): lines.append(f"| {b} | {l} | {np.mean([x['trajectory_cosine_to_common_mean'] for x in v]):.4f} | {np.mean([x['trajectory_variance'] for x in v]):.6f} |")
 lines += ['', 'Lower cosine and higher variance indicate relatively more trajectory-specific memory. No layer was selected for a primary claim.']; return '\n'.join(lines)+'\n'
if __name__=='__main__': main()
