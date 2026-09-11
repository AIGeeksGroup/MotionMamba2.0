"""Prospective actor-cluster bootstrap planning from the locked D3 effects."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--normal',required=True); p.add_argument('--farthest',required=True); p.add_argument('--output',required=True); p.add_argument('--markdown',required=True); p.add_argument('--seed',type=int,default=20260906); p.add_argument('--resamples',type=int,default=5000); a=p.parse_args()
 A,B=rows(a.normal),rows(a.farthest); metrics=['root_post_w5_auc_cm_windows','constraint_post_w5_auc_cm_windows']; by={m:{} for m in metrics}
 for x,y in zip(A,B):
  if x['panel_id']!=y['panel_id'] or x['actor_uid']!=y['actor_uid']: raise ValueError('unpaired rows')
  for m in metrics: by[m].setdefault(str(x['actor_uid']),[]).append(float(y[m])-float(x[m]))
 actor_effect={m:np.asarray([np.mean(v) for v in d.values()]) for m,d in by.items()}; rng=np.random.default_rng(a.seed); scenarios={}
 for scale in (1.0,.75,.5):
  vals={}
  for n in range(4,31):
   ok=0
   for _ in range(a.resamples):
    lows=[]
    for m in metrics:
     sample=rng.choice(actor_effect[m],size=n,replace=True)*scale; lows.append(float(np.mean(sample)-1.96*np.std(sample,ddof=1)/np.sqrt(n)))
    ok += int(all(x>0 for x in lows))
   vals['n'+str(n)]=ok/a.resamples
  scenarios[str(scale)]={'effect_scale':scale,'probability_dual_ci_lower_gt_zero':vals}
 result={'schema_version':'d3b_power_analysis_v1','source':'locked D3 persistent read permutation actor-level paired effects','seed':a.seed,'resamples':a.resamples,'actors_observed':len(next(iter(by.values()))),'actor_clusters_observed':len(actor_effect[metrics[0]]),'metrics':metrics,'scenarios':scenarios}
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 lines=['# D3b Prospective Power Analysis','',f"Observed source actors: **{result['actors_observed']}**; simulation resamples: **{a.resamples}**; seed: `{a.seed}`.",'','This is planning only, not confirmatory evidence. Actor effects are resampled with replacement from the locked D3 persistent-read intervention; the estimate is unstable because only seven actors are available.','', '| Effect scale | Actors for >=80% dual-CI probability |', '|---:|---:|']
 for sc,s in scenarios.items():
  hit=next((int(k[1:]) for k,v in s['probability_dual_ci_lower_gt_zero'].items() if v>=.8),None); lines.append(f'| {sc}x observed | {hit if hit is not None else "not reached (4-30)"} |')
 lines += ['', 'The 50% and 75% scenarios are sensitivity analyses. The simulation does not justify changing the locked panel or selecting donors after observing errors.']
 Path(a.markdown).write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
