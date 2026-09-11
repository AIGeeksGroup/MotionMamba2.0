"""Materialize compact D3 result artifacts and paper-facing summary."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO/'src'))
from long_memory_protocol import paired_cluster_bootstrap
ROOT=Path('/data4/zts/result/stage_d3_memory_specificity_20260905')
DOC=REPO/'docs'; METRICS=('root_post_w5_auc_cm_windows','constraint_post_w5_auc_cm_windows')
def load(p): return json.loads(p.read_text())
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def pair(a,b,seed):
 A,B=rows(a),rows(b); return {m:paired_cluster_bootstrap([{'actor_uid':u['actor_uid'],'id':u['motion_id'],'difference':float(v[m])-float(u[m])} for u,v in zip(A,B)],value_key='difference',resamples=20000,seed=seed) for m in METRICS}
def ps(condition): return ROOT/condition/'40s_in_horizon/per_sample.jsonl'
analysis=load(DOC/'D3_PERSISTENT_PERMUTATION_RESULTS.json')
freeze={'schema_version':'d3_freeze_recovery_v1','B_minus_A':analysis['comparisons']['one_shot_permute_memory'],'C_minus_A':analysis['comparisons']['freeze_write_after_w4'],'D_minus_A':analysis['comparisons']['one_shot_permute_plus_freeze_write'],'D_minus_C':pair(ps('freeze_write_after_w4'),ps('one_shot_permute_plus_freeze_write'),20260920)}
(DOC/'D3_FREEZE_RECOVERY_RESULTS.json').write_text(json.dumps(freeze,indent=2)+'\n')
sweep={}
for step in (250,500,1000,2000):
 p=ROOT/f'checkpoint_sweep/step{step}'
 normal=p/'normal/40s_in_horizon/per_sample.jsonl'
 sweep[str(step)]={'checkpoint':load(p/'normal/40s_in_horizon/result.json')['checkpoint'],'provenance':('D2R parent' if step==250 else 'D2R-L continuation'),'persistent_minus_normal':pair(normal,p/'persistent_read_permutation/40s_in_horizon/per_sample.jsonl',20260920+step),'one_shot_minus_normal':pair(normal,p/'permute_memory/40s_in_horizon/per_sample.jsonl',20261920+step),'write_off_minus_normal':pair(normal,p/'write_off/40s_in_horizon/per_sample.jsonl',20262920+step),'readout_off_minus_normal':pair(normal,p/'readout_off/40s_in_horizon/per_sample.jsonl',20263920+step)}
sweep['5000']={'checkpoint':'/data4/zts/result/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt','provenance':'D2R-L continuation','persistent_minus_normal':analysis['comparisons']['persistent_read_permutation'],'one_shot_minus_normal':analysis['comparisons']['one_shot_permute_memory'],'write_off_minus_normal':analysis['comparisons']['write_off'],'readout_off_minus_normal':analysis['comparisons']['readout_off']}
(DOC/'D3_CHECKPOINT_PERMUTATION_RESULTS.json').write_text(json.dumps({'schema_version':'d3_checkpoint_sweep_v1','checkpoints':sweep},indent=2)+'\n')
(DOC/'D3_SEMANTIC_MISMATCH_RESULTS.json').write_text(json.dumps({'schema_version':'d3_semantic_mismatch_v1','random_persistent_minus_normal':analysis['comparisons']['persistent_read_permutation'],'semantic_persistent_minus_normal':analysis['comparisons']['semantic_mismatch_persistent_read_permutation'],'interpretation':'With batch_size=2, the only derangement is the same swap for random and semantic pairing; semantic mismatch is therefore not an independent stronger intervention on this panel.'},indent=2)+'\n')

def fmt(x):
 return f"{x['mean_difference']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"
c=analysis['comparisons']; dc=freeze['D_minus_C']
md=f'''# D3 Memory Specificity Audit: Final Results

Date: 2026-09-05

Decision: **STOP**

Primary mechanism classification: **Case D - insufficient statistical power**

`ELIGIBLE_FOR_STAGE_E = false`

## Outcome

Persistent-memory write/read remains causally important, but this D3 run does not establish that the 5k memory content is trajectory-specific. Persistent read permutation worsened the mean Root and Constraint post-W5 AUC, but both actor-cluster 95% CIs cross zero. The required dual-AUC gate therefore fails.

| Intervention minus normal (lower is better) | Root post-W5 AUC | Constraint post-W5 AUC | Confirmatory conclusion |
|---|---:|---:|---|
| One-shot durable permutation | {fmt(c['one_shot_permute_memory']['root_post_w5_auc_cm_windows'])} | {fmt(c['one_shot_permute_memory']['constraint_post_w5_auc_cm_windows'])} | Null; reproduces 5k ambiguity |
| Persistent read permutation | {fmt(c['persistent_read_permutation']['root_post_w5_auc_cm_windows'])} | {fmt(c['persistent_read_permutation']['constraint_post_w5_auc_cm_windows'])} | Positive mean, CI crosses zero |
| Freeze write after W4 | {fmt(c['freeze_write_after_w4']['root_post_w5_auc_cm_windows'])} | {fmt(c['freeze_write_after_w4']['constraint_post_w5_auc_cm_windows'])} | Null control |
| One-shot permutation + freeze | {fmt(c['one_shot_permute_plus_freeze_write']['root_post_w5_auc_cm_windows'])} | {fmt(c['one_shot_permute_plus_freeze_write']['constraint_post_w5_auc_cm_windows'])} | Positive mean, CI crosses zero |
| Write off | {fmt(c['write_off']['root_post_w5_auc_cm_windows'])} | {fmt(c['write_off']['constraint_post_w5_auc_cm_windows'])} | Significant harm |
| Readout off after W4 | {fmt(c['readout_off']['root_post_w5_auc_cm_windows'])} | {fmt(c['readout_off']['constraint_post_w5_auc_cm_windows'])} | Significant harm |

The identity-isolating `D-C` comparison (permuted frozen minus correct frozen) was Root {fmt(dc['root_post_w5_auc_cm_windows'])} and Constraint {fmt(dc['constraint_post_w5_auc_cm_windows'])}. Both CIs cross zero, so self-recovery is plausible but not proven.

## Checkpoint trajectory

| Total step | Persistent permutation Root AUC gap | Constraint AUC gap |
|---:|---:|---:|
'''
for step in (250,500,1000,2000,5000):
 x=sweep[str(step)]['persistent_minus_normal']; md+=f"| {step} | {fmt(x['root_post_w5_auc_cm_windows'])} | {fmt(x['constraint_post_w5_auc_cm_windows'])} |\n"
md+='''

The mean specificity gap tends to increase with training, but no checkpoint has both primary AUC CIs above zero. This pattern supports Case D more than Case C: it is not evidence of homogeneous memory, because effects are consistently in the harmful direction and persistent permutation significantly worsened Root slope at 5k. It is still insufficient for the trajectory-specific claim.

Step 250 is the verified D2R parent checkpoint; steps 500-5000 are D2R-L continuation checkpoints. The requested no-delta replication was conditionally gated on full first establishing specificity. Because that condition failed, no-delta was not run and no delta claim is made.

## Protocol and implementation audit

- Checkpoint SHA-256 was recomputed as `f197e8d7c12fca6c82db0d36a1e31ccf34163074ec8ec9aa847f0a25ca696020`.
- Fixed BONES-SEED selection panel: 16 motions / 7 actors, SHA-256 `41909e76aa2a0011bb76cddec77048dd4bdb9aaa4cb8feac78a42e57dfa4579a`.
- The selection manifest cannot supply the requested 32 motions / 10 actors under the exact 40s span. No locked-final actor was read.
- Persistent permutation applies at every post-W4 read with one fixed derangement. Durable memory checksums are identical before/after each read; Attention K/V and Mamba-state checksums are recorded separately.
- Semantic mismatch could not become stronger than random pairing because the locked protocol uses batch size two: only one derangement exists per batch. Its result is identical and is not presented as independent confirmation.
- Exact K=2 tensor traces show one-shot recovery distance peaking at 7.87 after W5 and declining to 2.74 by W19. Across-layer averages are: trajectory cosine 0.80, temporal cosine 0.99, slot-collapse cosine 0.11, and per-element trajectory variance 0.088. Slots do not collapse, and the decay is supporting evidence for gradual self-recovery, but the confirmatory AUC tests remain underpowered.

## Scientific boundary

Currently established: **Persistent memory write/read improves long-horizon BONES selection performance.**

D3 sought to establish: **Using another trajectory's persistent memory causally harms future generation after W4 eviction.** The mean effect supports this direction, but the two primary CIs cross zero, so it is not established.

Still not established: **The 5k memory necessarily stores unique long-term history for each trajectory.**

No training was run, no weights were modified, and Stage E was not started. The correct next action is to obtain a larger actor-diverse selection protocol or an independently preregistered public-data panel, then rerun persistent/semantic permutation with batch-level mappings that allow genuinely different semantic pairings.

Raw evaluation and tensor artifacts are under `/data4/zts/result/stage_d3_memory_specificity_20260905`; large tensors are intentionally not copied into Git.
'''
(DOC/'D3_MEMORY_SPECIFICITY_FINAL_SUMMARY.md').write_text(md)
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figures=DOC/'figures'; figures.mkdir(exist_ok=True)
 steps=[250,500,1000,2000,5000]
 series={'One-shot':'one_shot_minus_normal','Persistent':'persistent_minus_normal','Write off':'write_off_minus_normal','Readout off':'readout_off_minus_normal'}
 fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharex=True)
 for ax,metric,title in zip(axes,METRICS,('Root AUC gap','Constraint AUC gap')):
  ax.axhline(0,color='black',lw=.8)
  for label,key in series.items(): ax.plot(steps,[sweep[str(x)][key][metric]['mean_difference'] for x in steps],marker='o',label=label)
  ax.set(xlabel='Optimizer step',ylabel='Intervention - normal (cm windows)',title=title)
 axes[1].legend(fontsize=7); fig.tight_layout()
 fig.savefig(figures/'d3_checkpoint_permutation_gap.png',dpi=180); fig.savefig(figures/'d3_checkpoint_permutation_gap.pdf'); plt.close(fig)
 names=['Random persistent','Semantic persistent']; rv=[c['persistent_read_permutation'][METRICS[0]]['mean_difference'],c['semantic_mismatch_persistent_read_permutation'][METRICS[0]]['mean_difference']]
 cv=[c['persistent_read_permutation'][METRICS[1]]['mean_difference'],c['semantic_mismatch_persistent_read_permutation'][METRICS[1]]['mean_difference']]
 x=range(2); fig,ax=plt.subplots(figsize=(5,3.5)); ax.bar([v-.18 for v in x],rv,width=.36,label='Root'); ax.bar([v+.18 for v in x],cv,width=.36,label='Constraint'); ax.set_xticks(list(x),names); ax.set_ylabel('AUC gap (cm windows)'); ax.legend(); fig.tight_layout(); fig.savefig(figures/'d3_random_vs_semantic_permutation.png',dpi=180); fig.savefig(figures/'d3_random_vs_semantic_permutation.pdf'); plt.close(fig)
except ImportError:
 pass
print(md)
