"""Analyze exact per-layer K=2 memory tensors from D3 paired rollouts."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

def load_trace(path):
    records=torch.load(path,map_location='cpu',weights_only=False); out={}
    for record in records:
        slots=record['slots']
        for row,sample in enumerate(record['sample_ids']):
            out[(sample,record['block'],int(record['layer']),int(record['completed_windows']))]=slots[row]
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--normal',required=True); p.add_argument('--permuted',required=True); p.add_argument('--output',required=True); p.add_argument('--figures',required=True)
    a=p.parse_args(); normal=load_trace(a.normal); permuted=load_trace(a.permuted)
    keys=sorted(normal); windows=sorted({k[3] for k in keys}); layers=sorted({(k[1],k[2]) for k in keys})
    layer_stats={}; recovery=defaultdict(list)
    for block,layer in layers:
        group={w:[] for w in windows}
        temporal=[]; collapse=[]
        for key,value in normal.items():
            if key[1:3]!=(block,layer): continue
            group[key[3]].append(value)
            collapse.append(float(F.cosine_similarity(value[0],value[1],dim=0)))
            previous=normal.get((key[0],block,layer,key[3]-1))
            if previous is not None: temporal.append(float(F.cosine_similarity(value.flatten(),previous.flatten(),dim=0)))
            other=permuted.get(key)
            if other is not None and key[3]>=4:
                recovery[key[3]].append(float(torch.linalg.vector_norm(other-value)))
        cosine=[]; variance=[]; l2=[]
        for values in group.values():
            if len(values)<2: continue
            tensor=torch.stack(values); flat=F.normalize(tensor.flatten(1),dim=1); sim=flat@flat.T
            tri=torch.triu_indices(len(values),len(values),offset=1)
            cosine.extend(sim[tri[0],tri[1]].tolist()); variance.append(float(tensor.var(dim=0,unbiased=False).mean()))
            l2.extend(torch.cdist(tensor.flatten(1),tensor.flatten(1))[tri[0],tri[1]].tolist())
        layer_stats[f'{block}:{layer}']={'across_trajectory_cosine_mean':float(np.mean(cosine)),'across_trajectory_l2_mean':float(np.mean(l2)),'per_slot_variance_mean':float(np.mean(variance)),'temporal_cosine_mean':float(np.mean(temporal)),'slot_collapse_cosine_mean':float(np.mean(collapse))}
    recovery_curve={str(w):float(np.mean(values)) for w,values in sorted(recovery.items())}
    diagnostics=defaultdict(lambda:defaultdict(list))
    state_path=Path(a.normal).with_name('state.jsonl')
    for line in state_path.read_text().splitlines():
        record=json.loads(line)
        for block,records in record['blocks'].items():
            for item in records:
                key=f"{block}:{item['layer']}"
                for field in ('content_memory_norm','content_memory_update_norm','content_memory_write_gate_mean','content_memory_read_residual_scale','content_memory_proposal_norm','state_summary_norm','state_delta_norm','augmented_minus_base_attention_norm'):
                    if item.get(field) is not None: diagnostics[key][field].append(float(item[field]))
    diagnostic_summary={key:{field:float(np.mean(values)) for field,values in fields.items()} for key,fields in diagnostics.items()}
    result={'schema_version':'d3_memory_trace_v2','status':'COMPLETE','normal_trace':str(Path(a.normal).resolve()),'permuted_trace':str(Path(a.permuted).resolve()),'layers':layer_stats,'layerwise_runtime_diagnostics':diagnostic_summary,'one_shot_recovery_l2_by_completed_window':recovery_curve,'definitions':{'recovery':'L2(permuted durable memory, matched normal durable memory), averaged across samples/root-body/layers','slot_collapse':'cosine(slot0,slot1)','trajectory_variance':'mean elementwise variance across trajectories at matched layer/window'}}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    figures=Path(a.figures); figures.mkdir(parents=True,exist_ok=True)
    x=[int(k) for k in recovery_curve]; y=list(recovery_curve.values()); fig,ax=plt.subplots(figsize=(5.5,3.5)); ax.plot(x,y,marker='o'); ax.set(xlabel='Completed windows',ylabel='Memory L2 to matched normal'); fig.tight_layout(); fig.savefig(figures/'d3_memory_recovery_curve.png',dpi=180); fig.savefig(figures/'d3_memory_recovery_curve.pdf'); plt.close(fig)
    labels=list(layer_stats); values=[layer_stats[k]['per_slot_variance_mean'] for k in labels]; fig,ax=plt.subplots(figsize=(8,3.8)); ax.bar(range(len(labels)),values); ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=6); ax.set_ylabel('Across-trajectory slot variance'); fig.tight_layout(); fig.savefig(figures/'d3_layerwise_memory_variance.png',dpi=180); fig.savefig(figures/'d3_layerwise_memory_variance.pdf'); plt.close(fig)

if __name__=='__main__': main()
