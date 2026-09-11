"""Pre-register nearest/random/farthest donor mappings from text embeddings."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--panel',required=True); p.add_argument('--text-index',required=True); p.add_argument('--embeddings',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=20260906); a=p.parse_args()
 panel=[json.loads(x) for x in Path(a.panel).read_text().splitlines() if x.strip()]; idx=json.loads(Path(a.text_index).read_text()); lookup={str(x['id']):(int(x['row']),str(x['caption'])) for x in idx['samples']}; ids=[str(x['id']) for x in panel]; actors={str(x['id']):str(x['actor_uid']) for x in panel}
 emb=np.load(a.embeddings,mmap_mode='r'); mat=np.asarray([emb[lookup[x][0]] for x in ids],dtype=np.float32); mat/=np.maximum(np.linalg.norm(mat,axis=1,keepdims=True),1e-12); sim=mat@mat.T; rng=np.random.default_rng(a.seed)
 def pick(i,mode):
  candidates=[j for j in range(len(ids)) if j!=i and actors[ids[j]]!=actors[ids[i]]];
  if not candidates: candidates=[j for j in range(len(ids)) if j!=i]
  if mode=='nearest': return min(candidates,key=lambda j:(-sim[i,j],ids[j]))
  if mode=='farthest': return min(candidates,key=lambda j:(sim[i,j],ids[j]))
  return int(rng.choice(candidates))
 mappings={'own':{x:x for x in ids}}
 for mode in ('nearest','random','farthest'): mappings[mode]={ids[i]:ids[pick(i,mode)] for i in range(len(ids))}
 out={'schema_version':'d3b_donor_pairing_v1','panel':str(Path(a.panel).resolve()),'panel_sha256':sha(Path(a.panel)),'text_index':str(Path(a.text_index).resolve()),'embeddings':str(Path(a.embeddings).resolve()),'seed':a.seed,'selection_source':'caption embedding cosine only, before inference','different_actor_preferred':True,'mappings':mappings}
 out['pairwise_distance']={mode:{rid:float(1-sim[i,ids.index(did)]) for i,(rid,did) in enumerate(m.items())} for mode,m in mappings.items()}
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
