"""Frozen, indexed JSONL sampler plan; every batch uses one H/F layout."""
import json,collections,os,time,hashlib
from pathlib import Path
import numpy as np
from . import data
from .schema import VERSION,validate_record
from .io import read,write,rows,sha
from ._compat.fusion_index import excluded,caption_task_tags
from g1_motion_contract import CONTRACT
class Plan:
 def __init__(self,header):
  self.header=header;self.offsets=np.load(header['offsets_path'],mmap_mode='r');self.file=open(header['plan_path'],'rb')
 def __len__(self):return self.header['planned_exposures']
 def __getitem__(self,s):
  assert isinstance(s,slice) and s.step is None and s.start%32==0 and s.stop-s.start==32
  self.file.seek(int(self.offsets[s.start//32]));return [json.loads(self.file.readline()) for _ in range(32)]
 def __iter__(self):
  with open(self.header['plan_path']) as f:
   for line in f:yield json.loads(line)
def load(path):
 header=read(path)
 if header.get('format')=='indexed_jsonl_v1':header['plan']=Plan(header)
 return header
def prepare(cfg,dest):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 manifest=dest/'MANIFEST.json'
 if manifest.exists():
  h=read(manifest);assert h['planned_exposures']==cfg['max_updates']*32
  assert h['index_sha256']==sha(cfg['data_index']) and h['exclusions_sha256']==sha(cfg['exclusions'])
  assert h['plan_sha256']==sha(h['plan_path']) and h['offsets_sha256']==sha(h['offsets_path'])
  return h
 records=rows(cfg['data_index']);ex=read(cfg['exclusions']);rng=np.random.default_rng(cfg['seed']);groups={};devfamilies=set()
 for source in ['bones','humanml3d','hiphi']:
  candidates=sorted({r['source_family_id'] for r in records if r['dataset']==source and r['project_split']=='train' and not excluded(r,ex) and data.intervals(r,248)})
  devfamilies.update(candidates[:2])
 for source in ['bones','humanml3d','hiphi']:
  for h,f in data.BUCKETS:
   by=collections.defaultdict(list)
   for r in records:
    if r['dataset']==source and r['project_split']=='train' and not excluded(r,ex) and r['source_family_id'] not in devfamilies:
     opts=data.intervals(r,h+52+f)
     if opts:by[r['source_family_id']].append((r,opts))
   assert by,(source,h,f);groups[source,h,f]=(sorted(by),by)
 dev=[]
 for family in sorted(devfamilies):
  r=next(r for r in records if r['source_family_id']==family and data.intervals(r,248));lo,hi,caption,span=data.intervals(r,248)[0]
  dev.append(dict(family=family,dataset=r['dataset'],feature_path=r['feature_path'],crop_start=lo,caption=caption,caption_span=span,H=0,F=196,G=52,seeds=[202610034401,202610034402],condition='root_waypoint',constraint_frame=103))
 out=dest/'PLAN.jsonl';tmp=out.with_suffix('.tmp');offsets=[];texts=set();families=set();source_counts=collections.Counter();hfcounts=collections.Counter();condcounts=collections.Counter();future=0;f_nonempty=0
 hc=[];sc=[];cc=[];started=time.time()
 with tmp.open('wb') as fp:
  for step in range(cfg['max_updates']):
   if not hc:hc=[v for v,n in zip(data.BUCKETS,[2,5,4,2,4,3]) for _ in range(n)];rng.shuffle(hc)
   h,f=hc.pop();offsets.append(fp.tell())
   for lane in range(32):
    if not sc:sc=['bones']*14+['humanml3d']*3+['hiphi']*3;rng.shuffle(sc)
    if not cc:cc=['none']*5+[v for v in data.CONDITIONS[1:] for _ in range(4)];rng.shuffle(cc)
    source=sc.pop();condition=cc.pop();keys,by=groups[source,h,f];family=keys[int(rng.integers(len(keys)))];rr=by[family];r,opts=rr[int(rng.integers(len(rr)))];lo,hi,caption,span=opts[int(rng.integers(len(opts)))];start=int(rng.integers(lo,hi+1));i=step*32+lane
    isfuture=False
    if f and condition!='none':isfuture=(f_nonempty%2==0);f_nonempty+=1
    record=validate_record(dict(schema=VERSION,id=f'{cfg["name"]}:{i}',source_family_id=family,dataset=source,split='TRAIN',feature_path=r['feature_path'],sample_id=r['sample_id'],crop_start=start,crop_end=start+h+52+f,caption=caption,caption_span=span,caption_tags=caption_task_tags(caption),native_start_sec=r.get('native_start_sec',0),supervision='GT',H=h,F=f,G=52,fps=25,skeleton='g1',representation='physical414',normalizer_version=CONTRACT,codec_version='g1_history_codec_104_v1',condition=condition,out_of_horizon=isfuture,seed=int(cfg['seed']+1000+i),quality_status=r.get('status'),retarget_quality=r.get('kinematic_status'),source_motion_id=r.get('source_motion_id'),valid_frames=h+52+f,earlier_prefix_frames=0,horizontal_rotation_rad=0.,augmentation='identity; world translation applied jointly to motion and conditions'))
    fp.write((json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n').encode());texts.add(caption);families.add(family);source_counts[source]+=1;hfcounts[str((h,f))]+=1;condcounts[condition]+=1;future+=isfuture
   if (step+1)%10000==0:print('PREPARED',step+1,'/',cfg['max_updates'],'seconds',round(time.time()-started),flush=True)
  offsets.append(fp.tell());fp.flush();os.fsync(fp.fileno())
 tmp.replace(out);np.save(dest/'OFFSETS.npy',np.array(offsets,dtype=np.int64))
 texts.update(x['caption'] for x in dev);write(dest/'CAPTIONS.json',sorted(texts))
 header=dict(format='indexed_jsonl_v1',schema=VERSION,recipe=cfg['name'],plan_path=str(out),offsets_path=str(dest/'OFFSETS.npy'),plan_sha256=sha(out),offsets_sha256=sha(dest/'OFFSETS.npy'),caption_sha256=sha(dest/'CAPTIONS.json'),index_sha256=sha(cfg['data_index']),exclusions_sha256=sha(cfg['exclusions']),additional_dev=dev,dev_families=sorted(devfamilies),planned_exposures=cfg['max_updates']*32,unique_families=len(families),source_counts=dict(source_counts),HGF_counts=dict(hfcounts),conditions=dict(condcounts),future_observed_before_drop=future,future_predictions_supervised=False,source_family_uniform=True,batch_layout_shared=True,condition_cycle_counts=[5,4,4,4,4,4],text_count=len(texts),project_choices=True)
 write(manifest,header);return header
