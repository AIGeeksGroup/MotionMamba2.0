"""Explicit output ownership, identity and shared budget; no import I/O."""
import contextlib, fcntl, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(p)
def append(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(v,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def identity(x):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for z in v:visit(z)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()
def copy_state(x,device='cpu'):
    if torch.is_tensor(x):return x.detach().clone().to(device)
    if isinstance(x,dict):return {k:copy_state(v,device) for k,v in x.items()}
    if isinstance(x,list):return [copy_state(v,device) for v in x]
    if isinstance(x,tuple):return tuple(copy_state(v,device) for v in x)
    return x
class Budget:
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def ticket(self,category,**details):
        with (self.root/'budget.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);rr=rows(self.root/'UPDATES.jsonl');limits=read(self.root/'LIMITS.json') if (self.root/'LIMITS.json').exists() else {};caps=limits.get('caps',{'checks':16,'B':128,'C':128,'teacher':512})
            assert category in caps and sum(r['category']==category for r in rr)<caps[category],('budget',category)
            if category!='teacher':assert sum(r['category']!='teacher' for r in rr)<limits.get('total_updates',272)
            row=dict(category=category,time=time.time(),**details);append(self.root/'UPDATES.jsonl',row);return row
    @contextlib.contextmanager
    def timed(self,name,closing=False,devices=1):
        start=time.time();pid=str(os.getpid())
        with (self.root/'budget.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            a=read(self.root/'ACTIVE.json') if (self.root/'ACTIVE.json').exists() else {}
            used=sum(r['seconds']*r.get('devices',1) for r in rows(self.root/'GPU_TIME.jsonl'))+sum((start-r['start'])*r.get('devices',1) for r in a.values())
            limits=read(self.root/'LIMITS.json') if (self.root/'LIMITS.json').exists() else {}
            cap=limits.get('gpu_hours',12)*(1 if closing else .75)
            assert used<cap*3600,('GPU_budget',used)
            a[pid]=dict(name=name,start=start,device=os.environ.get('CUDA_VISIBLE_DEVICES'),devices=devices);write(self.root/'ACTIVE.json',a)
        try:yield
        finally:
            with (self.root/'budget.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX);a=read(self.root/'ACTIVE.json');a.pop(pid,None);write(self.root/'ACTIVE.json',a)
                append(self.root/'GPU_TIME.jsonl',dict(name=name,seconds=time.time()-start,device=os.environ.get('CUDA_VISIBLE_DEVICES'),devices=devices))
