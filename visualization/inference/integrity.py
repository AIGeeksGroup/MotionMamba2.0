"""Exact tensor/state hashing for inference qualification; no training dependency."""
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def state_tree(x):
    if torch.is_tensor(x) or isinstance(x,np.ndarray): return x
    if isinstance(x,dict): return {k:state_tree(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [state_tree(v) for v in x]
    if hasattr(x,'__dict__'): return state_tree(vars(x))
    return x

def identity(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            v=v.detach().cpu().contiguous()
            h.update(str((str(v.dtype),tuple(v.shape))).encode())
            # Singleton tensors can be 'contiguous' with a non-unit stride.
            flat=torch.empty(v.numel(),dtype=v.dtype,device='cpu')
            flat.copy_(v.reshape(-1))
            h.update(flat.view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=str):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(b'list')
            for x in v:visit(x)
        elif hasattr(v,'__dict__'):visit(vars(v))
        else:h.update(json.dumps(v,sort_keys=True,default=str).encode()+b'\0')
    visit(value)
    return h.hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def stats(client):
    return {name:dict(replays=getattr(getattr(b.sequence_stack,'_inference_graphs',None),'replays',0),fallbacks=getattr(getattr(b.sequence_stack,'_inference_graphs',None),'fallbacks',0),build_events=getattr(getattr(b.sequence_stack,'_inference_graphs',None),'events',[])) for name,b in [('root',client.controller.root_block),('body',client.controller.body_block)]}
