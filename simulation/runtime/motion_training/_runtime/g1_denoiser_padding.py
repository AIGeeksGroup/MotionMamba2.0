"""Isolated valid-prefix intervention for diagnosing unused future tokens.

Not installed by default. Supports equal contiguous valid prefixes only;
mixed-length training and sparse future constraints need a general mask path.
"""
from contextlib import contextmanager
import inspect
import types
import torch


@contextmanager
def compact_valid_prefix(raw, num_frames_per_token=4):
    original=raw.forward
    signature=inspect.signature(original)
    events=[]
    def forward(self,*args,**kwargs):
        call=dict(signature.bind(*args,**kwargs).arguments)
        valid=call['history_token_mask']|call['generation_token_mask']|call['future_token_mask']
        x=call['x']
        indices=torch.arange(x.shape[1],device=x.device).expand_as(valid)
        end=int(torch.where(valid,indices+1,torch.zeros_like(indices)).max())
        if end==0 or not bool(valid[:,:end].all()):
            raise ValueError('Valid-prefix intervention requires equal contiguous masks')
        events.append(dict(input_tokens=x.shape[1],valid_tokens=end,batch=x.shape[0]))
        if end==x.shape[1]:return original(*args,**kwargs)
        frames=end*num_frames_per_token
        call['x']=x[:,:end]
        for name in ['history_token_mask','generation_token_mask','future_token_mask']:
            call[name]=call[name][:,:end]
        for name in ['history_mask','generation_mask','future_mask','motion_mask','observed_motion']:
            if call.get(name) is not None:call[name]=call[name][:,:frames]
        call['future_len']=(frames-call['history_len']-call['generation_len']).clamp_min(0)
        output=original(**call)
        return torch.cat((output,output.new_zeros(output.shape[0],x.shape[1]-end,output.shape[-1])),1)
    raw.forward=types.MethodType(forward,raw)
    try:yield events
    finally:raw.forward=original
