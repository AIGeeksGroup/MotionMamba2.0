"""Versioned Hybrid attention visibility and inactive-suffix compaction.

The attention mask excludes invalid keys/queries. Mamba still evaluates dense
time steps for gaps before real future constraints; accepted writes retain the
existing generation-only commit. This does not claim sparse recurrent stepping.
"""
import inspect
import types
import torch

PADDING_CONTRACT='g1_hybrid_attention_mask_v1'


def install_padding_contract(model, version):
    if version=='legacy':return
    if version!=PADDING_CONTRACT:raise ValueError(f'Unknown padding contract: {version}')
    raw=model.denoiser.model
    current=getattr(raw,'g1_padding_contract','legacy')
    if current==version:return
    if current!='legacy':raise ValueError('Cannot replace an installed padding contract')
    blocks=[raw.root_model,raw.body_model]
    if any(getattr(b,'sequence_mode',None)!='hybrid' for b in blocks):
        raise ValueError('Hybrid padding contract requires Hybrid root/body blocks')
    original=raw.forward
    signature=inspect.signature(original)
    nfpt=int(raw.num_frames_per_token)
    def forward(self,*args,**kwargs):
        call=dict(signature.bind(*args,**kwargs).arguments)
        x=call['x']
        valid=call['history_token_mask']|call['generation_token_mask']|call['future_token_mask']
        indices=torch.arange(x.shape[1],device=x.device).expand_as(valid)
        end=int(torch.where(valid,indices+1,torch.zeros_like(indices)).max())
        if end==0:raise ValueError('A denoiser call must contain valid tokens')
        if end==x.shape[1]:return original(*args,**kwargs)
        call['x']=x[:,:end]
        for key in ['history_token_mask','generation_token_mask','future_token_mask']:
            call[key]=call[key][:,:end]
        for key in ['history_mask','generation_mask','future_mask','motion_mask','observed_motion']:
            if call.get(key) is not None:call[key]=call[key][:,:end*nfpt]
        call['future_len']=(end*nfpt-call['history_len']-call['generation_len']).clamp_min(0)
        result=original(**call)
        return torch.cat((result,result.new_zeros(result.shape[0],x.shape[1]-end,result.shape[-1])),1)
    for block in blocks:block.hybrid_attention_mask_enabled=True
    raw.forward=types.MethodType(forward,raw)
    raw.g1_padding_contract=version
