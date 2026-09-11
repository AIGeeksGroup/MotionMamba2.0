"""Deterministic canonical-history state/output parity diagnostic."""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'ardy'),str(ROOT/'examples')]
for p in ('/data3/zts/packages/state-spaces-mamba','/data3/zts/packages/mamba_op_triton'): sys.path.insert(0,p)
from ardy.model import load_model
from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel
from ardy_adapter.history_commit import canonical_history_prime
from train_humanml3d_ardy_tiny import _replace_stats, _load_denoiser_weights

BASE=Path('/data4/zts/result')
CK={'minimal':BASE/'humanml3d_full_cachefix_20260820_minimal/minimal_humanml3d_ddp.pt','hybrid':BASE/'humanml3d_full_cachefix_20260820_hybrid/hybrid_humanml3d_ddp.pt'}

def tensor_max(a,b):
    return float((a.detach().float()-b.detach().float()).abs().max())
def state_diff(a,b):
    vals=[]
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor): return tensor_max(a,b)
    if hasattr(a,'__dict__') and hasattr(b,'__dict__'):
        for k in a.__dict__:
            if k.startswith('_') and k not in {'_state'}: continue
            if hasattr(b,k):
                try: vals.append(state_diff(getattr(a,k),getattr(b,k)))
                except Exception: pass
    elif isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        vals=[state_diff(x,y) for x,y in zip(a,b)]
    return max(vals,default=0.0)

def main():
    torch.manual_seed(20260822); out={}
    for mode,ck in CK.items():
        models=[]; raws=[]
        for _ in range(2):
            m=load_model('core',device='cuda:0',text_encoder=False); _replace_stats(m,str(BASE/'humanml3d_ardy_train'))
            install_ardy_sequence_kernel(m,ArdySequenceKernelConfig(mode=mode,attention_backend='sdpa',window_size_chunks=4,num_sink_tokens=4))
            _load_denoiser_weights(m.denoiser.model,torch.load(ck,map_location='cuda:0',weights_only=False))
            m.denoiser.model.eval(); models.append(m); raws.append(m.denoiser.model)
        history=torch.randn(2,10,148,device='cuda:0'); text=torch.randn(2,1,4096,device='cuda:0'); text_mask=torch.ones(2,1,dtype=torch.bool,device='cuda:0'); heading=torch.zeros(2,device='cuda:0')
        for raw in raws: canonical_history_prime(raw,history,text_feat=text,text_pad_mask=text_mask,first_heading_angle=heading,next_generation_frames=40,chunk_id=0)
        kv=[]; rec=[]
        for layer_a,layer_b in zip(raws[0].root_model._sequence_state.layers,raws[1].root_model._sequence_state.layers):
            ka,va,_=layer_a.attention_cache.get(); kb,vb,_=layer_b.attention_cache.get(); kv += [tensor_max(ka,kb),tensor_max(va,vb)]
        for sa,sb in zip(raws[0].root_model._sequence_state.layers,raws[1].root_model._sequence_state.layers):
            rec.append(state_diff(sa.mamba_state,sb.mamba_state))
        x=torch.randn(2,10,148,device='cuda:0'); h=torch.zeros(2,dtype=torch.long,device='cuda:0'); g=torch.full((2,),40,dtype=torch.long,device='cuda:0'); z=torch.zeros_like(h); hm=torch.zeros(2,40,dtype=torch.bool,device='cuda:0'); gm=torch.ones_like(hm); fm=torch.zeros_like(hm); hist_t=torch.zeros(2,10,dtype=torch.bool,device='cuda:0'); gen_t=torch.ones_like(hist_t); fut_t=torch.zeros_like(hist_t); text2=text; step=torch.full((2,),5,dtype=torch.long,device='cuda:0'); mm=torch.zeros(2,40,330,device='cuda:0'); out_a=raws[0](x,h,g,z,hm,gm,fm,hist_t,gen_t,fut_t,text2,text_mask,step,heading,mm,mm); out_b=raws[1](x,h,g,z,hm,gm,fm,hist_t,gen_t,fut_t,text2,text_mask,step,heading,mm,mm)
        out[mode]={'kv_max_abs_diff':max(kv),'mamba_state_max_abs_diff':max(rec), 'next_window_output_max_abs_diff':tensor_max(out_a,out_b), 'history_tokens':10,'position_origin':[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]}
        del models; torch.cuda.empty_cache()
    path=BASE/'cache_contract_gate_diagnostics_20260822.json'; path.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
