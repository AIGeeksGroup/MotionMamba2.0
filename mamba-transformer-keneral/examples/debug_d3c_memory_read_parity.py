"""Deterministic FP32 read-state shadow/recorded replay debugger.

This isolates the exact memory-content read path from the ARDY sampler. It
uses the same MinimalAttentionKernel projection and attention implementation,
but keeps the fixture tiny enough to report every intermediate tensor.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.pop(1) if len(sys.path) > 1 and sys.path[1].endswith("/examples") else None
import torch
from minimal_attention_kernel import MinimalAttentionKernel, MinimalAttentionKernelConfig
from state_memory_operator.content_memory import PersistentContentMemory

def digest(x):
    if x is None: return "none"
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def stats(a,b):
    if a is None or b is None: return {"checksum_normal":digest(a),"checksum_replay":digest(b),"max_abs_diff":None,"mean_abs_diff":None,"relative_l2_diff":None,"first_mismatch":None}
    d=(a-b).detach().float(); nz=(d.abs()>0).nonzero(as_tuple=False)
    return {"checksum_normal":digest(a),"checksum_replay":digest(b),"max_abs_diff":float(d.abs().max()),"mean_abs_diff":float(d.abs().mean()),"relative_l2_diff":float(d.norm()/a.detach().float().norm().clamp_min(1e-12)),"first_mismatch":None if not len(nz) else nz[0].tolist()}
def run(attn, x, memory, cache, scale):
    base,_,aux0=attn.forward_chunk(x,cache=cache.clone(),do_read=True,extra_content=None)
    mem,_,aux=attn.forward_chunk(x,cache=cache.clone(),do_read=True,extra_content=memory)
    mk=attn._split_heads(attn.k_proj(memory)); mv=attn._split_heads(attn.v_proj(memory))
    read_scaled=base+scale*(mem-base)
    fused=read_scaled
    dit=x+fused
    return {"A_base":base,"A_mem":mem,"memory_K":mk,"memory_V":mv,"read_scaled_A":read_scaled,"fused_output":fused,"DiT_output":dit,"aux_base":aux0,"aux_mem":aux}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--report',required=True); p.add_argument('--artifact',required=True); a=p.parse_args()
 torch.manual_seed(20260906); torch.use_deterministic_algorithms(True)
 cfg=MinimalAttentionKernelConfig(d_model=16,num_heads=4,window_size_chunks=2,use_attention_sinks=False,num_sink_tokens=0,attention_backend='sdpa',allow_backend_fallback=False)
 attn=MinimalAttentionKernel(cfg).float().eval(); content=PersistentContentMemory(state_dim=4,d_model=16,slots=2,read_scale_init=.25).float().eval()
 x=torch.randn(2,3,16); slots=torch.randn(2,2,16); read_ready=content.read(slots).to(x.dtype).contiguous(); cache=attn.init_cache(batch_size=2,device=x.device,dtype=x.dtype)
 normal=run(attn,x,read_ready,cache,float(content.read_residual_scale)); shadow=run(attn,x,read_ready.detach().clone(),cache,float(content.read_residual_scale)); recorded=torch.save(read_ready.detach().clone(),a.artifact) if False else None
 stages=['durable_slots','read_ready_tensor','memory_K','memory_V','A_base','A_mem','read_scaled_A','fused_output','DiT_output']; pairs={'durable_slots':(slots,slots.clone()),'read_ready_tensor':(read_ready,read_ready.detach().clone())}; pairs.update({k:(normal[k],shadow[k]) for k in stages if k not in pairs})
 result={'schema_version':'d3c_p0_shadow_parity_v1','status':'PASS' if all(stats(*pairs[k])['max_abs_diff'] in (None,0.0) or stats(*pairs[k])['max_abs_diff']<=1e-6 for k in stages) else 'FAIL','deterministic':True,'dtype':str(x.dtype),'device':str(x.device),'attention_backend':'sdpa','shape':list(read_ready.shape),'read_ready_stride':list(read_ready.stride()),'stages':{k:stats(*pairs[k]) for k in stages},'read_residual_scale':float(content.read_residual_scale)}
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 first=next((k for k in stages if result['stages'][k]['max_abs_diff'] not in (None,0.0) and result['stages'][k]['max_abs_diff']>1e-6),'NONE')
 lines=['# D3c First-Divergence Report','','`FIRST_DIVERGENCE_STAGE = '+first+'`','', '| Stage | Normal checksum | Replay checksum | Max abs diff | First mismatch |','|---|---|---|---:|---|']
 for k in stages:
  s=result['stages'][k]; lines.append(f"| {k} | `{s['checksum_normal'][:12]}` | `{s['checksum_replay'][:12]}` | {s['max_abs_diff']} | {s['first_mismatch']} |")
 lines += ['', 'P0 compares the exact read-ready tensor and all downstream FP32 SDPA stages with identical hidden input, cache, projections, scale and seed.', '', 'The fixture is intentionally minimal and deterministic; it is an instrument calibration, not a model-quality or donor-causality result.']
 Path(a.report).write_text('\n'.join(lines)+'\n')
 # Save the exact read-ready tensor separately for P1/P2 recorded replay.
 torch.save({'read_ready':read_ready,'durable_slots':slots,'x':x},a.artifact)
if __name__=='__main__': main()
