"""Inference dispatch (derived from validated historical v2): fuse Mamba-3 A/DT pointwise preprocessing.
Preserves eager intermediate casts; no weights/state mutation, no backward.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit
def _adt(A,DTIN,BIAS,DT,ADT,L:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
         AS0:tl.constexpr,AS1:tl.constexpr,DS0:tl.constexpr,DS1:tl.constexpr,
         FLOOR:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 h=i%H;l=(i//H)%L;b=i//(H*L);ok=i<N
 a=tl.load(A+b*AS0+l*AS1+h,ok,0).to(tl.float32)
 dt=tl.load(DTIN+b*DS0+l*DS1+h,ok,0).to(tl.float32)
 bias=tl.load(BIAS+h,ok,0).to(tl.float32)
 pos=tl.maximum(a,0.);neg=tl.minimum(a,0.)
 av=tl.minimum(-(pos+libdevice.div_rn(1.,(1.-neg))),-FLOOR)
 # torch add/softplus each materialize in the promoted dtype.
 u=(dt+bias).to(DT.dtype.element_ty).to(tl.float32)
 v=tl.where(u>20.,u,libdevice.log1p(libdevice.exp(u))).to(DT.dtype.element_ty)
 address=(b*H+h)*L+l
 tl.store(DT+address,v,ok)
 tl.store(ADT+address,av*v.to(tl.float32),ok)

def fused_adt(dd_a,dd_dt,dt_bias,a_floor):
 if torch.is_grad_enabled() and any(x.requires_grad for x in (dd_a,dd_dt,dt_bias)):
  raise RuntimeError('Inference prototype has no backward')
 assert dd_a.is_cuda and dd_a.shape==dd_dt.shape and dd_a.ndim==3
 assert dd_a.stride(-1)==dd_dt.stride(-1)==1 and dt_bias.is_contiguous()
 b,l,h=dd_a.shape
 dt=torch.empty((b,h,l),device=dd_a.device,dtype=torch.promote_types(dd_dt.dtype,dt_bias.dtype))
 adt=torch.empty((b,h,l),device=dd_a.device,dtype=torch.float32)
 _adt[(triton.cdiv(b*l*h,128),)](dd_a,dd_dt,dt_bias,dt,adt,l,h,b*l*h,*dd_a.stride()[:2],*dd_dt.stride()[:2],a_floor,128,enable_fp_fusion=False)
 return dt,adt


def preprocess(dd_a, dd_dt, dt_bias, a_floor, *, enabled=False, training=False):
    """Inference-only fusion; unsupported layouts/dtypes and grad use eager.

    No detached tensors, no runtime monkey patches, same softplus threshold.
    DT/ADT retain BHL semantics; supported fusion produces contiguous BHL.
    """
    supported = (
        enabled and not training and not torch.is_grad_enabled()
        and dd_a.is_cuda and dd_dt.device == dd_a.device == dt_bias.device
        and dd_a.ndim == 3 and dd_a.shape == dd_dt.shape
        and dd_a.numel() > 0 and dd_a.stride(-1) == dd_dt.stride(-1) == 1
        and dt_bias.shape == (dd_a.shape[-1],) and dt_bias.is_contiguous()
        and dd_a.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and dd_dt.dtype == dd_a.dtype
        and dt_bias.dtype in (torch.float32, dd_dt.dtype)
    )
    if supported:
        return fused_adt(dd_a, dd_dt, dt_bias, a_floor)
    from mamba_ssm.modules.mamba3 import heavy_tail_activation
    av = torch.clamp(-heavy_tail_activation(dd_a.float()), max=-a_floor)
    dt = torch.nn.functional.softplus(dd_dt + dt_bias)
    return dt.transpose(1, 2), (av * dt).transpose(1, 2)
