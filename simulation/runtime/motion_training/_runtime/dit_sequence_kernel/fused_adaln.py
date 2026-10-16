"""Optional inference LayerNorm + modulation, with explicit eager fallback."""
import torch
import triton
import triton.language as tl

@triton.jit
def _lnmod(X,S,C,Y,L:tl.constexpr,D:tl.constexpr,EPS:tl.constexpr,
           SS:tl.constexpr,CS:tl.constexpr,F:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,F);b=row//L
    x=tl.load(X+row*D+i,i<D,0).to(tl.float32)
    mean=tl.sum(x,0)/D
    z=tl.where(i<D,x-mean,0.)
    var=tl.sum(z*z,0)/D
    norm=(x-mean)*tl.rsqrt(var+EPS)
    # Eager LayerNorm materializes in input dtype before modulation.
    norm=norm.to(X.dtype.element_ty).to(tl.float32)
    sh=tl.load(S+b*SS+i,i<D,0).to(tl.float32)
    sc=tl.load(C+b*CS+i,i<D,0).to(tl.float32)
    one=(1.+sc).to(C.dtype.element_ty).to(tl.float32)
    prod=(norm*one).to(X.dtype.element_ty).to(tl.float32)
    y=prod+sh
    tl.store(Y+row*D+i,y,i<D)

def modulated_norm(x,shift,scale,norm,*,dtype=None,enabled=False):
    dtype=x.dtype if dtype is None else dtype
    supported=(enabled and not torch.is_grad_enabled() and not norm.training
               and x.is_cuda and x.is_contiguous() and x.ndim==3
               and x.dtype==shift.dtype==scale.dtype==torch.float32
               and norm.weight is None and norm.bias is None and x.shape[-1]<=4096
               and shift.stride(-1)==scale.stride(-1)==1)
    if not supported:return (norm(x)*(1.+scale[:,None])+shift[:,None]).to(dtype)
    b,l,d=x.shape;y=torch.empty(x.shape,device=x.device,dtype=dtype)
    _lnmod[(b*l,)](x,shift,scale,y,l,d,norm.eps,shift.stride(0),scale.stride(0),triton.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
    return y
