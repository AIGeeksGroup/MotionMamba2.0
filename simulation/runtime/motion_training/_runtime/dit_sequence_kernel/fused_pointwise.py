"""Experimental inference-only fusion; retain PyTorch LayerNorm and FP32 rounding."""
import torch
import triton
import triton.language as tl

@triton.jit
def _mod(X, S, C, Y, N:tl.constexpr, L:tl.constexpr, D:tl.constexpr,
         SS:tl.constexpr, CS:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = i // (L * D)
    d = i % D
    x = tl.load(X+i, i<N, 0)
    s = tl.load(S+b*SS+d, i<N, 0)
    c = tl.load(C+b*CS+d, i<N, 0)
    one = 1.0 + c
    prod = x * one
    tl.store(Y+i, prod+s, i<N)

@triton.jit
def _res(X, G, M, Y, N:tl.constexpr, L:tl.constexpr, D:tl.constexpr,
         GS:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = i // (L * D)
    d = i % D
    x = tl.load(X+i, i<N, 0)
    g = tl.load(G+b*GS+d, i<N, 0)
    m = tl.load(M+i, i<N, 0)
    prod = g * m
    tl.store(Y+i, x+prod, i<N)

def supported(x, *params):
    return (not torch.is_grad_enabled() and x.is_cuda and x.ndim==3
            and x.is_contiguous() and x.dtype==torch.float32
            and all(p.device==x.device and p.dtype==torch.float32 and p.ndim==2
                    and p.shape==(x.shape[0],x.shape[2]) and p.stride(-1)==1 for p in params))

def modulated_norm(x, shift, scale, norm, *, enabled=False, pointwise=False):
    if not pointwise or norm.training or not supported(x, shift, scale):
        from .fused_adaln import modulated_norm as original
        return original(x, shift, scale, norm, enabled=enabled)
    z=norm(x)
    y=torch.empty_like(z)
    _mod[(triton.cdiv(z.numel(),256),)](z,shift,scale,y,z.numel(),z.shape[1],z.shape[2],
        shift.stride(0),scale.stride(0),256,num_warps=4,enable_fp_fusion=False)
    return y

def residual(x, gate, mixed, *, enabled=False):
    if not (enabled and supported(x,gate) and mixed.dtype==x.dtype
            and mixed.device==x.device and mixed.shape==x.shape and mixed.is_contiguous()):
        return x+gate[:,None]*mixed
    y=torch.empty_like(x)
    _res[(triton.cdiv(x.numel(),256),)](x,gate,mixed,y,x.numel(),x.shape[1],x.shape[2],
        gate.stride(0),256,num_warps=4,enable_fp_fusion=False)
    return y
