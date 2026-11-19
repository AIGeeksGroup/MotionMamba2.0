'Angle step.'
from typing import Tuple
from typing import Optional
import torch
from torch import Tensor
import triton
import triton.language as tl
from motion_mamba2.kernels.activation_functions import tanh_approx
from motion_mamba2.kernels.activation_functions import sech2_approx

@triton.autotune(configs=[triton.Config({}, num_stages=s, num_warps=w) for s in [1, 2, 3] for w in [2, 4, 8]], key=['CHUNK_SIZE', 'BLOCK_D', 'HAS_INIT_STATE', 'RETURN_OUTPUT_STATE', 'IS_VARLEN'])
@triton.jit
def angle_dt_fwd_kernel(OUT, OUTPUT_STATE, ANGLE, DT, INIT_STATE, CU_SEQLENS, stride_out_batch, stride_out_seq, stride_out_head, stride_out_dim, stride_output_state_seq, stride_output_state_head, stride_output_state_dim, stride_angle_batch, stride_angle_seq, stride_angle_head, stride_angle_dim, stride_dt_batch, stride_dt_head, stride_dt_seq, stride_init_seq, stride_init_head, stride_init_dim, stride_cu_seqlen, seqlen, dim, CHUNK_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, HAS_INIT_STATE: tl.constexpr, RETURN_OUTPUT_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    if IS_VARLEN:
        pid_seq = tl.program_id(2)
        seq_idx = pid_seq
        cu_seqlen_start = tl.load(CU_SEQLENS + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_end = tl.load(CU_SEQLENS + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seq_len = cu_seqlen_end - cu_seqlen_start
        seq_offset = cu_seqlen_start
    else:
        seq_idx = pid_b
        seq_len = seqlen
        seq_offset = 0
    nchunks = tl.cdiv(seq_len, CHUNK_SIZE)
    ANGLE += pid_b * stride_angle_batch + pid_h * stride_angle_head + seq_offset * stride_angle_seq
    DT += pid_b * stride_dt_batch + pid_h * stride_dt_head + seq_offset * stride_dt_seq
    OUT += pid_b * stride_out_batch + pid_h * stride_out_head + seq_offset * stride_out_seq
    dim_range = tl.arange(0, BLOCK_D)
    dim_mask = dim_range < dim
    if HAS_INIT_STATE:
        init_ptrs = INIT_STATE + seq_idx * stride_init_seq + pid_h * stride_init_head + dim_range * stride_init_dim
        state = tl.load(init_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)
    PI = 3.141592653589793
    TWO_PI = 2 * PI
    for chunk_idx in range(nchunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        seq_range = tl.arange(0, CHUNK_SIZE)
        seq_mask = chunk_start + seq_range < seq_len
        angle_ptrs = ANGLE + (chunk_start + seq_range[:, None]) * stride_angle_seq + dim_range[None, :] * stride_angle_dim
        angle_vals = tl.load(angle_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        angle_vals = tanh_approx(angle_vals) * PI
        dt_ptrs = DT + (chunk_start + seq_range) * stride_dt_seq
        dt_vals = tl.load(dt_ptrs, mask=seq_mask, other=0.0).to(tl.float32)
        vals = angle_vals * dt_vals[:, None]
        chunk_cumsum = tl.cumsum(vals, axis=0)
        out_vals = chunk_cumsum + state[None, :]
        out_vals = out_vals - TWO_PI * tl.floor(out_vals / TWO_PI)
        out_ptrs = OUT + (chunk_start + seq_range[:, None]) * stride_out_seq + dim_range[None, :] * stride_out_dim
        tl.store(out_ptrs, out_vals, mask=seq_mask[:, None] & dim_mask[None, :])
        chunk_sum = tl.sum(vals, axis=0)
        state = state + chunk_sum
        state = state - TWO_PI * tl.floor(state / TWO_PI)
    if RETURN_OUTPUT_STATE:
        output_state_ptrs = OUTPUT_STATE + seq_idx * stride_output_state_seq + pid_h * stride_output_state_head + dim_range * stride_output_state_dim
        tl.store(output_state_ptrs, state, mask=dim_mask)

def angle_dt_fwd(angle: Tensor, dt: Tensor, init_state: Optional[Tensor]=None, chunk_size: int=64, return_output_state: bool=False, cu_seqlens: Optional[Tensor]=None) -> Tensor | Tuple[Tensor, Tensor]:
    (batch, seqlen, nheads, dim) = angle.shape
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert batch == 1, 'Varlen mode requires batch=1'
        num_sequences = cu_seqlens.shape[0] - 1
    else:
        num_sequences = batch
    assert dt.shape == (batch, nheads, seqlen), f'dt shape mismatch: {dt.shape}'
    if init_state is not None:
        assert init_state.shape == (num_sequences, nheads, dim), f'init_state shape mismatch: {init_state.shape}'
    out = torch.empty_like(angle)
    BLOCK_D = triton.next_power_of_2(dim)
    HAS_INIT_STATE = init_state is not None
    if not HAS_INIT_STATE:
        init_state = angle
        stride_init = (0, 0, 0)
    else:
        stride_init = init_state.stride()
    if return_output_state:
        output_state = torch.empty(num_sequences, nheads, dim, device=angle.device, dtype=angle.dtype)
        stride_output_state = output_state.stride()
    else:
        output_state = out
        stride_output_state = (0, 0, 0)
    if cu_seqlens is not None:
        stride_cu_seqlen = cu_seqlens.stride(0)
    else:
        cu_seqlens = angle
        stride_cu_seqlen = 0
    if is_varlen:
        grid = (nheads, batch, num_sequences)
    else:
        grid = (nheads, batch)
    angle_dt_fwd_kernel[grid](out, output_state, angle, dt, init_state, cu_seqlens, out.stride(0), out.stride(1), out.stride(2), out.stride(3), stride_output_state[0], stride_output_state[1], stride_output_state[2], angle.stride(0), angle.stride(1), angle.stride(2), angle.stride(3), dt.stride(0), dt.stride(1), dt.stride(2), stride_init[0], stride_init[1], stride_init[2], stride_cu_seqlen, seqlen, dim, CHUNK_SIZE=chunk_size, BLOCK_D=BLOCK_D, HAS_INIT_STATE=HAS_INIT_STATE, RETURN_OUTPUT_STATE=return_output_state, IS_VARLEN=is_varlen)
    if return_output_state:
        return (out, output_state)
    return out

@triton.autotune(configs=[triton.Config({}, num_stages=s, num_warps=w) for s in [1, 2, 3] for w in [2, 4, 8]], key=['CHUNK_SIZE', 'BLOCK_D', 'HAS_INIT_STATE', 'HAS_GRAD_OUTPUT_STATE', 'IS_VARLEN'])
@triton.jit
def angle_dt_bwd_kernel(GRAD_ANGLE, GRAD_DT, GRAD_INIT_STATE, GRAD_OUT, GRAD_OUTPUT_STATE, ANGLE, DT, CU_SEQLENS, stride_grad_angle_batch, stride_grad_angle_seq, stride_grad_angle_head, stride_grad_angle_dim, stride_grad_dt_batch, stride_grad_dt_head, stride_grad_dt_seq, stride_grad_init_seq, stride_grad_init_head, stride_grad_init_dim, stride_grad_out_batch, stride_grad_out_seq, stride_grad_out_head, stride_grad_out_dim, stride_grad_output_state_seq, stride_grad_output_state_head, stride_grad_output_state_dim, stride_angle_batch, stride_angle_seq, stride_angle_head, stride_angle_dim, stride_dt_batch, stride_dt_head, stride_dt_seq, stride_cu_seqlen, seqlen, dim, CHUNK_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, HAS_INIT_STATE: tl.constexpr, HAS_GRAD_OUTPUT_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    if IS_VARLEN:
        pid_seq = tl.program_id(2)
        seq_idx = pid_seq
        cu_seqlen_start = tl.load(CU_SEQLENS + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_end = tl.load(CU_SEQLENS + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seq_len = cu_seqlen_end - cu_seqlen_start
        seq_offset = cu_seqlen_start
    else:
        seq_idx = pid_b
        seq_len = seqlen
        seq_offset = 0
    nchunks = tl.cdiv(seq_len, CHUNK_SIZE)
    GRAD_ANGLE += pid_b * stride_grad_angle_batch + pid_h * stride_grad_angle_head + seq_offset * stride_grad_angle_seq
    GRAD_DT += pid_b * stride_grad_dt_batch + pid_h * stride_grad_dt_head + seq_offset * stride_grad_dt_seq
    GRAD_OUT += pid_b * stride_grad_out_batch + pid_h * stride_grad_out_head + seq_offset * stride_grad_out_seq
    ANGLE += pid_b * stride_angle_batch + pid_h * stride_angle_head + seq_offset * stride_angle_seq
    DT += pid_b * stride_dt_batch + pid_h * stride_dt_head + seq_offset * stride_dt_seq
    dim_range = tl.arange(0, BLOCK_D)
    dim_mask = dim_range < dim
    PI = 3.141592653589793
    if HAS_GRAD_OUTPUT_STATE:
        grad_output_state_ptrs = GRAD_OUTPUT_STATE + seq_idx * stride_grad_output_state_seq + pid_h * stride_grad_output_state_head + dim_range * stride_grad_output_state_dim
        grad_state = tl.load(grad_output_state_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
    else:
        grad_state = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for chunk_idx in range(nchunks - 1, -1, -1):
        chunk_start = chunk_idx * CHUNK_SIZE
        seq_range = tl.arange(0, CHUNK_SIZE)
        seq_mask = chunk_start + seq_range < seq_len
        grad_out_ptrs = GRAD_OUT + (chunk_start + seq_range[:, None]) * stride_grad_out_seq + dim_range[None, :] * stride_grad_out_dim
        grad_out_vals = tl.load(grad_out_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        chunk_sum = tl.sum(grad_out_vals, axis=0)
        fwd_cumsum = tl.cumsum(grad_out_vals, axis=0)
        rev_cumsum = chunk_sum[None, :] - fwd_cumsum + grad_out_vals
        grad_vals = rev_cumsum + grad_state[None, :]
        angle_ptrs = ANGLE + (chunk_start + seq_range[:, None]) * stride_angle_seq + dim_range[None, :] * stride_angle_dim
        pretanh_angle_vals = tl.load(angle_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        angle_vals = tanh_approx(pretanh_angle_vals) * PI
        dt_ptrs = DT + (chunk_start + seq_range) * stride_dt_seq
        dt_vals = tl.load(dt_ptrs, mask=seq_mask, other=0.0).to(tl.float32)
        grad_angle_vals = grad_vals * dt_vals[:, None] * PI * sech2_approx(pretanh_angle_vals)
        grad_dt_vals = tl.sum(grad_vals * angle_vals, axis=1)
        grad_angle_ptrs = GRAD_ANGLE + (chunk_start + seq_range[:, None]) * stride_grad_angle_seq + dim_range[None, :] * stride_grad_angle_dim
        tl.store(grad_angle_ptrs, grad_angle_vals, mask=seq_mask[:, None] & dim_mask[None, :])
        grad_dt_ptrs = GRAD_DT + (chunk_start + seq_range) * stride_grad_dt_seq
        tl.store(grad_dt_ptrs, grad_dt_vals, mask=seq_mask)
        grad_state = grad_state + chunk_sum
    if HAS_INIT_STATE:
        grad_init_ptrs = GRAD_INIT_STATE + seq_idx * stride_grad_init_seq + pid_h * stride_grad_init_head + dim_range * stride_grad_init_dim
        tl.store(grad_init_ptrs, grad_state, mask=dim_mask)

def angle_dt_bwd(grad_out: Tensor, angle: Tensor, dt: Tensor, has_init_state: bool=False, chunk_size: int=64, grad_output_state: Optional[Tensor]=None, cu_seqlens: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    (batch, seqlen, nheads, dim) = angle.shape
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert batch == 1, 'Varlen mode requires batch=1'
        num_sequences = cu_seqlens.shape[0] - 1
    else:
        num_sequences = batch
    grad_angle = torch.empty_like(angle)
    grad_dt = torch.empty_like(dt)
    BLOCK_D = triton.next_power_of_2(dim)
    if has_init_state:
        grad_init_state = torch.empty(num_sequences, nheads, dim, device=angle.device, dtype=torch.float32)
        stride_grad_init = grad_init_state.stride()
    else:
        grad_init_state = None
        stride_grad_init = (0, 0, 0)
        grad_init_dummy = grad_angle
    HAS_GRAD_OUTPUT_STATE = grad_output_state is not None
    if not HAS_GRAD_OUTPUT_STATE:
        grad_output_state = grad_angle
        stride_grad_output_state = (0, 0, 0)
    else:
        stride_grad_output_state = grad_output_state.stride()
    if cu_seqlens is not None:
        stride_cu_seqlen = cu_seqlens.stride(0)
    else:
        cu_seqlens = angle
        stride_cu_seqlen = 0
    if is_varlen:
        grid = (nheads, batch, num_sequences)
    else:
        grid = (nheads, batch)
    angle_dt_bwd_kernel[grid](grad_angle, grad_dt, grad_init_state if has_init_state else grad_init_dummy, grad_out, grad_output_state, angle, dt, cu_seqlens, grad_angle.stride(0), grad_angle.stride(1), grad_angle.stride(2), grad_angle.stride(3), grad_dt.stride(0), grad_dt.stride(1), grad_dt.stride(2), stride_grad_init[0], stride_grad_init[1], stride_grad_init[2], grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3), stride_grad_output_state[0], stride_grad_output_state[1], stride_grad_output_state[2], angle.stride(0), angle.stride(1), angle.stride(2), angle.stride(3), dt.stride(0), dt.stride(1), dt.stride(2), stride_cu_seqlen, seqlen, dim, CHUNK_SIZE=chunk_size, BLOCK_D=BLOCK_D, HAS_INIT_STATE=has_init_state, HAS_GRAD_OUTPUT_STATE=HAS_GRAD_OUTPUT_STATE, IS_VARLEN=is_varlen)
    return (grad_angle, grad_dt, grad_init_state)
