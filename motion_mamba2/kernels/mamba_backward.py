# Copyright (c) 2026, Dao AI Lab, Goombalab
'Mamba backward.'
from typing import Optional
from typing import Tuple
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
import triton
import triton.language as tl
from motion_mamba2.kernels.activation_functions import cos_approx
from motion_mamba2.kernels.activation_functions import sin_approx
from motion_mamba2.kernels.activation_functions import sigmoid_approx

@triton.autotune(configs=[triton.Config({'CHUNK_SIZE': cs}, num_stages=s, num_warps=w, maxnreg=r) for cs in [32, 64] for s in [1, 2, 3] for w in [2, 4, 8] for r in [None, 128, 256]], key=['HEADDIM_V'])
@triton.jit
def mamba3_siso_bwd_kernel_dzdo(DO, Z, O, Dz, DO_scaled, stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_vdim, stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_vdim, stride_o_batch, stride_o_seqlen, stride_o_head, stride_o_vdim, stride_dz_batch, stride_dz_seqlen, stride_dz_head, stride_dz_vdim, stride_do_scaled_batch, stride_do_scaled_seqlen, stride_do_scaled_head, stride_do_scaled_vdim, seqlen, headdim_v, CHUNK_SIZE: tl.constexpr, HEADDIM_V: tl.constexpr):
    pid_chunk = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_batch = tl.program_id(2)
    do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head
    z_offset = pid_batch * stride_z_batch + pid_head * stride_z_head
    o_offset = pid_batch * stride_o_batch + pid_head * stride_o_head
    dz_offset = pid_batch * stride_dz_batch + pid_head * stride_dz_head
    do_scaled_offset = pid_batch * stride_do_scaled_batch + pid_head * stride_do_scaled_head
    chunk_start = pid_chunk * CHUNK_SIZE
    offs_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
    offs_dim = tl.arange(0, HEADDIM_V)
    mask = (offs_seq[:, None] < seqlen) & (offs_dim[None, :] < HEADDIM_V)
    do_ptrs = DO + do_offset + offs_seq[:, None] * stride_do_seqlen + offs_dim[None, :] * stride_do_vdim
    do_block = tl.load(do_ptrs, mask=mask, other=0.0)
    z_ptrs = Z + z_offset + offs_seq[:, None] * stride_z_seqlen + offs_dim[None, :] * stride_z_vdim
    z_block = tl.load(z_ptrs, mask=mask, other=0.0)
    o_ptrs = O + o_offset + offs_seq[:, None] * stride_o_seqlen + offs_dim[None, :] * stride_o_vdim
    o_block = tl.load(o_ptrs, mask=mask, other=0.0)
    sigmoid_z = tl.sigmoid(z_block.to(tl.float32))
    do_block = do_block * sigmoid_z
    dz_block = do_block * o_block * (1 + z_block * (1 - sigmoid_z))
    dz_ptrs = Dz + dz_offset + offs_seq[:, None] * stride_dz_seqlen + offs_dim[None, :] * stride_dz_vdim
    tl.store(dz_ptrs, dz_block, mask=mask)
    do_block = do_block * z_block
    do_scaled_ptrs = DO_scaled + do_scaled_offset + offs_seq[:, None] * stride_do_scaled_seqlen + offs_dim[None, :] * stride_do_scaled_vdim
    tl.store(do_scaled_ptrs, do_block, mask=mask)

def compute_dzdo(do: torch.Tensor, z: torch.Tensor, o: torch.Tensor, chunk_size: int=64) -> Tuple[torch.Tensor, torch.Tensor]:
    (batch, seqlen, nheads, headdim_v) = do.shape
    assert z is not None and o is not None and (do is not None), 'Z, O, and DO tensors must be provided'
    assert z.is_cuda and o.is_cuda and do.is_cuda, 'All tensors must be on CUDA'
    assert z.shape == do.shape and o.shape == do.shape, f'Shape mismatch: Z={z.shape}, O={o.shape}, DO={do.shape}'
    if do.stride(-1) != 1:
        do = do.contiguous()
    if z.stride(-1) != 1:
        z = z.contiguous()
    if o.stride(-1) != 1:
        o = o.contiguous()
    dz = torch.empty_like(z, dtype=do.dtype)
    do_scaled = torch.empty_like(do, dtype=do.dtype)
    HEADDIM_V = triton.next_power_of_2(headdim_v)

    def grid(META):
        return (triton.cdiv(seqlen, META['CHUNK_SIZE']), nheads, batch)
    mamba3_siso_bwd_kernel_dzdo[grid](do, z, o, dz, do_scaled, do.stride(0), do.stride(1), do.stride(2), do.stride(3), z.stride(0), z.stride(1), z.stride(2), z.stride(3), o.stride(0), o.stride(1), o.stride(2), o.stride(3), dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3), do_scaled.stride(0), do_scaled.stride(1), do_scaled.stride(2), do_scaled.stride(3), seqlen, headdim_v, HEADDIM_V=HEADDIM_V)
    return (dz, do_scaled)

@triton.autotune(configs=[triton.Config({}, num_stages=s, num_warps=w, maxnreg=r) for s in [1, 2, 3] for w in [2, 4, 8] for r in [None, 128, 256]], key=['CHUNK_SIZE', 'HEADDIM_QK', 'HEADDIM_V', 'IS_VARLEN'])
@triton.jit
def mamba3_siso_bwd_kernel_dqkv(Q, K, V, DA_CS, DA_CS_SUM, QK_Dot, D, SSM_States, dO, d_OSSM_State, Cu_Seqlens, dQ, dK, dV, dADT, dQK_Dot, dD, d_ISSM_State, stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_qkdim, stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim, stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_vdim, stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seqlen, stride_da_cs_sum_batch, stride_da_cs_sum_head, stride_da_cs_sum_seqlen, stride_qk_dot_batch, stride_qk_dot_head, stride_qk_dot_seqlen, stride_d_head, stride_ssm_states_batch, stride_ssm_states_head, stride_ssm_states_vdim, stride_ssm_states_qkdim, stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_vdim, stride_d_ossm_state_batch, stride_d_ossm_state_head, stride_d_ossm_state_vdim, stride_d_ossm_state_qkdim, stride_cu_seqlen, stride_dq_batch, stride_dq_seqlen, stride_dq_head, stride_dq_qkdim, stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_qkdim, stride_dv_batch, stride_dv_seqlen, stride_dv_head, stride_dv_vdim, stride_dadt_batch, stride_dadt_head, stride_dadt_seqlen, stride_dQK_dot_batch, stride_dQK_dot_head, stride_dQK_dot_seqlen, stride_dd_batch, stride_dd_head, stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_vdim, stride_d_issm_state_qkdim, seqlen, nheads_qk, headdim_qk, headdim_v, CHUNK_SIZE: tl.constexpr, HEADDIM_QK: tl.constexpr, HEADDIM_V: tl.constexpr, RECOMPUTE_MASK: tl.constexpr, HAS_D_OSSM_STATE: tl.constexpr, RETURN_D_ISSM_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    if IS_VARLEN:
        pid_seq = pid_batch
        pid_batch = 0
        cu_seqlen = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_next = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seqlen = cu_seqlen_next - cu_seqlen
        cu_chunks = pid_seq + cu_seqlen // CHUNK_SIZE
    else:
        cu_seqlen = 0
        cu_chunks = 0
        pid_seq = 0
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)
    q_offset = pid_batch * stride_q_batch + head_idx_qk * stride_q_head + IS_VARLEN * cu_seqlen * stride_q_seqlen
    k_offset = pid_batch * stride_k_batch + head_idx_qk * stride_k_head + IS_VARLEN * cu_seqlen * stride_k_seqlen
    v_offset = pid_batch * stride_v_batch + pid_head * stride_v_head + IS_VARLEN * cu_seqlen * stride_v_seqlen
    da_cs_offset = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head + IS_VARLEN * cu_seqlen * stride_da_cs_seqlen
    da_cs_sum_offset = pid_batch * stride_da_cs_sum_batch + pid_head * stride_da_cs_sum_head + IS_VARLEN * cu_chunks * stride_da_cs_sum_seqlen
    qk_dot_offset = pid_batch * stride_qk_dot_batch + pid_head * stride_qk_dot_head + IS_VARLEN * cu_seqlen * stride_qk_dot_seqlen
    ssm_states_offset = pid_batch * stride_ssm_states_batch + pid_head * stride_ssm_states_head + IS_VARLEN * cu_chunks * HEADDIM_QK * stride_ssm_states_qkdim
    do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head + IS_VARLEN * cu_seqlen * stride_do_seqlen
    if HAS_D_OSSM_STATE:
        d_ossm_state_offset = (pid_batch + IS_VARLEN * pid_seq) * stride_d_ossm_state_batch + pid_head * stride_d_ossm_state_head
    if D is not None:
        D_offset = pid_head * stride_d_head
        D_val = tl.load(D + D_offset)
    dq_offset = pid_batch * stride_dq_batch + pid_head * stride_dq_head + IS_VARLEN * cu_seqlen * stride_dq_seqlen
    dk_offset = pid_batch * stride_dk_batch + pid_head * stride_dk_head + IS_VARLEN * cu_seqlen * stride_dk_seqlen
    dv_offset = pid_batch * stride_dv_batch + pid_head * stride_dv_head + IS_VARLEN * cu_seqlen * stride_dv_seqlen
    dadt_offset = pid_batch * stride_dadt_batch + pid_head * stride_dadt_head + IS_VARLEN * cu_seqlen * stride_dadt_seqlen
    dQK_dot_offset = pid_batch * stride_dQK_dot_batch + pid_head * stride_dQK_dot_head + IS_VARLEN * cu_seqlen * stride_dQK_dot_seqlen
    if D is not None:
        dD_offset = pid_head * stride_dd_head + pid_batch * stride_dd_batch + IS_VARLEN * pid_seq * stride_dd_batch
        dD_acc = tl.zeros([1], dtype=tl.float32)
    if RETURN_D_ISSM_STATE:
        d_issm_state_offset = (pid_batch + IS_VARLEN * pid_seq) * stride_d_issm_state_batch + pid_head * stride_d_issm_state_head
    if HAS_D_OSSM_STATE:
        d_ssm_ptrs = d_OSSM_State + d_ossm_state_offset + tl.arange(0, HEADDIM_V)[:, None] * stride_d_ossm_state_vdim + tl.arange(0, HEADDIM_QK)[None, :] * stride_d_ossm_state_qkdim
        d_ssm_states_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk)
        d_ssm_states_acc = tl.load(d_ssm_ptrs, mask=d_ssm_states_mask, other=0.0).to(tl.float32)
    else:
        d_ssm_states_acc = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    q_desc = tl.make_tensor_descriptor(Q + q_offset, shape=[seqlen, headdim_qk], strides=[stride_q_seqlen, stride_q_qkdim], block_shape=[CHUNK_SIZE, HEADDIM_QK])
    k_desc = tl.make_tensor_descriptor(K + k_offset, shape=[seqlen, headdim_qk], strides=[stride_k_seqlen, stride_k_qkdim], block_shape=[CHUNK_SIZE, HEADDIM_QK])
    v_desc = tl.make_tensor_descriptor(V + v_offset, shape=[seqlen, headdim_v], strides=[stride_v_seqlen, stride_v_vdim], block_shape=[CHUNK_SIZE, HEADDIM_V])
    ssm_states_desc = tl.make_tensor_descriptor(SSM_States + ssm_states_offset, shape=[headdim_v, num_chunks * headdim_qk], strides=[stride_ssm_states_vdim, stride_ssm_states_qkdim], block_shape=[HEADDIM_V, HEADDIM_QK])
    do_desc = tl.make_tensor_descriptor(dO + do_offset, shape=[seqlen, headdim_v], strides=[stride_do_seqlen, stride_do_vdim], block_shape=[CHUNK_SIZE, HEADDIM_V])
    dq_desc = tl.make_tensor_descriptor(dQ + dq_offset, shape=[seqlen, headdim_qk], strides=[stride_dq_seqlen, stride_dq_qkdim], block_shape=[CHUNK_SIZE, HEADDIM_QK])
    dk_desc = tl.make_tensor_descriptor(dK + dk_offset, shape=[seqlen, headdim_qk], strides=[stride_dk_seqlen, stride_dk_qkdim], block_shape=[CHUNK_SIZE, HEADDIM_QK])
    dv_desc = tl.make_tensor_descriptor(dV + dv_offset, shape=[seqlen, headdim_v], strides=[stride_dv_seqlen, stride_dv_vdim], block_shape=[CHUNK_SIZE, HEADDIM_V])
    for chunk_idx_loop in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_idx_loop
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_cs = chunk_start + tl.arange(0, CHUNK_SIZE)
        seq_mask = offs_cs < seqlen
        da_cs_ptrs = DA_CS + da_cs_offset + offs_cs * stride_da_cs_seqlen
        da_cs = tl.load(da_cs_ptrs, mask=seq_mask, other=0.0)
        da_cs_sum_ptrs = DA_CS_SUM + da_cs_sum_offset + chunk_idx * stride_da_cs_sum_seqlen
        da_cs_chunk_sum = tl.load(da_cs_sum_ptrs)
        do_block = do_desc.load([chunk_start, 0])
        v_block = v_desc.load([chunk_start, 0])
        q_block = q_desc.load([chunk_start, 0])
        k_block = k_desc.load([chunk_start, 0])
        ssm_states_block = ssm_states_desc.load([0, chunk_idx * headdim_qk])
        da_cs_rev = da_cs_chunk_sum - da_cs
        exp_da_cs_rev = tl.math.exp2(da_cs_rev)
        exp_da_cs = tl.math.exp2(da_cs)
        if not RECOMPUTE_MASK:
            causal_decay_mask = tl.where(tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None], tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0)), 0.0)
        dAinv = tl.dot(v_block, tl.trans(do_block))
        if RECOMPUTE_MASK:
            dAinv *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dAinv = tl.where(tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None], dAinv, 0.0)
        else:
            dAinv *= causal_decay_mask
        dAinv *= tl.dot(k_block, tl.trans(q_block))
        dM_rev_vector = tl.sum(dAinv, axis=0) - tl.sum(dAinv, axis=1)
        dp_t_block = tl.dot(v_block, tl.trans(do_block))
        if RECOMPUTE_MASK:
            dp_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dp_t_block = tl.where(tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None], dp_t_block, 0.0)
        else:
            dp_t_block *= causal_decay_mask
        acc_dk = tl.dot(dp_t_block.to(q_block.dtype), q_block)
        acc_dk += tl.dot(v_block, d_ssm_states_acc.to(v_block.dtype)) * exp_da_cs_rev[:, None]
        dk_desc.store([chunk_start, 0], acc_dk)
        s_block = tl.dot(v_block, tl.trans(do_block))
        if RECOMPUTE_MASK:
            s_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            s_block = tl.where(tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None], s_block, 0.0)
        else:
            s_block *= causal_decay_mask
        acc_dq = tl.dot(tl.trans(s_block).to(k_block.dtype), k_block)
        acc_dq += tl.dot(do_block, ssm_states_block) * exp_da_cs[:, None]
        dq_desc.store([chunk_start, 0], acc_dq)
        p_t_block = tl.dot(k_block, tl.trans(q_block))
        if RECOMPUTE_MASK:
            p_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            p_t_block = tl.where(tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None], p_t_block, 0.0)
        else:
            p_t_block *= causal_decay_mask
        acc_dv = tl.dot(p_t_block.to(do_block.dtype), do_block)
        acc_dv += tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype)) * exp_da_cs_rev[:, None]
        dO_reloaded = tl.load(dO + do_offset + offs_cs[:, None] * stride_do_seqlen + tl.arange(0, HEADDIM_V)[None, :] * stride_do_vdim, mask=seq_mask[:, None] & (tl.arange(0, HEADDIM_V)[None, :] < headdim_v), other=0.0, volatile=True)
        qk_dot = tl.load(QK_Dot + qk_dot_offset + offs_cs * stride_qk_dot_seqlen, mask=seq_mask, other=0.0)
        if D is not None:
            acc_dv += dO_reloaded * (D_val + qk_dot[:, None])
        else:
            acc_dv += dO_reloaded * qk_dot[:, None]
        dv_desc.store([chunk_start, 0], acc_dv)
        v_block_reloaded = tl.load(V + v_offset + offs_cs[:, None] * stride_v_seqlen + tl.arange(0, HEADDIM_V)[None, :] * stride_v_vdim, mask=seq_mask[:, None] & (tl.arange(0, HEADDIM_V)[None, :] < headdim_v), other=0.0, volatile=True)
        dQK_dot_block = tl.dot(dO_reloaded * v_block_reloaded, tl.full([HEADDIM_V, 1], 1, dtype=dO_reloaded.dtype))
        tl.store(dQK_Dot + dQK_dot_offset + offs_cs * stride_dQK_dot_seqlen, dQK_dot_block.reshape(CHUNK_SIZE), mask=seq_mask)
        if D is not None:
            dD_acc += tl.dot(tl.full([1, CHUNK_SIZE], 1, dtype=tl.float32), dQK_dot_block).reshape(1)
        QS = tl.dot(q_block, tl.trans(ssm_states_block))
        dM_rev_vector += tl.sum(QS * dO_reloaded, axis=1) * exp_da_cs
        SSM_States_ptrs = SSM_States + ssm_states_offset + tl.arange(0, HEADDIM_V)[:, None] * stride_ssm_states_vdim + (chunk_idx * headdim_qk + tl.arange(0, HEADDIM_QK)[None, :]) * stride_ssm_states_qkdim
        SSM_States_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (chunk_idx * headdim_qk + tl.arange(0, HEADDIM_QK)[None, :] < num_chunks * headdim_qk)
        SSM_States_reloaded = tl.load(SSM_States_ptrs, volatile=True, mask=SSM_States_mask)
        dM_scalar = tl.sum(SSM_States_reloaded * d_ssm_states_acc) * tl.math.exp2(da_cs_chunk_sum)
        dSK = tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype))
        dM_vector = tl.sum(dSK * v_block_reloaded, axis=1) * exp_da_cs_rev
        dM_rev_vector += tl.sum(dM_rev_vector) + dM_scalar + tl.cumsum(dM_vector - dM_rev_vector) - dM_vector
        dadt_ptrs = dADT + dadt_offset + offs_cs * stride_dadt_seqlen
        tl.store(dadt_ptrs, dM_rev_vector, mask=seq_mask)
        dO_reloaded *= exp_da_cs[:, None]
        d_ssm_states_acc = tl.math.exp2(da_cs_chunk_sum) * d_ssm_states_acc + tl.dot(tl.trans(dO_reloaded).to(q_block.dtype), q_block)
    if D is not None:
        tl.store(dD + dD_offset + tl.arange(0, 1), dD_acc)
    if RETURN_D_ISSM_STATE:
        d_ISSM_State_ptrs = d_ISSM_State + d_issm_state_offset + tl.arange(0, HEADDIM_V)[:, None] * stride_d_issm_state_vdim + tl.arange(0, HEADDIM_QK)[None, :] * stride_d_issm_state_qkdim
        d_ISSM_State_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk)
        tl.store(d_ISSM_State_ptrs, d_ssm_states_acc, mask=d_ISSM_State_mask)

def compute_dqkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, da_cs: torch.Tensor, da_cs_sum: torch.Tensor, qk_dot: torch.Tensor, SSM_States: torch.Tensor, do: torch.Tensor, d_ossm_state: Optional[torch.Tensor]=None, d_ov_state: Optional[torch.Tensor]=None, D: Optional[torch.Tensor]=None, chunk_size: int=64, has_input_state: bool=False, Cu_Seqlens: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    (batch, seqlen, nheads_qk, headdim_qk) = q.shape
    (_, _, nheads, headdim_v) = v.shape
    is_varlen = Cu_Seqlens is not None
    if is_varlen:
        num_sequences = Cu_Seqlens.shape[0] - 1
        assert batch == 1
        nchunks = num_sequences + seqlen // chunk_size
    else:
        num_sequences = batch
        nchunks = (seqlen + chunk_size - 1) // chunk_size
    assert nheads % nheads_qk == 0, 'nheads must be divisible by nheads_qk (for GQA support)'
    assert q.is_cuda and k.is_cuda and v.is_cuda and da_cs.is_cuda and da_cs_sum.is_cuda and do.is_cuda, 'All tensors must be on CUDA'
    assert k.shape == q.shape
    assert v.shape == (batch, seqlen, nheads, headdim_v)
    assert da_cs.shape == (batch, nheads, seqlen)
    assert da_cs_sum.shape == (batch, nheads, nchunks)
    assert qk_dot.shape == (batch, nheads, seqlen)
    assert SSM_States.shape == (batch, nheads, headdim_v, nchunks * headdim_qk)
    assert do.shape == (batch, seqlen, nheads, headdim_v)
    assert d_ossm_state is None or d_ossm_state.shape == (num_sequences, nheads, headdim_v, headdim_qk)
    assert d_ov_state is None or d_ov_state.shape == (num_sequences, nheads, headdim_v)
    if D is not None:
        assert D.shape == (nheads,)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if da_cs.stride(-1) != 1:
        da_cs = da_cs.contiguous()
    if da_cs_sum.stride(-1) != 1:
        da_cs_sum = da_cs_sum.contiguous()
    if qk_dot.stride(-1) != 1:
        qk_dot = qk_dot.contiguous()
    if not SSM_States.is_contiguous():
        SSM_States = SSM_States.contiguous()
    if not do.is_contiguous():
        do = do.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if d_ossm_state is not None and (not d_ossm_state.is_contiguous()):
        d_ossm_state = d_ossm_state.contiguous()
    if d_ov_state is not None and (not d_ov_state.is_contiguous()):
        d_ov_state = d_ov_state.contiguous()
    dq = torch.empty((batch, seqlen, nheads, headdim_qk), dtype=q.dtype, device=q.device)
    dk = torch.empty((batch, seqlen, nheads, headdim_qk), dtype=k.dtype, device=k.device)
    dv = torch.empty((batch, seqlen, nheads, headdim_v), dtype=v.dtype, device=v.device)
    dAdt = torch.empty_like(da_cs)
    dQK = torch.empty_like(da_cs)
    dD = torch.empty((num_sequences, nheads), dtype=torch.float32, device=q.device) if D is not None else None
    d_issm_state = torch.empty((num_sequences, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=q.device) if has_input_state else None
    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    HEADDIM_V = triton.next_power_of_2(headdim_v)
    if is_varlen:
        grid = (nheads, num_sequences)
    else:
        grid = (nheads, batch)
    mamba3_siso_bwd_kernel_dqkv[grid](q, k, v, da_cs, da_cs_sum, qk_dot, D, SSM_States, do, d_ossm_state, Cu_Seqlens, dq, dk, dv, dAdt, dQK, dD, d_issm_state, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), da_cs.stride(0), da_cs.stride(1), da_cs.stride(2), da_cs_sum.stride(0), da_cs_sum.stride(1), da_cs_sum.stride(2), qk_dot.stride(0), qk_dot.stride(1), qk_dot.stride(2), D.stride(0) if D is not None else 0, SSM_States.stride(0), SSM_States.stride(1), SSM_States.stride(2), SSM_States.stride(3), do.stride(0), do.stride(1), do.stride(2), do.stride(3), d_ossm_state.stride(0) if d_ossm_state is not None else 0, d_ossm_state.stride(1) if d_ossm_state is not None else 0, d_ossm_state.stride(2) if d_ossm_state is not None else 0, d_ossm_state.stride(3) if d_ossm_state is not None else 0, Cu_Seqlens.stride(0) if Cu_Seqlens is not None else 0, dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3), dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3), dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3), dAdt.stride(0), dAdt.stride(1), dAdt.stride(2), dQK.stride(0), dQK.stride(1), dQK.stride(2), dD.stride(0) if D is not None else 0, dD.stride(1) if D is not None else 0, d_issm_state.stride(0) if d_issm_state is not None else 0, d_issm_state.stride(1) if d_issm_state is not None else 0, d_issm_state.stride(2) if d_issm_state is not None else 0, d_issm_state.stride(3) if d_issm_state is not None else 0, seqlen, nheads_qk, headdim_qk, headdim_v, CHUNK_SIZE=chunk_size, HEADDIM_QK=HEADDIM_QK, HEADDIM_V=HEADDIM_V, RECOMPUTE_MASK=False, HAS_D_OSSM_STATE=d_ossm_state is not None, RETURN_D_ISSM_STATE=has_input_state, IS_VARLEN=is_varlen)
    if d_ov_state is not None:
        if is_varlen:
            last_token_idx = Cu_Seqlens[1:] - 1
            dv[0, last_token_idx] += d_ov_state
        else:
            dv[:, -1, :, :] += d_ov_state
    dD = dD.sum(dim=0) if dD is not None else None
    return (dq, dk, dv, dAdt, dQK, dD, d_issm_state)

@triton.autotune(configs=[triton.Config({}, num_stages=s, num_warps=w, maxnreg=r) for s in [1, 2, 3] for w in [2, 4, 8] for r in [None, 128, 256]], key=['CHUNK_SIZE', 'BLOCK_HEADDIM_QK', 'HEADDIM_QK', 'GQA_RATIO'])
@triton.jit
def mamba3_siso_bwd_kernel_rotary_bias_angles(Q, K, Scale, Gamma, Q_bias, K_bias, Angles, dQ_in, dK_in, dQK, dQ, dK, dAngles, dScale, dGamma, dQ_bias, dK_bias, stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_qkdim, stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim, stride_scale_batch, stride_scale_head, stride_scale_seqlen, stride_gamma_batch, stride_gamma_head, stride_gamma_seqlen, stride_q_bias_head, stride_q_bias_qkdim, stride_k_bias_head, stride_k_bias_qkdim, stride_angles_batch, stride_angles_seqlen, stride_angles_head, stride_angles_qkdim, stride_dq_in_batch, stride_dq_in_seqlen, stride_dq_in_head, stride_dq_in_qkdim, stride_dk_in_batch, stride_dk_in_seqlen, stride_dk_in_head, stride_dk_in_qkdim, stride_dqk_batch, stride_dqk_head, stride_dqk_seqlen, stride_dq_batch, stride_dq_seqlen, stride_dq_head, stride_dq_qkdim, stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_qkdim, stride_dangles_batch, stride_dangles_seqlen, stride_dangles_head, stride_dangles_qkdim, stride_dscale_batch, stride_dscale_head, stride_dscale_nqkchunks, stride_dscale_seqlen, stride_dgamma_batch, stride_dgamma_head, stride_dgamma_nqkchunks, stride_dgamma_seqlen, stride_dq_bias_batch, stride_dq_bias_nchunks, stride_dq_bias_head, stride_dq_bias_qkdim, stride_dk_bias_batch, stride_dk_bias_nchunks, stride_dk_bias_head, stride_dk_bias_qkdim, seqlen, nheads_qk, nheads, headdim_qk, headdim_angles, CHUNK_SIZE: tl.constexpr, HEADDIM_QK: tl.constexpr, BLOCK_HEADDIM_QK: tl.constexpr, GQA_RATIO: tl.constexpr):
    pid_nchunk = tl.program_id(0)
    pid_batch = tl.program_id(1)
    nchunks = tl.cdiv(seqlen, CHUNK_SIZE)
    q_offset_base = pid_batch * stride_q_batch
    k_offset_base = pid_batch * stride_k_batch
    scale_offset_base = pid_batch * stride_scale_batch
    gamma_offset_base = pid_batch * stride_gamma_batch
    angle_offset_base = pid_batch * stride_angles_batch
    dq_in_offset_base = pid_batch * stride_dq_in_batch
    dk_in_offset_base = pid_batch * stride_dk_in_batch
    dqk_offset_base = pid_batch * stride_dqk_batch
    dq_offset_base = pid_batch * stride_dq_batch
    dk_offset_base = pid_batch * stride_dk_batch
    dangle_offset_base = pid_batch * stride_dangles_batch
    dscale_offset_base = pid_batch * stride_dscale_batch
    dgamma_offset_base = pid_batch * stride_dgamma_batch
    dq_bias_offset_base = pid_batch * stride_dq_bias_batch + pid_nchunk * stride_dq_bias_nchunks
    dk_bias_offset_base = pid_batch * stride_dk_bias_batch + pid_nchunk * stride_dk_bias_nchunks
    num_nheads_qk = HEADDIM_QK // BLOCK_HEADDIM_QK
    for nhead_qk_id in range(num_nheads_qk):
        offs_s = tl.arange(0, CHUNK_SIZE) + pid_nchunk * CHUNK_SIZE
        offs_d = tl.arange(0, BLOCK_HEADDIM_QK) + nhead_qk_id * BLOCK_HEADDIM_QK
        offs_dr = tl.arange(0, BLOCK_HEADDIM_QK // 2) + nhead_qk_id * (BLOCK_HEADDIM_QK // 2)
        for qk_head_idx in range(nheads_qk):
            q_offset = q_offset_base + qk_head_idx * stride_q_head
            k_offset = k_offset_base + qk_head_idx * stride_k_head
            q_ptrs = Q + q_offset + offs_s[:, None] * stride_q_seqlen + offs_d[None, :] * stride_q_qkdim
            k_ptrs = K + k_offset + offs_s[:, None] * stride_k_seqlen + offs_d[None, :] * stride_k_qkdim
            dq_acc = tl.zeros((CHUNK_SIZE, BLOCK_HEADDIM_QK), dtype=tl.float32)
            dk_acc = tl.zeros((CHUNK_SIZE, BLOCK_HEADDIM_QK), dtype=tl.float32)
            for gqa_idx in range(GQA_RATIO):
                nhead_idx = qk_head_idx * GQA_RATIO + gqa_idx
                q_bias = tl.load(Q_bias + nhead_idx * stride_q_bias_head + offs_d * stride_q_bias_qkdim, mask=offs_d < headdim_qk).to(tl.float32)
                k_bias = tl.load(K_bias + nhead_idx * stride_k_bias_head + offs_d * stride_k_bias_qkdim, mask=offs_d < headdim_qk).to(tl.float32)
                q0 = tl.load(q_ptrs, mask=(offs_s[:, None] < seqlen) & (offs_d[None, :] < headdim_qk), other=0.0)
                k0 = tl.load(k_ptrs, mask=(offs_s[:, None] < seqlen) & (offs_d[None, :] < headdim_qk), other=0.0)
                Q_wbias = q0 + q_bias[None, :]
                K_wbias = k0 + k_bias[None, :]
                dqk_offset = dqk_offset_base + nhead_idx * stride_dqk_head
                dqk = tl.load(dQK + dqk_offset + offs_s * stride_dqk_seqlen, mask=offs_s < seqlen, other=0.0)
                scale_offset = scale_offset_base + nhead_idx * stride_scale_head
                gamma_offset = gamma_offset_base + nhead_idx * stride_gamma_head
                scale = tl.load(Scale + scale_offset + offs_s * stride_scale_seqlen, mask=offs_s < seqlen, other=0.0).to(tl.float32)
                gamma = tl.load(Gamma + gamma_offset + offs_s * stride_gamma_seqlen, mask=offs_s < seqlen, other=0.0).to(tl.float32)
                angle_offset = angle_offset_base + nhead_idx * stride_angles_head
                theta = tl.load(Angles + angle_offset + offs_s[:, None] * stride_angles_seqlen + offs_dr[None, :] * stride_angles_qkdim, mask=(offs_dr[None, :] < headdim_angles) & (offs_s[:, None] < seqlen), other=0.0).to(tl.float32)
                dq_in_offset = dq_in_offset_base + nhead_idx * stride_dq_in_head
                dk_in_offset = dk_in_offset_base + nhead_idx * stride_dk_in_head
                dQ_in_load = tl.load(dQ_in + dq_in_offset + offs_s[:, None] * stride_dq_in_seqlen + offs_d[None, :] * stride_dq_in_qkdim, mask=(offs_s[:, None] < seqlen) & (offs_d[None, :] < headdim_qk), other=0.0)
                dK_in_load = tl.load(dK_in + dk_in_offset + offs_s[:, None] * stride_dk_in_seqlen + offs_d[None, :] * stride_dk_in_qkdim, mask=(offs_s[:, None] < seqlen) & (offs_d[None, :] < headdim_qk), other=0.0)
                QK_dot = tl.sum(Q_wbias * K_wbias, axis=1)
                d_gamma = dqk * QK_dot
                dgamma_store_offset = dgamma_offset_base + nhead_idx * stride_dgamma_head
                tl.store(dGamma + dgamma_store_offset + offs_s * stride_dgamma_seqlen + nhead_qk_id * stride_dgamma_nqkchunks, d_gamma, mask=offs_s < seqlen)
                cos_angle = cos_approx(theta.to(tl.float32))
                sin_angle = sin_approx(theta.to(tl.float32))
                K_r = tl.reshape(K_wbias, [CHUNK_SIZE, BLOCK_HEADDIM_QK // 2, 2])
                (K_r0, K_r1) = tl.split(K_r)
                K_rot0 = K_r0 * cos_angle - K_r1 * sin_angle
                K_rot1 = K_r0 * sin_angle + K_r1 * cos_angle
                K_rot = tl.reshape(tl.join(K_rot0, K_rot1), [CHUNK_SIZE, BLOCK_HEADDIM_QK])
                dscale_val = tl.sum(dK_in_load * K_rot, axis=1)
                dscale_store_offset = dscale_offset_base + nhead_idx * stride_dscale_head
                tl.store(dScale + dscale_store_offset + offs_s * stride_dscale_seqlen + nhead_qk_id * stride_dscale_nqkchunks, dscale_val, mask=offs_s < seqlen)
                dK_in_scaled = dK_in_load * scale[:, None]
                Q_r = tl.reshape(Q_wbias, [CHUNK_SIZE, BLOCK_HEADDIM_QK // 2, 2])
                (Q_r0, Q_r1) = tl.split(Q_r)
                dQ_in_r = tl.reshape(dQ_in_load, [CHUNK_SIZE, BLOCK_HEADDIM_QK // 2, 2])
                dK_in_r = tl.reshape(dK_in_scaled, [CHUNK_SIZE, BLOCK_HEADDIM_QK // 2, 2])
                (dQ_in_r0, dQ_in_r1) = tl.split(dQ_in_r)
                (dK_in_r0, dK_in_r1) = tl.split(dK_in_r)
                dq0 = dQ_in_r0 * cos_angle + dQ_in_r1 * sin_angle
                dq1 = -dQ_in_r0 * sin_angle + dQ_in_r1 * cos_angle
                dk0 = dK_in_r0 * cos_angle + dK_in_r1 * sin_angle
                dk1 = -dK_in_r0 * sin_angle + dK_in_r1 * cos_angle
                dQ_pre = tl.reshape(tl.join(dq0, dq1), [CHUNK_SIZE, BLOCK_HEADDIM_QK])
                dK_pre = tl.reshape(tl.join(dk0, dk1), [CHUNK_SIZE, BLOCK_HEADDIM_QK])
                dqk_scaled = (dqk * gamma)[:, None]
                dQ_pre = dQ_pre + dqk_scaled * K_wbias
                dK_pre = dK_pre + dqk_scaled * Q_wbias
                dq_acc += dQ_pre
                dk_acc += dK_pre
                dq_bias_out = tl.sum(dQ_pre, axis=0)
                dk_bias_out = tl.sum(dK_pre, axis=0)
                dq_bias_store_offset = dq_bias_offset_base + nhead_idx * stride_dq_bias_head
                dk_bias_store_offset = dk_bias_offset_base + nhead_idx * stride_dk_bias_head
                tl.store(dQ_bias + dq_bias_store_offset + offs_d * stride_dq_bias_qkdim, dq_bias_out, mask=offs_d < headdim_qk)
                tl.store(dK_bias + dk_bias_store_offset + offs_d * stride_dk_bias_qkdim, dk_bias_out, mask=offs_d < headdim_qk)
                dtheta_q = dQ_in_r0 * (-Q_r0 * sin_angle - Q_r1 * cos_angle) + dQ_in_r1 * (Q_r0 * cos_angle - Q_r1 * sin_angle)
                dtheta_k = dK_in_r0 * (-K_r0 * sin_angle - K_r1 * cos_angle) + dK_in_r1 * (K_r0 * cos_angle - K_r1 * sin_angle)
                dtheta = dtheta_q + dtheta_k
                dangle_store_offset = dangle_offset_base + nhead_idx * stride_dangles_head
                tl.store(dAngles + dangle_store_offset + offs_s[:, None] * stride_dangles_seqlen + offs_dr[None, :] * stride_dangles_qkdim, dtheta, mask=(offs_dr[None, :] < headdim_angles) & (offs_s[:, None] < seqlen))
            dq_offset = dq_offset_base + qk_head_idx * stride_dq_head
            dk_offset = dk_offset_base + qk_head_idx * stride_dk_head
            dq_ptrs = dQ + dq_offset + offs_s[:, None] * stride_dq_seqlen + offs_d[None, :] * stride_dq_qkdim
            dk_ptrs = dK + dk_offset + offs_s[:, None] * stride_dk_seqlen + offs_d[None, :] * stride_dk_qkdim
            tl.store(dq_ptrs, dq_acc, mask=(offs_s[:, None] < seqlen) & (offs_d[None, :] < headdim_qk))
            tl.store(dk_ptrs, dk_acc, mask=(offs_s[:, None] < seqlen) & (offs_d[None, :] < headdim_qk))

@triton.jit
def mamba3_siso_bwd_kernel_dk_state_post(dK_State, Angles, K, K_bias, Cu_Seqlens, dK, dK_bias, dAngles, stride_dk_state_batch, stride_dk_state_head, stride_dk_state_qkdim, stride_angles_batch, stride_angles_seqlen, stride_angles_head, stride_angles_qkdim, stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim, stride_k_bias_head, stride_k_bias_qkdim, stride_cu_seqlen, stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_qkdim, stride_dk_bias_head, stride_dk_bias_qkdim, stride_dangles_batch, stride_dangles_seqlen, stride_dangles_head, stride_dangles_qkdim, seqlen, headdim_qk, headdim_angles, HEADDIM_QK: tl.constexpr, GQA_RATIO: tl.constexpr, IS_VARLEN: tl.constexpr):
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    if IS_VARLEN:
        pid_seq = pid_batch
        pid_batch = 0
        cu_seqlen = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        last_pos = cu_seqlen - 1
    else:
        pid_seq = 0
        last_pos = seqlen - 1
    qk_head_idx = pid_head // GQA_RATIO
    offs_d = tl.arange(0, HEADDIM_QK)
    offs_dr = tl.arange(0, HEADDIM_QK // 2)
    dk_state_base = dK_State + (pid_batch + pid_seq) * stride_dk_state_batch + pid_head * stride_dk_state_head
    dk_state = tl.load(dk_state_base + offs_d * stride_dk_state_qkdim, mask=offs_d < headdim_qk, other=0.0).to(tl.float32)
    dk_state_r = tl.reshape(dk_state, [HEADDIM_QK // 2, 2])
    (dk_state_r0, dk_state_r1) = tl.split(dk_state_r)
    angles_base = Angles + pid_batch * stride_angles_batch + last_pos * stride_angles_seqlen + pid_head * stride_angles_head
    angles_val = tl.load(angles_base + offs_dr * stride_angles_qkdim, mask=offs_dr < headdim_angles, other=0.0).to(tl.float32)
    cos_ang = cos_approx(angles_val)
    sin_ang = sin_approx(angles_val)
    dk0 = dk_state_r0 * cos_ang + dk_state_r1 * sin_ang
    dk1 = -dk_state_r0 * sin_ang + dk_state_r1 * cos_ang
    dk_rotated = tl.reshape(tl.join(dk0, dk1), [HEADDIM_QK])
    dk_base = dK + pid_batch * stride_dk_batch + last_pos * stride_dk_seqlen + qk_head_idx * stride_dk_head
    tl.atomic_add(dk_base + offs_d * stride_dk_qkdim, dk_rotated, mask=offs_d < headdim_qk)
    dk_bias_base = dK_bias + pid_head * stride_dk_bias_head
    tl.atomic_add(dk_bias_base + offs_d * stride_dk_bias_qkdim, dk_rotated, mask=offs_d < headdim_qk)
    k_base = K + pid_batch * stride_k_batch + last_pos * stride_k_seqlen + qk_head_idx * stride_k_head
    k_val = tl.load(k_base + offs_d * stride_k_qkdim, mask=offs_d < headdim_qk, other=0.0).to(tl.float32)
    kr = tl.reshape(k_val, [HEADDIM_QK // 2, 2])
    (k_r0, k_r1) = tl.split(kr)
    k_bias_base = K_bias + pid_head * stride_k_bias_head
    k_bias_val = tl.load(k_bias_base + offs_d * stride_k_bias_qkdim, mask=offs_d < headdim_qk, other=0.0).to(tl.float32)
    kbr = tl.reshape(k_bias_val, [HEADDIM_QK // 2, 2])
    (kb_r0, kb_r1) = tl.split(kbr)
    K_wbias_r0 = k_r0 + kb_r0
    K_wbias_r1 = k_r1 + kb_r1
    dtheta_k = dk_state_r0 * (-K_wbias_r0 * sin_ang - K_wbias_r1 * cos_ang) + dk_state_r1 * (K_wbias_r0 * cos_ang - K_wbias_r1 * sin_ang)
    da_base = dAngles + pid_batch * stride_dangles_batch + last_pos * stride_dangles_seqlen + pid_head * stride_dangles_head
    tl.atomic_add(da_base + offs_dr * stride_dangles_qkdim, dtheta_k, mask=offs_dr < headdim_angles)

def compute_dqktheta(q: torch.Tensor, k: torch.Tensor, scale: torch.Tensor, gamma: torch.Tensor, q_bias: torch.Tensor, k_bias: torch.Tensor, angles: torch.Tensor, dq_in: torch.Tensor, dk_in: torch.Tensor, dqk: torch.Tensor, d_ok_state: Optional[torch.Tensor]=None, chunk_size: int=64, Cu_Seqlens: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (batch, seqlen, nheads_qk, headdim_qk) = q.shape
    assert q.shape == k.shape
    nheads = scale.shape[1]
    nchunks = triton.cdiv(seqlen, chunk_size)
    GQA_RATIO = nheads // nheads_qk
    assert scale.shape == (batch, nheads, seqlen)
    assert gamma.shape == (batch, nheads, seqlen)
    assert q_bias.shape == (nheads, headdim_qk)
    assert k_bias.shape == (nheads, headdim_qk)
    headdim_angles = angles.shape[-1]
    assert angles.shape == (batch, seqlen, nheads, headdim_angles)
    assert dq_in.shape == (batch, seqlen, nheads, headdim_qk)
    assert dk_in.shape == (batch, seqlen, nheads, headdim_qk)
    assert dqk.shape == (batch, nheads, seqlen)
    if d_ok_state is not None:
        num_sequences = Cu_Seqlens.shape[0] - 1 if Cu_Seqlens is not None else batch
        assert d_ok_state.shape == (num_sequences, nheads, headdim_qk)
    assert nheads % nheads_qk == 0, 'nheads must be multiple of nheads_qk for GQA support'
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    if not dqk.is_contiguous():
        dqk = dqk.contiguous()
    if not angles.is_contiguous():
        angles = angles.contiguous()
    if not dq_in.is_contiguous():
        dq_in = dq_in.contiguous()
    if not dk_in.is_contiguous():
        dk_in = dk_in.contiguous()
    if q_bias.stride(-1) != 1:
        q_bias = q_bias.contiguous()
    if k_bias.stride(-1) != 1:
        k_bias = k_bias.contiguous()
    if d_ok_state is not None and (not d_ok_state.is_contiguous()):
        d_ok_state = d_ok_state.contiguous()
    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    BLOCK_HEADDIM_QK = min(HEADDIM_QK, 64)
    dq = torch.empty((batch, seqlen, nheads_qk, headdim_qk), dtype=dq_in.dtype, device=q.device)
    dk = torch.empty((batch, seqlen, nheads_qk, headdim_qk), dtype=dk_in.dtype, device=k.device)
    dangles = torch.empty((batch, seqlen, nheads, headdim_angles), dtype=angles.dtype, device=angles.device)
    dscale = torch.empty((batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, seqlen), dtype=scale.dtype, device=scale.device)
    dgamma = torch.empty((batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, seqlen), dtype=gamma.dtype, device=gamma.device)
    dq_bias_partial = torch.empty((batch, nchunks, nheads, headdim_qk), dtype=torch.float32, device=q.device)
    dk_bias_partial = torch.empty((batch, nchunks, nheads, headdim_qk), dtype=torch.float32, device=k.device)
    grid = (nchunks, batch)
    mamba3_siso_bwd_kernel_rotary_bias_angles[grid](q, k, scale, gamma, q_bias, k_bias, angles, dq_in, dk_in, dqk, dq, dk, dangles, dscale, dgamma, dq_bias_partial, dk_bias_partial, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), scale.stride(0), scale.stride(1), scale.stride(2), gamma.stride(0), gamma.stride(1), gamma.stride(2), q_bias.stride(0), q_bias.stride(1), k_bias.stride(0), k_bias.stride(1), angles.stride(0), angles.stride(1), angles.stride(2), angles.stride(3), dq_in.stride(0), dq_in.stride(1), dq_in.stride(2), dq_in.stride(3), dk_in.stride(0), dk_in.stride(1), dk_in.stride(2), dk_in.stride(3), dqk.stride(0), dqk.stride(1), dqk.stride(2), dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3), dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3), dangles.stride(0), dangles.stride(1), dangles.stride(2), dangles.stride(3), dscale.stride(0), dscale.stride(1), dscale.stride(2), dscale.stride(3), dgamma.stride(0), dgamma.stride(1), dgamma.stride(2), dgamma.stride(3), dq_bias_partial.stride(0), dq_bias_partial.stride(1), dq_bias_partial.stride(2), dq_bias_partial.stride(3), dk_bias_partial.stride(0), dk_bias_partial.stride(1), dk_bias_partial.stride(2), dk_bias_partial.stride(3), seqlen, nheads_qk, nheads, headdim_qk, headdim_angles, CHUNK_SIZE=chunk_size, HEADDIM_QK=HEADDIM_QK, BLOCK_HEADDIM_QK=BLOCK_HEADDIM_QK, GQA_RATIO=GQA_RATIO)
    dscale = torch.sum(dscale, dim=2)
    dgamma = torch.sum(dgamma, dim=2)
    dq_bias = dq_bias_partial.sum(dim=(0, 1))
    dk_bias = dk_bias_partial.sum(dim=(0, 1))
    if d_ok_state is not None:
        apply_dk_state_post(d_ok_state, angles, k, k_bias, dk, dk_bias, dangles, Cu_Seqlens)
    return (dq, dk, dq_bias, dk_bias, dangles, dscale, dgamma)

def apply_dk_state_post(d_ok_state: torch.Tensor, angles: torch.Tensor, k: torch.Tensor, k_bias: torch.Tensor, dk: torch.Tensor, dk_bias: torch.Tensor, dangles: torch.Tensor, Cu_Seqlens: Optional[torch.Tensor]=None):
    (batch, seqlen, nheads, headdim_angles) = angles.shape
    (_, _, headdim_qk) = d_ok_state.shape
    nheads_qk = k.shape[2]
    GQA_RATIO = nheads // nheads_qk
    is_varlen = Cu_Seqlens is not None
    if is_varlen:
        num_sequences = Cu_Seqlens.shape[0] - 1
        assert batch == 1
    else:
        num_sequences = batch
    if not d_ok_state.is_contiguous():
        d_ok_state = d_ok_state.contiguous()
    if not angles.is_contiguous():
        angles = angles.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not k_bias.is_contiguous():
        k_bias = k_bias.contiguous()
    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    grid = (nheads, num_sequences)
    mamba3_siso_bwd_kernel_dk_state_post[grid](d_ok_state, angles, k, k_bias, Cu_Seqlens, dk, dk_bias, dangles, d_ok_state.stride(0), d_ok_state.stride(1), d_ok_state.stride(2), angles.stride(0), angles.stride(1), angles.stride(2), angles.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), k_bias.stride(0), k_bias.stride(1), Cu_Seqlens.stride(0) if is_varlen else 0, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3), dk_bias.stride(0), dk_bias.stride(1), dangles.stride(0), dangles.stride(1), dangles.stride(2), dangles.stride(3), seqlen, headdim_qk, headdim_angles, HEADDIM_QK=HEADDIM_QK, GQA_RATIO=GQA_RATIO, IS_VARLEN=is_varlen, num_warps=2, num_stages=3)

@triton.autotune(configs=[triton.Config({'CHUNK_SIZE': cs}, num_stages=s, num_warps=w, maxnreg=r) for cs in [64, 128, 256] for s in [1, 2, 3] for w in [2, 4, 8] for r in [None, 128, 256]], key=['HEADDIM_V', 'HEADDIM_QK', 'HAS_INPUT_STATE', 'IS_VARLEN'])
@triton.jit
def mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states(dScale, dGamma, DT, Trap, d_ISSM_State, Input_K_State, Input_V_State, Cu_Seqlens, dDT, dTrap, dInput_SSM_State, dInput_K_State, dInput_V_State, stride_dscale_batch, stride_dscale_head, stride_dscale_seqlen, stride_dgamma_batch, stride_dgamma_head, stride_dgamma_seqlen, stride_dt_batch, stride_dt_head, stride_dt_seqlen, stride_trap_batch, stride_trap_head, stride_trap_seqlen, stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_vdim, stride_d_issm_state_qkdim, stride_input_k_state_batch, stride_input_k_state_head, stride_input_k_state_qkdim, stride_input_v_state_batch, stride_input_v_state_head, stride_input_v_state_vdim, stride_cu_seqlen, stride_ddt_batch, stride_ddt_head, stride_ddt_seqlen, stride_dtrap_batch, stride_dtrap_head, stride_dtrap_seqlen, stride_dinput_ssm_state_batch, stride_dinput_ssm_state_head, stride_dinput_ssm_state_vdim, stride_dinput_ssm_state_qkdim, stride_dinput_k_state_batch, stride_dinput_k_state_head, stride_dinput_k_state_qkdim, stride_dinput_v_state_batch, stride_dinput_v_state_head, stride_dinput_v_state_vdim, seqlen, headdim_v, headdim_qk, CHUNK_SIZE: tl.constexpr, HEADDIM_V: tl.constexpr, HEADDIM_QK: tl.constexpr, HAS_INPUT_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    if IS_VARLEN:
        pid_seq = pid_batch
        pid_batch = 0
        cu_seqlen = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_next = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seqlen = cu_seqlen_next - cu_seqlen
    else:
        pid_seq = 0
        cu_seqlen = 0
    dscale_offset = pid_batch * stride_dscale_batch + pid_head * stride_dscale_head + IS_VARLEN * cu_seqlen * stride_dscale_seqlen
    dgamma_offset = pid_batch * stride_dgamma_batch + pid_head * stride_dgamma_head + IS_VARLEN * cu_seqlen * stride_dgamma_seqlen
    dt_offset = pid_batch * stride_dt_batch + pid_head * stride_dt_head + IS_VARLEN * cu_seqlen * stride_dt_seqlen
    trap_offset = pid_batch * stride_trap_batch + pid_head * stride_trap_head + IS_VARLEN * cu_seqlen * stride_trap_seqlen
    ddt_offset = pid_batch * stride_ddt_batch + pid_head * stride_ddt_head + IS_VARLEN * cu_seqlen * stride_ddt_seqlen
    dtrap_offset = pid_batch * stride_dtrap_batch + pid_head * stride_dtrap_head + IS_VARLEN * cu_seqlen * stride_dtrap_seqlen
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    for chunk_idx in range(num_chunks):
        offs_s = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
        mask = offs_s < seqlen
        dscale_t = tl.load(dScale + dscale_offset + offs_s * stride_dscale_seqlen, mask=mask, other=0.0)
        dgamma_t = tl.load(dGamma + dgamma_offset + offs_s * stride_dgamma_seqlen, mask=mask, other=0.0)
        trap_presig_t = tl.load(Trap + trap_offset + offs_s * stride_trap_seqlen, mask=mask, other=0.0).to(tl.float32)
        trap_t = sigmoid_approx(trap_presig_t)
        dt_t = tl.load(DT + dt_offset + offs_s * stride_dt_seqlen, mask=mask, other=0.0)
        offs_s_prev = offs_s - 1
        mask_prev = (offs_s_prev >= 0) & (offs_s_prev < seqlen)
        dscale_prev = tl.load(dScale + dscale_offset + offs_s_prev * stride_dscale_seqlen, mask=mask_prev, other=0.0)
        ddt_t = (dgamma_t + dscale_t) * trap_t + dscale_prev * (1.0 - trap_t)
        dtrap_t = (dgamma_t + dscale_t) * dt_t - dscale_prev * dt_t
        dtrap_presig_t = dtrap_t * trap_t * (1.0 - trap_t)
        tl.store(dDT + ddt_offset + offs_s * stride_ddt_seqlen, ddt_t, mask=mask)
        tl.store(dTrap + dtrap_offset + offs_s * stride_dtrap_seqlen, dtrap_presig_t, mask=mask)
    if HAS_INPUT_STATE:
        d_issm_offset = (pid_batch + pid_seq) * stride_d_issm_state_batch + pid_head * stride_d_issm_state_head
        input_k_offset = (pid_batch + pid_seq) * stride_input_k_state_batch + pid_head * stride_input_k_state_head
        input_v_offset = (pid_batch + pid_seq) * stride_input_v_state_batch + pid_head * stride_input_v_state_head
        dinput_ssm_offset = (pid_batch + pid_seq) * stride_dinput_ssm_state_batch + pid_head * stride_dinput_ssm_state_head
        dinput_k_offset = (pid_batch + pid_seq) * stride_dinput_k_state_batch + pid_head * stride_dinput_k_state_head
        dinput_v_offset = (pid_batch + pid_seq) * stride_dinput_v_state_batch + pid_head * stride_dinput_v_state_head
        dt_0 = tl.load(DT + dt_offset).to(tl.float32)
        trap_presig_0 = tl.load(Trap + trap_offset).to(tl.float32)
        trap_0 = sigmoid_approx(trap_presig_0)
        scalar = dt_0 * (1.0 - trap_0)
        offs_v = tl.arange(0, HEADDIM_V)
        offs_qk = tl.arange(0, HEADDIM_QK)
        input_k = tl.load(Input_K_State + input_k_offset + offs_qk * stride_input_k_state_qkdim, mask=offs_qk < headdim_qk, other=0.0).to(tl.float32)
        input_v = tl.load(Input_V_State + input_v_offset + offs_v * stride_input_v_state_vdim, mask=offs_v < headdim_v, other=0.0).to(tl.float32)
        d_issm = tl.load(d_ISSM_State + d_issm_offset + offs_v[:, None] * stride_d_issm_state_vdim + offs_qk[None, :] * stride_d_issm_state_qkdim, mask=(offs_v[:, None] < headdim_v) & (offs_qk[None, :] < headdim_qk), other=0.0).to(tl.float32)
        tl.store(dInput_SSM_State + dinput_ssm_offset + offs_v[:, None] * stride_dinput_ssm_state_vdim + offs_qk[None, :] * stride_dinput_ssm_state_qkdim, d_issm, mask=(offs_v[:, None] < headdim_v) & (offs_qk[None, :] < headdim_qk))
        outer_product = input_v[:, None] * input_k[None, :]
        d_scalar = tl.sum(d_issm * outer_product)
        dinput_v = tl.sum(d_issm * input_k[None, :], axis=1) * scalar
        dinput_k = tl.sum(d_issm * input_v[:, None], axis=0) * scalar
        tl.store(dInput_V_State + dinput_v_offset + offs_v * stride_dinput_v_state_vdim, dinput_v, mask=offs_v < headdim_v)
        tl.store(dInput_K_State + dinput_k_offset + offs_qk * stride_dinput_k_state_qkdim, dinput_k, mask=offs_qk < headdim_qk)
        ddt_0_contrib = d_scalar * (1.0 - trap_0)
        dtrap_0_contrib = d_scalar * -dt_0
        dtrap_0_presig_contrib = dtrap_0_contrib * trap_0 * (1.0 - trap_0)
        tl.atomic_add(dDT + ddt_offset, ddt_0_contrib)
        tl.atomic_add(dTrap + dtrap_offset, dtrap_0_presig_contrib)

def compute_ddt_dtrap_dinput_states(dscale: torch.Tensor, dgamma: torch.Tensor, dt: torch.Tensor, trap: torch.Tensor, d_issm_state: Optional[torch.Tensor]=None, input_k_state: Optional[torch.Tensor]=None, input_v_state: Optional[torch.Tensor]=None, Cu_Seqlens: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    (batch, nheads, seqlen) = dscale.shape
    has_input_state = d_issm_state is not None
    is_varlen = Cu_Seqlens is not None
    if is_varlen:
        num_sequences = Cu_Seqlens.shape[0] - 1
        assert batch == 1, 'Batch size must be 1 when using variable-length sequences.'
    else:
        num_sequences = batch
    assert dgamma.shape == (batch, nheads, seqlen), f'dgamma shape mismatch: {dgamma.shape}'
    assert dt.shape == (batch, nheads, seqlen), f'dt shape mismatch: {dt.shape}'
    assert trap.shape == (batch, nheads, seqlen), f'trap shape mismatch: {trap.shape}'
    if has_input_state:
        assert input_k_state is not None and input_v_state is not None, 'input_k_state and input_v_state must be provided with d_issm_state'
        (headdim_v, headdim_qk) = (d_issm_state.shape[2], d_issm_state.shape[3])
        assert d_issm_state.shape == (num_sequences, nheads, headdim_v, headdim_qk), f'd_issm_state shape mismatch: {d_issm_state.shape}'
        assert input_k_state.shape == (num_sequences, nheads, headdim_qk), f'input_k_state shape mismatch: {input_k_state.shape}'
        assert input_v_state.shape == (num_sequences, nheads, headdim_v), f'input_v_state shape mismatch: {input_v_state.shape}'
    else:
        (headdim_v, headdim_qk) = (64, 128)
    dscale = dscale.contiguous() if not dscale.is_contiguous() else dscale
    dgamma = dgamma.contiguous() if not dgamma.is_contiguous() else dgamma
    dt = dt.contiguous() if not dt.is_contiguous() else dt
    trap = trap.contiguous() if not trap.is_contiguous() else trap
    if has_input_state:
        d_issm_state = d_issm_state.contiguous() if not d_issm_state.is_contiguous() else d_issm_state
        input_k_state = input_k_state.contiguous() if not input_k_state.is_contiguous() else input_k_state
        input_v_state = input_v_state.contiguous() if not input_v_state.is_contiguous() else input_v_state
    dDT = torch.empty_like(dt, dtype=torch.float32)
    dTrap = torch.empty_like(trap, dtype=torch.float32)
    if has_input_state:
        d_Input_SSM_State = torch.empty_like(d_issm_state)
        d_Input_K_State = torch.empty((num_sequences, nheads, headdim_qk), dtype=torch.float32, device=dt.device)
        d_Input_V_State = torch.empty((num_sequences, nheads, headdim_v), dtype=torch.float32, device=dt.device)
    else:
        d_Input_SSM_State = None
        d_Input_K_State = None
        d_Input_V_State = None
    HEADDIM_V = triton.next_power_of_2(headdim_v) if has_input_state else 64
    HEADDIM_QK = triton.next_power_of_2(headdim_qk) if has_input_state else 128
    if is_varlen:
        grid = (nheads, num_sequences)
    else:
        grid = (nheads, batch)
    mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states[grid](dscale, dgamma, dt, trap, d_issm_state if has_input_state else dscale, input_k_state if has_input_state else dscale, input_v_state if has_input_state else dscale, Cu_Seqlens, dDT, dTrap, d_Input_SSM_State if has_input_state else dDT, d_Input_K_State if has_input_state else dDT, d_Input_V_State if has_input_state else dDT, dscale.stride(0), dscale.stride(1), dscale.stride(2), dgamma.stride(0), dgamma.stride(1), dgamma.stride(2), dt.stride(0), dt.stride(1), dt.stride(2), trap.stride(0), trap.stride(1), trap.stride(2), d_issm_state.stride(0) if has_input_state else 0, d_issm_state.stride(1) if has_input_state else 0, d_issm_state.stride(2) if has_input_state else 0, d_issm_state.stride(3) if has_input_state else 0, input_k_state.stride(0) if has_input_state else 0, input_k_state.stride(1) if has_input_state else 0, input_k_state.stride(2) if has_input_state else 0, input_v_state.stride(0) if has_input_state else 0, input_v_state.stride(1) if has_input_state else 0, input_v_state.stride(2) if has_input_state else 0, Cu_Seqlens.stride(0) if Cu_Seqlens is not None else 0, dDT.stride(0), dDT.stride(1), dDT.stride(2), dTrap.stride(0), dTrap.stride(1), dTrap.stride(2), d_Input_SSM_State.stride(0) if has_input_state else 0, d_Input_SSM_State.stride(1) if has_input_state else 0, d_Input_SSM_State.stride(2) if has_input_state else 0, d_Input_SSM_State.stride(3) if has_input_state else 0, d_Input_K_State.stride(0) if has_input_state else 0, d_Input_K_State.stride(1) if has_input_state else 0, d_Input_K_State.stride(2) if has_input_state else 0, d_Input_V_State.stride(0) if has_input_state else 0, d_Input_V_State.stride(1) if has_input_state else 0, d_Input_V_State.stride(2) if has_input_state else 0, seqlen, headdim_v, headdim_qk, HEADDIM_V=HEADDIM_V, HEADDIM_QK=HEADDIM_QK, HAS_INPUT_STATE=has_input_state, IS_VARLEN=is_varlen)
    return (dDT, dTrap, d_Input_SSM_State, d_Input_K_State, d_Input_V_State)

def _alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device='cuda', dtype=torch.int8)
triton.set_allocator(_alloc_fn)
