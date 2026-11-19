# Copyright (c) 2025, Tri Dao.
'Rotary step.'
import math
from typing import Optional
from typing import Tuple
import torch
import triton
import triton.language as tl
import triton.testing

@triton.jit
def rotary_qk_inference_kernel(OUT_Q, OUT_K, OUT_ANGLE_STATE, Q, K, ANGLE_STATE, ANGLE_PROJ, DT, BIAS_Q, BIAS_K, nheads, headdim, stride_out_q, stride_out_k, stride_out_angle_state, stride_q, stride_k, stride_angle_state, stride_angle_proj, stride_dt, stride_bias_q, stride_bias_k, ROTARY_DIM: tl.constexpr, CONJUGATE: tl.constexpr, HAS_BIAS_Q: tl.constexpr, HAS_BIAS_K: tl.constexpr, MIMO_DIM: tl.constexpr, BLOCK_D: tl.constexpr, ROTATE_PAIRWISE: tl.constexpr):
    pid_nheads = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    Q = Q + pid_batch * stride_q[0] + pid_nheads * stride_q[2]
    K = K + pid_batch * stride_k[0] + pid_nheads * stride_k[2]
    ANGLE_STATE = ANGLE_STATE + pid_batch * stride_angle_state[0] + pid_nheads * stride_angle_state[1]
    ANGLE_PROJ = ANGLE_PROJ + pid_batch * stride_angle_proj[0] + pid_nheads * stride_angle_proj[1]
    DT = DT + pid_batch * stride_dt[0] + pid_nheads * stride_dt[1]
    OUT_Q = OUT_Q + pid_batch * stride_out_q[0] + pid_nheads * stride_out_q[2]
    OUT_K = OUT_K + pid_batch * stride_out_k[0] + pid_nheads * stride_out_k[2]
    OUT_ANGLE_STATE = OUT_ANGLE_STATE + pid_batch * stride_out_angle_state[0] + pid_nheads * stride_out_angle_state[1]
    rm = tl.arange(0, MIMO_DIM)
    rd = tl.arange(0, BLOCK_D)
    rd_half = tl.arange(0, BLOCK_D // 2)
    ANGLE_STATE = ANGLE_STATE + rd_half * stride_angle_state[2]
    mask_angle = rd_half < ROTARY_DIM // 2
    angle_state = tl.load(ANGLE_STATE, mask=mask_angle, other=0.0).to(tl.float32)
    ANGLE_PROJ = ANGLE_PROJ + rd_half * stride_angle_proj[2]
    angle_proj = tl.load(ANGLE_PROJ, mask=mask_angle, other=0.0).to(tl.float32)
    dt = tl.load(DT, mask=True, other=0.0).to(tl.float32)
    angle_proj = tl.sigmoid(2.0 * angle_proj) * 2.0 - 1.0
    angle = angle_state + angle_proj * dt * 3.141592653589793
    OUT_ANGLE_STATE = OUT_ANGLE_STATE + rd_half * stride_out_angle_state[2]
    tl.store(OUT_ANGLE_STATE, angle, mask=mask_angle)
    angle = angle[None, :]
    cos = tl.cos(angle)
    sin = tl.sin(angle)
    if CONJUGATE:
        sin = -sin
    Q = Q + (rm[:, None] * stride_q[1] + rd[None, :] * stride_q[3])
    OUT_Q = OUT_Q + (rm[:, None] * stride_out_q[1] + rd[None, :] * stride_out_q[3])
    mask = rd[None, :] < headdim
    q = tl.load(Q, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS_Q:
        BIAS_Q = BIAS_Q + pid_nheads * stride_bias_q[1]
        BIAS_Q = BIAS_Q + (rm[:, None] * stride_bias_q[0] + rd[None, :] * stride_bias_q[2])
        bias_q = tl.load(BIAS_Q, mask=mask, other=0.0).to(tl.float32)
        q = q + bias_q
    if ROTATE_PAIRWISE:
        (q0, q1) = tl.split(tl.reshape(q, [MIMO_DIM, BLOCK_D // 2, 2]))
        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        qo = tl.reshape(tl.join(qo0, qo1), [MIMO_DIM, BLOCK_D])
        tl.store(OUT_Q, qo, mask=mask)
    else:
        q_reshaped = tl.reshape(q, [MIMO_DIM, 2, BLOCK_D // 2])
        q_permuted = tl.permute(q_reshaped, (0, 2, 1))
        (q0, q1) = tl.split(q_permuted)
        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        q_joined = tl.join(qo0, qo1)
        q_final = tl.permute(q_joined, (0, 2, 1))
        qo = tl.reshape(q_final, [MIMO_DIM, BLOCK_D])
        tl.store(OUT_Q, qo, mask=mask)
    K = K + (rm[:, None] * stride_k[1] + rd[None, :] * stride_k[3])
    OUT_K = OUT_K + (rm[:, None] * stride_out_k[1] + rd[None, :] * stride_out_k[3])
    k = tl.load(K, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS_K:
        BIAS_K = BIAS_K + pid_nheads * stride_bias_k[1]
        BIAS_K = BIAS_K + (rm[:, None] * stride_bias_k[0] + rd[None, :] * stride_bias_k[2])
        bias_k = tl.load(BIAS_K, mask=mask, other=0.0).to(tl.float32)
        k = k + bias_k
    if ROTATE_PAIRWISE:
        (k0, k1) = tl.split(tl.reshape(k, [MIMO_DIM, BLOCK_D // 2, 2]))
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos
        ko = tl.reshape(tl.join(ko0, ko1), [MIMO_DIM, BLOCK_D])
        tl.store(OUT_K, ko, mask=mask)
    else:
        k_reshaped = tl.reshape(k, [MIMO_DIM, 2, BLOCK_D // 2])
        k_permuted = tl.permute(k_reshaped, (0, 2, 1))
        (k0, k1) = tl.split(k_permuted)
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos
        k_joined = tl.join(ko0, ko1)
        k_final = tl.permute(k_joined, (0, 2, 1))
        ko = tl.reshape(k_final, [MIMO_DIM, BLOCK_D])
        tl.store(OUT_K, ko, mask=mask)

def apply_rotary_qk_inference_fwd(q: torch.Tensor, k: torch.Tensor, angle_state: torch.Tensor, angle_proj: torch.Tensor, dt: torch.Tensor, bias_q: Optional[torch.Tensor]=None, bias_k: Optional[torch.Tensor]=None, inplace=False, conjugate=False, rotate_pairwise=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (batch, mimo_dim, nheads, headdim) = q.shape
    assert headdim % 2 == 0
    assert k.shape == q.shape, f'k shape {k.shape} != q shape {q.shape}'
    rotary_dim = angle_state.shape[-1] * 2
    assert angle_state.shape == (batch, nheads, rotary_dim // 2)
    assert angle_state.shape == angle_proj.shape
    assert dt.shape == (batch, nheads)
    assert rotary_dim <= headdim, 'rotary_dim must be <= headdim'
    assert headdim <= 256, 'Only support headdim <= 256'
    if bias_q is not None:
        assert bias_q.shape == (mimo_dim, nheads, headdim), f'bias_q shape {bias_q.shape} != (mimo_dim, nheads, headdim) {(mimo_dim, nheads, headdim)}'
        bias_q = bias_q.contiguous()
    if bias_k is not None:
        assert bias_k.shape == (mimo_dim, nheads, headdim), f'bias_k shape {bias_k.shape} != (mimo_dim, nheads, headdim) {(mimo_dim, nheads, headdim)}'
        bias_k = bias_k.contiguous()
    output_q = torch.empty_like(q) if not inplace else q
    output_k = torch.empty_like(k) if not inplace else k
    output_angle_state = torch.empty_like(angle_state) if not inplace else angle_state
    grid = lambda META: (nheads, batch)
    with torch.cuda.device(q.device.index):
        torch.library.wrap_triton(rotary_qk_inference_kernel)[grid](output_q, output_k, output_angle_state, q, k, angle_state, angle_proj, dt, bias_q, bias_k, nheads, headdim, output_q.stride(), output_k.stride(), output_angle_state.stride(), q.stride(), k.stride(), angle_state.stride(), angle_proj.stride(), dt.stride(), bias_q.stride() if bias_q is not None else (0, 0, 0), bias_k.stride() if bias_k is not None else (0, 0, 0), rotary_dim, conjugate, bias_q is not None, bias_k is not None, MIMO_DIM=mimo_dim, BLOCK_D=triton.next_power_of_2(headdim), num_warps=8, ROTATE_PAIRWISE=rotate_pairwise)
    return (output_q, output_k, output_angle_state)

def apply_rotary_qk_inference_reference(q: torch.Tensor, k: torch.Tensor, angle_state: torch.Tensor, angle_proj: torch.Tensor, dt: torch.Tensor, bias_q: Optional[torch.Tensor]=None, bias_k: Optional[torch.Tensor]=None, conjugate=False, rotate_pairwise=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (batch, mimo_dim, nheads, headdim) = q.shape
    rotary_dim = angle_state.shape[-1] * 2
    angle_proj = torch.tanh(angle_proj)
    angle = angle_state + angle_proj * dt[:, :, None] * math.pi
    angle_state_new = angle
    angle = angle.unsqueeze(1).expand(-1, mimo_dim, -1, -1)
    if bias_q is not None:
        q = q + bias_q[None, :, :, :]
    if bias_k is not None:
        k = k + bias_k[None, :, :, :]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    if conjugate:
        sin = -sin
    if rotate_pairwise:
        q_rot = q_rot.reshape(batch, mimo_dim, nheads, rotary_dim // 2, 2)
        (q0, q1) = (q_rot[..., 0], q_rot[..., 1])
        k_rot = k_rot.reshape(batch, mimo_dim, nheads, rotary_dim // 2, 2)
        (k0, k1) = (k_rot[..., 0], k_rot[..., 1])
        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos
        qout_rot = torch.stack([qo0, qo1], dim=-1).reshape(batch, mimo_dim, nheads, rotary_dim)
        kout_rot = torch.stack([ko0, ko1], dim=-1).reshape(batch, mimo_dim, nheads, rotary_dim)
        if rotary_dim < headdim:
            q_out = torch.cat([qout_rot, q_pass], dim=-1)
            k_out = torch.cat([kout_rot, k_pass], dim=-1)
        else:
            q_out = qout_rot
            k_out = kout_rot
    else:
        half = headdim // 2
        (q0, q1) = (q[..., :half], q[..., half:])
        (k0, k1) = (k[..., :half], k[..., half:])
        rdim_half = rotary_dim // 2
        if half > rdim_half:
            pad_shape = list(cos.shape)
            pad_shape[-1] = half - rdim_half
            cos = torch.cat([cos, torch.ones(pad_shape, device=cos.device, dtype=cos.dtype)], dim=-1)
            sin = torch.cat([sin, torch.zeros(pad_shape, device=sin.device, dtype=sin.dtype)], dim=-1)
        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos
        q_out = torch.cat([qo0, qo1], dim=-1)
        k_out = torch.cat([ko0, ko1], dim=-1)
    return (q_out, k_out, angle_state_new)
