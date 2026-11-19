'Dual attention.'
import torch
import triton
import triton.language as tl

@triton.jit
def _dual(Q, K, V, MK, MV, VALID, QVALID, O0, OA, QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr, VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr, MS0: tl.constexpr, MS1: tl.constexpr, MS2: tl.constexpr, NS0: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr, H: tl.constexpr, L: tl.constexpr, N: tl.constexpr, M: tl.constexpr, D: tl.constexpr, PREFIX: tl.constexpr, CAUSAL: tl.constexpr, T: tl.constexpr, U: tl.constexpr, F: tl.constexpr):
    row = tl.program_id(0)
    qidx = row % L
    head = row // L % H
    batch = row // (L * H)
    d = tl.arange(0, F)
    k = tl.arange(0, T)
    m = tl.arange(0, U)
    q = tl.load(Q + batch * QS0 + head * QS1 + qidx * QS2 + d, d < D, 0).to(tl.float32)
    keys = tl.load(K + batch * KS0 + head * KS1 + k[:, None] * KS2 + d[None, :], (k[:, None] < N) & (d[None, :] < D), 0).to(tl.float32)
    valid = tl.load(VALID + batch * N + k, k < N, 0)
    qvalid = tl.load(QVALID + batch * L + qidx)
    allowed = (k < N) & valid & qvalid
    if CAUSAL:
        allowed = allowed & (k <= PREFIX + qidx)
    scores = tl.sum(keys * q[None, :], 1) * D ** (-0.5)
    scores = tl.where(allowed, scores, float('-inf'))
    mx = tl.max(scores, 0)
    safe = tl.where(mx == float('-inf'), 0.0, mx)
    prob = tl.exp(scores - safe)
    denom = tl.sum(prob, 0)
    vals = tl.load(V + batch * VS0 + head * VS1 + k[:, None] * VS2 + d[None, :], (k[:, None] < N) & (d[None, :] < D), 0).to(tl.float32)
    num = tl.sum(prob[:, None] * vals, 0)
    out0 = num / tl.where(denom > 0, denom, 1.0)
    memk = tl.load(MK + batch * MS0 + head * MS1 + m[:, None] * MS2 + d[None, :], (m[:, None] < M) & (d[None, :] < D), 0).to(tl.float32)
    memv = tl.load(MV + batch * NS0 + head * NS1 + m[:, None] * NS2 + d[None, :], (m[:, None] < M) & (d[None, :] < D), 0).to(tl.float32)
    scorem = tl.sum(memk * q[None, :], 1) * D ** (-0.5)
    scorem = tl.where((m < M) & qvalid, scorem, float('-inf'))
    mm = tl.max(scorem, 0)
    both = tl.maximum(mx, mm)
    bothsafe = tl.where(both == float('-inf'), 0.0, both)
    scale0 = tl.exp(mx - bothsafe)
    pm = tl.exp(scorem - bothsafe)
    dm = tl.sum(pm, 0)
    nm = tl.sum(pm[:, None] * memv, 0)
    total = denom * scale0 + dm
    outa = (num * scale0 + nm) / tl.where(total > 0, total, 1.0)
    tl.store(O0 + row * D + d, out0, d < D)
    tl.store(OA + row * D + d, outa, d < D)

def dual_attention(q, k, v, mk, mv, key_valid, query_valid, *, prefix, causal):
    if torch.is_grad_enabled():
        raise RuntimeError('dual attention is inference only')
    assert q.is_cuda and q.dtype == torch.float32
    (b, h, l, d) = q.shape
    n = k.shape[2]
    m = mk.shape[2]
    assert d <= 128 and n <= 256 and (m <= 16)
    assert all((x.stride(-1) == 1 and x.dtype == q.dtype for x in [q, k, v, mk, mv]))
    o = torch.empty_like(q, memory_format=torch.contiguous_format)
    a = torch.empty_like(o)
    _dual[b * h * l,](q, k, v, mk, mv, key_valid.contiguous(), query_valid.contiguous(), o, a, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *mk.stride()[:3], *mv.stride()[:3], h, l, n, m, d, prefix, causal, triton.next_power_of_2(max(n, 1)), triton.next_power_of_2(max(m, 1)), triton.next_power_of_2(d), num_warps=4, enable_fp_fusion=False)
    return (o, a)
