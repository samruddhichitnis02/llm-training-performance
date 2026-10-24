"""Fused Triton kernels for the Llama style training step, forward and backward.

Four fused operations, each wired into torch.autograd.Function:
  rmsnorm / add_rmsnorm        residual add plus RMSNorm (8.1)
  swiglu                       silu(a) * b (8.2)
  rope_qkv                     RoPE on the packed qkv output, split into q, k, v (8.3)
  fused_linear_cross_entropy   chunked lm head plus cross entropy, no full logits (8.4)

All math inside the kernels is float32. Loads and stores use the tensor dtype.
"""

import functools

import torch
import triton
import triton.language as tl


@functools.lru_cache(maxsize=None)
def _sm_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _row_warps(block):
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    if block >= 1024:
        return 4
    return 2


# --------------------------------------------------------------------------
# 8.1 Residual add plus RMSNorm
# --------------------------------------------------------------------------

@triton.jit
def _rms_fwd_kernel(A, R, W, Y, S, RSTD, D, eps,
                    HAS_A: tl.constexpr, BLOCK: tl.constexpr):
    """One program per row, whole row in one block. s = a + r, y = s * rstd * w."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    off = row * D + cols
    s = tl.load(R + off, mask=mask, other=0.0).to(tl.float32)
    if HAS_A:
        s += tl.load(A + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(S + off, s.to(S.dtype.element_ty), mask=mask)
    rstd = 1.0 / tl.sqrt(tl.sum(s * s, axis=0) / D + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + off, (s * rstd * w).to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rms_bwd_kernel(DY, DSIN, S, W, RSTD, DA, DR, DWP, N, D,
                    HAS_DS: tl.constexpr, HAS_A: tl.constexpr, BLOCK: tl.constexpr):
    """Persistent backward: each program loops over rows pid, pid + P, ...

    Per row, with g = dy * w and c = mean(g * s):
        ds_total = rstd * g - s * rstd^3 * c + ds_in      (same value to a and r)
    The weight gradient dw = sum over rows of dy * s * rstd is accumulated in
    registers and written once per program into a (P, D) buffer.
    """
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK], dtype=tl.float32)
    for row in range(pid, N, P):
        off = row.to(tl.int64) * D + cols
        dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(S + off, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row)
        g = dy * w
        c = tl.sum(g * s, axis=0) / D
        ds = rstd * g - s * (rstd * rstd * rstd) * c
        if HAS_DS:
            ds += tl.load(DSIN + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(DR + off, ds.to(DR.dtype.element_ty), mask=mask)
        if HAS_A:
            tl.store(DA + off, ds.to(DA.dtype.element_ty), mask=mask)
        dw += dy * s * rstd
    tl.store(DWP + pid * D + cols, dw, mask=mask)


def _rms_forward(a2, r2, w, eps, out_dtype, s_dtype):
    N, D = r2.shape
    BLOCK = triton.next_power_of_2(D)
    y = torch.empty((N, D), dtype=out_dtype, device=r2.device)
    s = torch.empty((N, D), dtype=s_dtype, device=r2.device) if a2 is not None else r2
    rstd = torch.empty(N, dtype=torch.float32, device=r2.device)
    _rms_fwd_kernel[(N,)](a2 if a2 is not None else r2, r2, w, y, s, rstd, D, eps,
                          HAS_A=a2 is not None, BLOCK=BLOCK, num_warps=_row_warps(BLOCK))
    return y, s, rstd


def _rms_backward(dy, ds_in, s, w, rstd, a_dtype, r_dtype):
    N, D = s.shape
    BLOCK = triton.next_power_of_2(D)
    P = max(1, min(4 * _sm_count(s.device.index or 0), N))
    dr = torch.empty((N, D), dtype=r_dtype, device=s.device)
    da = torch.empty((N, D), dtype=a_dtype, device=s.device) if a_dtype is not None else dr
    dwp = torch.empty((P, D), dtype=torch.float32, device=s.device)
    _rms_bwd_kernel[(P,)](dy, ds_in if ds_in is not None else dy, s, w, rstd, da, dr, dwp, N, D,
                          HAS_DS=ds_in is not None, HAS_A=a_dtype is not None,
                          BLOCK=BLOCK, num_warps=_row_warps(BLOCK))
    return (da if a_dtype is not None else None), dr, dwp.sum(0).to(w.dtype)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps, out_dtype):
        shape, D = x.shape, x.shape[-1]
        x2 = x.reshape(-1, D).contiguous()
        y, s, rstd = _rms_forward(None, x2, w.contiguous(), eps, out_dtype or x.dtype, x.dtype)
        ctx.save_for_backward(s, w, rstd)
        ctx.x_dtype = x.dtype
        return y.view(*shape[:-1], D)

    @staticmethod
    def backward(ctx, dy):
        s, w, rstd = ctx.saved_tensors
        D = s.shape[-1]
        _, dx, dw = _rms_backward(dy.reshape(-1, D).contiguous(), None, s, w, rstd, None, ctx.x_dtype)
        return dx.view(dy.shape), dw, None, None


class _AddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, r, w, eps, out_dtype):
        shape, D = r.shape, r.shape[-1]
        a2 = a.reshape(-1, D).contiguous()
        r2 = r.reshape(-1, D).contiguous()
        s_dtype = torch.promote_types(a.dtype, r.dtype)
        y, s, rstd = _rms_forward(a2, r2, w.contiguous(), eps, out_dtype or a.dtype, s_dtype)
        ctx.save_for_backward(s, w, rstd)
        ctx.dtypes = (a.dtype, r.dtype)
        return y.view(*shape[:-1], D), s.view(*shape[:-1], D)

    @staticmethod
    def backward(ctx, dy, ds_in):
        s, w, rstd = ctx.saved_tensors
        D = s.shape[-1]
        if dy is None:
            dy = torch.zeros_like(ds_in)
        ds2 = ds_in.reshape(-1, D).contiguous() if ds_in is not None else None
        da, dr, dw = _rms_backward(dy.reshape(-1, D).contiguous(), ds2, s, w, rstd, *ctx.dtypes)
        return da.view(dy.shape), dr.view(dy.shape), dw, None, None


def rmsnorm(x, w, eps=1e-5, out_dtype=None):
    """y = x * rstd * w, with y in out_dtype (default x.dtype)."""
    return _RMSNorm.apply(x, w, eps, out_dtype)


def add_rmsnorm(a, r, w, eps=1e-5, out_dtype=None):
    """s = a + r, y = rmsnorm(s) * w. Returns (y, s). a and r may have different dtypes;
    s gets the wider of the two, y gets out_dtype (default a.dtype)."""
    return _AddRMSNorm.apply(a, r, w, eps, out_dtype)


# --------------------------------------------------------------------------
# 8.2 SwiGLU
# --------------------------------------------------------------------------

SWIGLU_BLOCK = 2048


@triton.jit
def _swiglu_fwd_kernel(A, B, O, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    out = a * tl.sigmoid(a) * b
    tl.store(O + offs, out.to(O.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(DO, A, B, DA, DB, n, BLOCK: tl.constexpr):
    """sig = sigmoid(a); da = dout * b * sig * (1 + a * (1 - sig)); db = dout * a * sig."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    do = tl.load(DO + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(a)
    da = do * b * sig * (1.0 + a * (1.0 - sig))
    db = do * a * sig
    tl.store(DA + offs, da.to(DA.dtype.element_ty), mask=mask)
    tl.store(DB + offs, db.to(DB.dtype.element_ty), mask=mask)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a, b = a.contiguous(), b.contiguous()
        out = torch.empty_like(a)
        n = a.numel()
        _swiglu_fwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](a, b, out, n, BLOCK=SWIGLU_BLOCK, num_warps=8)
        ctx.save_for_backward(a, b)   # sigmoid is recomputed in backward
        return out

    @staticmethod
    def backward(ctx, dout):
        a, b = ctx.saved_tensors
        dout = dout.contiguous()
        da, db = torch.empty_like(a), torch.empty_like(b)
        n = a.numel()
        _swiglu_bwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](dout, a, b, da, db, n, BLOCK=SWIGLU_BLOCK, num_warps=8)
        return da, db


def swiglu(a, b):
    """silu(a) * b."""
    return _SwiGLU.apply(a, b)


# --------------------------------------------------------------------------
# 8.3 RoPE on packed qkv
# --------------------------------------------------------------------------

@triton.jit
def _rope_fwd_kernel(QKV, COS, SIN, OUT, T, BHTD, H,
                     HD: tl.constexpr, HALF: tl.constexpr, HB: tl.constexpr):
    """One program per (token, q|k|v), all heads as an (HB, HALF) tile per half.

    Input row layout is (3, H, HD). Output is (3, B, H, T, HD) so q, k, v are contiguous.
    Rotation per pair (x1, x2): y1 = x1 c - x2 s, y2 = x2 c + x1 s. v is copied (c = 1, s = 0).
    """
    bt = tl.program_id(0)
    which = tl.program_id(1)
    b = bt // T
    t = bt % T
    h = tl.arange(0, HB)[:, None]
    d = tl.arange(0, HALF)[None, :]
    mask = h < H
    src = QKV + bt.to(tl.int64) * (3 * H * HD) + which * (H * HD) + h * HD + d
    x1 = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(src + HALF, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(COS + t * HD + d)
    s = tl.load(SIN + t * HD + d)
    rot = which < 2
    c = tl.where(rot, c, 1.0)
    s = tl.where(rot, s, 0.0)
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    dst = OUT + which.to(tl.int64) * BHTD + b.to(tl.int64) * (H * T * HD) + h * (T * HD) + t * HD + d
    tl.store(dst, y1.to(OUT.dtype.element_ty), mask=mask)
    tl.store(dst + HALF, y2.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _rope_bwd_kernel(DQ, DK, DV, COS, SIN, DQKV, T, H,
                     HD: tl.constexpr, HALF: tl.constexpr, HB: tl.constexpr):
    """Transpose rotation: dx1 = dy1 c + dy2 s, dx2 = dy2 c - dy1 s, written straight
    back into the packed (B, T, 3 * H * HD) layout."""
    bt = tl.program_id(0)
    which = tl.program_id(1)
    b = bt // T
    t = bt % T
    h = tl.arange(0, HB)[:, None]
    d = tl.arange(0, HALF)[None, :]
    mask = h < H
    off = b.to(tl.int64) * (H * T * HD) + h * (T * HD) + t * HD + d
    if which == 0:
        g1 = tl.load(DQ + off, mask=mask, other=0.0).to(tl.float32)
        g2 = tl.load(DQ + off + HALF, mask=mask, other=0.0).to(tl.float32)
    elif which == 1:
        g1 = tl.load(DK + off, mask=mask, other=0.0).to(tl.float32)
        g2 = tl.load(DK + off + HALF, mask=mask, other=0.0).to(tl.float32)
    else:
        g1 = tl.load(DV + off, mask=mask, other=0.0).to(tl.float32)
        g2 = tl.load(DV + off + HALF, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(COS + t * HD + d)
    s = tl.load(SIN + t * HD + d)
    rot = which < 2
    c = tl.where(rot, c, 1.0)
    s = tl.where(rot, s, 0.0)
    dx1 = g1 * c + g2 * s
    dx2 = g2 * c - g1 * s
    dst = DQKV + bt.to(tl.int64) * (3 * H * HD) + which * (H * HD) + h * HD + d
    tl.store(dst, dx1.to(DQKV.dtype.element_ty), mask=mask)
    tl.store(dst + HALF, dx2.to(DQKV.dtype.element_ty), mask=mask)


class _RopeQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, H):
        B, T, C3 = qkv.shape
        HD = C3 // (3 * H)
        assert HD * 3 * H == C3 and HD & (HD - 1) == 0, "head dim must be a power of two"
        qkv = qkv.contiguous()
        cos, sin = cos.float().contiguous(), sin.float().contiguous()
        out = torch.empty((3, B, H, T, HD), dtype=qkv.dtype, device=qkv.device)
        HB = triton.next_power_of_2(H)
        _rope_fwd_kernel[(B * T, 3)](qkv, cos, sin, out, T, B * H * T * HD, H,
                                     HD=HD, HALF=HD // 2, HB=HB, num_warps=4)
        ctx.save_for_backward(cos, sin)
        ctx.meta = (B, T, H, HD, qkv.dtype)
        return out[0], out[1], out[2]

    @staticmethod
    def backward(ctx, dq, dk, dv):
        cos, sin = ctx.saved_tensors
        B, T, H, HD, dtype = ctx.meta
        z = None
        grads = []
        for g in (dq, dk, dv):
            if g is None:
                z = z if z is not None else torch.zeros((B, H, T, HD), dtype=dtype, device=cos.device)
                g = z
            grads.append(g.contiguous())
        dqkv = torch.empty((B, T, 3 * H * HD), dtype=dtype, device=cos.device)
        HB = triton.next_power_of_2(H)
        _rope_bwd_kernel[(B * T, 3)](grads[0], grads[1], grads[2], cos, sin, dqkv, T, H,
                                     HD=HD, HALF=HD // 2, HB=HB, num_warps=4)
        return dqkv, None, None, None


def rope_qkv(qkv, cos, sin, H):
    """qkv: (B, T, 3*H*HD) packed linear output. cos, sin: (T, HD) float32, halves duplicated.
    Returns q, k, v each (B, H, T, HD) contiguous, with RoPE applied to q and k."""
    return _RopeQKV.apply(qkv, cos, sin, H)


# --------------------------------------------------------------------------
# 8.4 Chunked linear plus cross entropy
# --------------------------------------------------------------------------

CE_BLOCK = 4096
CE_CHUNK = 2048


@triton.jit
def _ce_fwd_bwd_kernel(LOGITS, TGT, LOSS, V, stride, inv_n, BLOCK: tl.constexpr):
    """One program per row. Online max and log sum exp over the vocab, then the loss,
    then overwrite the logits in place with their gradient (softmax - onehot) / N."""
    row = tl.program_id(0).to(tl.int64)
    base = LOGITS + row * stride
    m_vec = tl.full([BLOCK], -1e30, dtype=tl.float32)
    s_vec = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(base + cols, mask=mask, other=-1e30).to(tl.float32)
        new_m = tl.maximum(m_vec, x)
        s_vec = s_vec * tl.exp(m_vec - new_m) + tl.where(mask, tl.exp(x - new_m), 0.0)
        m_vec = new_m
    m = tl.max(m_vec, axis=0)
    lse = m + tl.log(tl.sum(s_vec * tl.exp(m_vec - m), axis=0))
    tgt = tl.load(TGT + row)
    x_t = tl.load(base + tgt).to(tl.float32)
    tl.store(LOSS + row, lse - x_t)
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        p = tl.where(cols == tgt, p - 1.0, p) * inv_n
        tl.store(base + cols, p.to(LOGITS.dtype.element_ty), mask=mask)


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, tgt, chunk):
        N, D = h.shape
        V = W.shape[0]
        dt = h.dtype
        h = h.contiguous()
        tgt = tgt.contiguous()
        Wc = W if W.dtype == dt else W.to(dt)        # one cast per step, like autocast would do
        dh = torch.empty_like(h)
        dW = torch.zeros((V, D), dtype=torch.float32, device=h.device)
        loss = torch.empty(N, dtype=torch.float32, device=h.device)
        inv_n = 1.0 / N
        for i in range(0, N, chunk):
            hc = h[i:i + chunk]
            n = hc.shape[0]
            logits = torch.mm(hc, Wc.t())               # (n, V) in the activation dtype
            _ce_fwd_bwd_kernel[(n,)](logits, tgt[i:i + n], loss[i:i + n], V, logits.stride(0), inv_n,
                                     BLOCK=CE_BLOCK, num_warps=8)
            torch.mm(logits, Wc, out=dh[i:i + n])      # dh_chunk = dlogits @ W
            if dt == torch.float32:
                dW.addmm_(logits.t(), hc)
            else:                                       # accumulate dW += dlogits^T @ h in float32
                torch.addmm(dW, logits.t(), hc, out_dtype=torch.float32, out=dW)
        ctx.save_for_backward(dh, dW)
        ctx.w_dtype = W.dtype
        return loss.sum() * inv_n

    @staticmethod
    def backward(ctx, g):
        dh, dW = ctx.saved_tensors
        return dh * g.to(dh.dtype), (dW * g).to(ctx.w_dtype), None, None


def fused_linear_cross_entropy(h, W, targets, chunk=CE_CHUNK):
    """Mean cross entropy of (h @ W^T) against targets without building the full (N, V) logits.
    h: (N, D) activations, W: (V, D) lm head weight, targets: (N,) int64.
    Gradients are computed during the forward pass and scaled by the incoming gradient."""
    return _FusedLinearCE.apply(h, W, targets, chunk)
