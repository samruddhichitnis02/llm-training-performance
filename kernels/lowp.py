"""Low precision linear layers for training, built on torch._scaled_mm.

fp8:   all three GEMMs (forward, input gradient, weight gradient) in e4m3 with
       tensorwise current scaling: scale = 448 / amax of the tensor, computed every step.
       Because the scale is per tensor, a quantized tensor can be transposed and reused,
       so x and w are saved in fp8 for the backward pass (half the bf16 memory).

fp4:   NVFP4, the Blackwell 4 bit format. Values are e2m1 (magnitudes 0, 0.5, 1, 1.5,
       2, 3, 4, 6), packed two per byte. Every 16 values along the contraction dimension
       share one e4m3 scale, and one fp32 scale per tensor keeps the block scales inside
       the e4m3 range. Blocks run along the contraction dimension, so each GEMM needs its
       operands quantized along its own contraction axis:
         forward  y  = x @ w^T     blocks along K (in features)
         dgrad    dx = dy @ w      blocks along N (out features)
         wgrad    dw = dy^T @ x    blocks along M (tokens)
       Gradients (dy) use stochastic rounding, as in NVIDIA's NVFP4 training recipe, so
       rounding errors average out instead of biasing the update. x and w use round to
       nearest. Not included from that recipe: random Hadamard transforms on the wgrad
       inputs and 2D 16x16 weight blocks.

The lm head is not replaced: it stays bf16 inside the chunked cross entropy kernel,
as low precision recipes keep the last layer in higher precision.
"""

import torch
import torch.nn as nn

# Each quantizer is compiled once per tensor shape. A model has about a dozen distinct
# shapes (x, w and their transposes for every linear), more than the default limit of 8.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)

E4M3_MAX = 448.0
FP4_MAX = 6.0
FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BLOCK = 16


def fp4_supported():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0) \
        and hasattr(torch, "float4_e2m1fn_x2")


def fp8_supported():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


# --------------------------------------------------------------------------
# FP8
# --------------------------------------------------------------------------

def _fp8_quant_impl(x):
    """Tensorwise e4m3 quantization. Returns (q, inverse scale) for _scaled_mm."""
    amax = x.abs().amax().float().clamp_min(1e-12)
    scale = E4M3_MAX / amax
    q = (x.float() * scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale.reciprocal()


def _fp8_quant_t_impl(x):
    """Same, plus the transposed copy that the backward GEMMs need."""
    q, inv = _fp8_quant_impl(x)
    return q, q.t().contiguous(), inv


_fp8_quant = torch.compile(_fp8_quant_impl, dynamic=False)
_fp8_quant_t = torch.compile(_fp8_quant_t_impl, dynamic=False)


def _mm8(a, b_colmajor, sa, sb, out_dtype=torch.bfloat16):
    return torch._scaled_mm(a, b_colmajor, scale_a=sa, scale_b=sb, out_dtype=out_dtype)


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, x, w):
        ctx.x_dtype = x.dtype
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, xq_t, xs = _fp8_quant_t(x2)            # (M, K), (K, M)
        wq, wq_t, ws = _fp8_quant_t(w)             # (N, K), (K, N)
        y = _mm8(xq, wq.t(), xs, ws)               # b = (K, N) column major
        ctx.save_for_backward(xq_t, xs, wq_t, ws)
        ctx.shape = shape
        return y.view(*shape[:-1], w.shape[0])

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dy):
        xq_t, xs, wq_t, ws = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1])
        dyq, dyq_t, dys = _fp8_quant_t(dy2)        # (M, N), (N, M)
        dx = _mm8(dyq, wq_t.t(), dys, ws)          # (M, N) @ (N, K), b column major
        dw = _mm8(dyq_t, xq_t.t(), dys, xs, torch.float32)   # (N, M) @ (M, K), fp32 out for the fp32 weight
        return dx.view(ctx.shape).to(ctx.x_dtype), dw


# --------------------------------------------------------------------------
# NVFP4
# --------------------------------------------------------------------------

def _to_blocked(s):
    """Rearrange an (R, C) block scale matrix into the 128 x 4 tiled layout that the
    Blackwell block scaled GEMM reads. R must be a multiple of 128 and C of 4."""
    R, C = s.shape
    blocks = s.view(R // 128, 128, C // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _fp4_quant_impl(x, noise=None):
    """NVFP4 quantization of x (R, K) with 16 wide blocks along K.

    Returns packed uint8 (R, K/2), block scales e4m3 (R, K/16) in row layout, and the
    fp32 per tensor decode scale. If noise (uniform [0, 1), same shape as x) is given,
    values are rounded stochastically between the two neighbouring grid points.
    """
    R, K = x.shape
    xf = x.float()
    g = (xf.abs().amax() / (FP4_MAX * E4M3_MAX)).clamp_min(1e-30)
    blocks = xf.view(R, K // BLOCK, BLOCK)
    bamax = blocks.abs().amax(-1)
    bscale = (bamax / (FP4_MAX * g)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    denom = bscale.float() * g
    v = torch.where(denom[..., None] > 0, blocks / denom[..., None], 0.0).clamp(-FP4_MAX, FP4_MAX)
    mag = v.abs()
    grid = torch.tensor(FP4_GRID, device=x.device)
    lo_idx = (mag[..., None] >= grid[1:]).sum(-1)                 # 0..7
    hi_idx = (lo_idx + 1).clamp(max=7)
    lo, hi = grid[lo_idx], grid[hi_idx]
    frac = torch.where(hi > lo, (mag - lo) / (hi - lo), 0.0)
    if noise is None:
        up = frac > 0.5
    else:
        up = noise.view(R, K // BLOCK, BLOCK) < frac
    idx = torch.where(up, hi_idx, lo_idx)
    code = (idx | ((v < 0) & (idx > 0)).to(idx.dtype) << 3).to(torch.uint8).view(R, K // 2, 2)
    packed = code[..., 0] | (code[..., 1] << 4)                   # first value in the low nibble
    return packed, bscale, g


def _fp4_quant_blocked_impl(x, noise=None):
    """PyTorch reference quantizer with the scales in GEMM layout (used by the tests)."""
    packed, bscale, g = _fp4_quant_impl(x, noise)
    return packed, _to_blocked(bscale), g


# --------------------------------------------------------------------------
# NVFP4 quantizer as one Triton kernel
# --------------------------------------------------------------------------
# The torch.compile version of _fp4_quant_impl plus the transposes it needs took 78 ms of
# a 121 ms step on `small` (the FP4 GEMMs themselves took under 5 ms). This kernel does
# the whole job in one pass: it reads the input through strides (so x^T needs no copy),
# computes block amax and the e4m3 block scale, rounds to e2m1 (nearest, or stochastic
# with noise drawn inside the kernel), packs two values per byte, and writes the block
# scales straight into the 128 x 4 tiled layout of the block scaled GEMM.

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _e2m1_value(i):
    """Magnitude of e2m1 code i (0..7): 0, 0.5, 1, 1.5, 2, 3, 4, 6."""
    return tl.where(i < 4, i.to(tl.float32) * 0.5,
                    tl.where(i == 4, 2.0, tl.where(i == 5, 3.0, tl.where(i == 6, 4.0, 6.0))))


@triton.jit
def _e2m1_code(xb, inv, u, STOCHASTIC: tl.constexpr):
    """4 bit e2m1 codes for values xb (BR, NB, 8) with per block inverse scale inv (BR, NB)."""
    v = xb * inv[:, :, None]
    v = tl.minimum(tl.maximum(v, -6.0), 6.0)
    mag = tl.abs(v)
    if STOCHASTIC:
        lo = ((mag >= 0.5).to(tl.int32) + (mag >= 1.0).to(tl.int32) + (mag >= 1.5).to(tl.int32)
              + (mag >= 2.0).to(tl.int32) + (mag >= 3.0).to(tl.int32) + (mag >= 4.0).to(tl.int32)
              + (mag >= 6.0).to(tl.int32))
        hi = tl.minimum(lo + 1, 7)
        lo_v = _e2m1_value(lo)
        hi_v = _e2m1_value(hi)
        frac = tl.where(hi_v > lo_v, (mag - lo_v) / (hi_v - lo_v), 0.0)
        idx = tl.where(u < frac, hi, lo)
    else:
        # Round to nearest without a division: count the midpoints between grid values that
        # mag is strictly above. Ties round down, the same as frac > 0.5 in the reference.
        idx = ((mag > 0.25).to(tl.int32) + (mag > 0.75).to(tl.int32) + (mag > 1.25).to(tl.int32)
               + (mag > 1.75).to(tl.int32) + (mag > 2.5).to(tl.int32) + (mag > 3.5).to(tl.int32)
               + (mag > 5.0).to(tl.int32))
    return idx | (((v < 0) & (idx > 0)).to(tl.int32) << 3)


@triton.jit
def _nvfp4_quant_kernel(X, G, OUT, SCALE, R, K, stride_r, stride_k, seed,
                        STOCHASTIC: tl.constexpr, BR: tl.constexpr, NB: tl.constexpr):
    # Even and odd columns are loaded as two separate tiles, so the two values that share
    # a byte sit at the same tile position and pack with no reshuffle. (A first version
    # paired them with reshape + split and got wrong bits for contiguous fp32 input.)
    pid_r = tl.program_id(0)
    pid_k = tl.program_id(1)
    BK: tl.constexpr = NB * 16
    HALF: tl.constexpr = BK // 2
    rows = pid_r * BR + tl.arange(0, BR)
    pcols = pid_k * HALF + tl.arange(0, HALF)                   # packed byte columns
    ce = 2 * pcols                                              # even source columns
    base = X + rows[:, None].to(tl.int64) * stride_r
    xe = tl.load(base + ce[None, :].to(tl.int64) * stride_k).to(tl.float32)
    xo = tl.load(base + (ce + 1)[None, :].to(tl.int64) * stride_k).to(tl.float32)
    g = tl.load(G)                                              # per tensor decode scale

    xe = tl.reshape(xe, (BR, NB, 8))
    xo = tl.reshape(xo, (BR, NB, 8))
    bamax = tl.maximum(tl.max(tl.abs(xe), axis=2), tl.max(tl.abs(xo), axis=2))   # (BR, NB)
    s8 = tl.minimum(bamax / (6.0 * g), 448.0).to(tl.float8e4nv)
    denom = s8.to(tl.float32) * g
    inv = tl.where(denom > 0, 1.0 / denom, 0.0)
    if STOCHASTIC:
        offs = rows[:, None] * K + ce[None, :]
        ue = tl.reshape(tl.rand(seed, offs), (BR, NB, 8))
        uo = tl.reshape(tl.rand(seed, offs + 1), (BR, NB, 8))
    else:
        ue = xe
        uo = xo
    c0 = tl.reshape(_e2m1_code(xe, inv, ue, STOCHASTIC), (BR, HALF))
    c1 = tl.reshape(_e2m1_code(xo, inv, uo, STOCHASTIC), (BR, HALF))
    packed = (c0 | (c1 << 4)).to(tl.uint8)                      # first value in the low nibble
    tl.store(OUT + rows[:, None].to(tl.int64) * (K // 2) + pcols[None, :], packed)

    # Block scale for (row r, block column c) goes to the tiled GEMM layout:
    # ((r // 128) * (C // 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    C = K // 16
    bcols = pid_k * NB + tl.arange(0, NB)
    r2 = rows[:, None]
    c2 = bcols[None, :]
    soff = ((r2 // 128) * (C // 4) + c2 // 4) * 512 + (r2 % 32) * 16 + ((r2 % 128) // 32) * 4 + c2 % 4
    tl.store(SCALE + soff.to(tl.int64), s8.to(tl.uint8, bitcast=True))


@triton.jit
def _amax_kernel(X, OUT, n, BLOCK: tl.constexpr):
    """abs max of a contiguous tensor in one pass, one atomic per program. Non negative
    floats order the same as their int32 bit patterns, so the atomic is a native integer
    max instead of a compare and swap loop on a float."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < n, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(x), axis=0)
    tl.atomic_max(OUT, m.to(tl.int32, bitcast=True))


def nvfp4_global_scale(x):
    """Per tensor decode scale g = amax / (6 * 448), as a 1 element fp32 tensor (no host sync)."""
    assert x.is_contiguous()
    amax = torch.zeros(1, dtype=torch.int32, device=x.device)
    n = x.numel()
    BLOCK = 4096
    _amax_kernel[(triton.cdiv(n, BLOCK),)](x, amax, n, BLOCK=BLOCK, num_warps=8)
    return (amax.view(torch.float32) / (FP4_MAX * E4M3_MAX)).clamp_min_(1e-30)


_seed_counter = [0]


def nvfp4_quantize(x, stochastic=False, transpose=False, g=None):
    """NVFP4 quantize x (R, K) along K, or x^T along x's rows if transpose=True, in one pass.

    g: per tensor scale from nvfp4_global_scale(x). Pass it in when quantizing the same
    tensor twice (normal and transposed) so the amax pass runs once.
    Returns packed uint8 (rows, cols/2), block scales (e4m3) in the GEMM layout, and g."""
    if transpose:
        K, R = x.shape
        stride_r, stride_k = x.stride(1), x.stride(0)
    else:
        R, K = x.shape
        stride_r, stride_k = x.stride(0), x.stride(1)
    assert R % 128 == 0 and K % 64 == 0, "NVFP4 path needs dims that are multiples of 128 and 64"
    if g is None:
        g = nvfp4_global_scale(x)
    out = torch.empty((R, K // 2), dtype=torch.uint8, device=x.device)
    scale = torch.empty(R * (K // 16), dtype=torch.float8_e4m3fn, device=x.device)
    _seed_counter[0] += 1
    # Tile sizes picked by a sweep on a (4096, 2816) bf16 tensor on sm_120 (see LOG.md, milestone 5).
    BR, NB = _QUANT_TILES[(transpose, stochastic)]
    if R % BR or K % (16 * NB):
        BR, NB = 32, 4
    _nvfp4_quant_kernel[(R // BR, K // (16 * NB))](x, g, out, scale.view(torch.uint8), R, K, stride_r, stride_k,
                                                   _seed_counter[0], STOCHASTIC=stochastic, BR=BR, NB=NB,
                                                   num_warps=4)
    return out, scale, g


_QUANT_TILES = {(False, False): (64, 4), (False, True): (32, 4), (True, False): (32, 8), (True, True): (16, 8)}


def fp4_dequant(packed, bscale, g):
    """Reference decode back to float32 (used by the tests)."""
    grid = torch.tensor(FP4_GRID, device=packed.device)
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    codes = torch.stack([lo, hi], dim=-1).flatten(-2)
    vals = grid[codes & 7] * torch.where((codes & 8) > 0, -1.0, 1.0)
    R, K = vals.shape
    return (vals.view(R, K // BLOCK, BLOCK) * (bscale.float() * g)[..., None]).view(R, K)


def _mm4(a, b, sa, sb, ga, gb, out_dtype=torch.bfloat16):
    """a (M, K) and b (N, K), both quantized along K. Returns a @ b^T.
    The block scales are applied inside the GEMM; the two per tensor scales after it."""
    out = torch._scaled_mm(a.view(torch.float4_e2m1fn_x2), b.view(torch.float4_e2m1fn_x2).t(),
                           scale_a=sa, scale_b=sb, out_dtype=out_dtype)
    return out.mul_((ga * gb).to(out.dtype))


class _FP4Linear(torch.autograd.Function):
    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, x, w):
        ctx.x_dtype = x.dtype
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        xg = nvfp4_global_scale(x2)
        wg = nvfp4_global_scale(w)
        xq, xs, _ = nvfp4_quantize(x2, g=xg)                     # (M, K), blocks along K
        wq, ws, _ = nvfp4_quantize(w, g=wg)                      # (N, K)
        y = _mm4(xq, wq, xs, ws, xg, wg)
        xtq, xts, xtg = nvfp4_quantize(x2, transpose=True, g=xg)  # (K, M), blocks along M, for wgrad
        ctx.save_for_backward(xtq, xts, xtg, w, wg)
        ctx.shape = shape
        return y.view(*shape[:-1], w.shape[0])

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dy):
        xtq, xts, xtg, w, wg = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).contiguous()
        dyg = nvfp4_global_scale(dy2)
        dyq, dys, _ = nvfp4_quantize(dy2, stochastic=True, g=dyg)                   # (M, N), along N
        wtq, wts, wtg = nvfp4_quantize(w, transpose=True, g=wg)                     # (K, N), along N
        dx = _mm4(dyq, wtq, dys, wts, dyg, wtg)                                     # (M, K)
        dytq, dyts, dytg = nvfp4_quantize(dy2, stochastic=True, transpose=True, g=dyg)  # (N, M), along M
        dw = _mm4(dytq, xtq, dyts, xts, dytg, xtg, torch.float32)              # (N, K)
        return dx.view(ctx.shape).to(ctx.x_dtype), dw


# --------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------

class LowPrecisionLinear(nn.Linear):
    """nn.Linear (no bias) whose three GEMMs run in fp8 or NVFP4. Weights stay fp32."""

    def __init__(self, in_features, out_features, fmt):
        super().__init__(in_features, out_features, bias=False)
        if fmt == "fp8" and not fp8_supported():
            raise RuntimeError("fp8 GEMMs need compute capability 8.9 or newer")
        if fmt == "fp4" and not fp4_supported():
            raise RuntimeError("NVFP4 GEMMs need Blackwell (compute capability 10.0 or newer)")
        if fmt == "fp4" and (in_features % 128 or out_features % 128):
            raise ValueError("NVFP4 path needs feature sizes that are multiples of 128")
        self.fmt = fmt

    def forward(self, x):
        fn = _FP8Linear if self.fmt == "fp8" else _FP4Linear
        return fn.apply(x, self.weight)

    def extra_repr(self):
        return super().extra_repr() + f", fmt={self.fmt}"


def make_linear(in_features, out_features, fmt="bf16"):
    if fmt in (None, "bf16"):
        return nn.Linear(in_features, out_features, bias=False)
    return LowPrecisionLinear(in_features, out_features, fmt)
