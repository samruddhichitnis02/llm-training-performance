"""Tests for the fp8 and NVFP4 linear layers in kernels/lowp.py.

1. NVFP4 quantizer: round trip error, every value lands on the e2m1 grid, and
   stochastic rounding is unbiased (its mean over many draws matches the input).
2. NVFP4 GEMM: _scaled_mm on the quantized operands equals a float32 matmul of the
   dequantized operands. This checks the packing, nibble order and scale layout.
3. Linear layers: forward output and both gradients against a bf16 reference.
   The tolerances are the error the format itself produces on Gaussian data, not
   kernel error: about 3 to 4% for e4m3 and about 10 to 15% for e2m1.

  python kernels/test_lowp.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernels import lowp  # noqa: E402

DEV = "cuda"


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _n(dt):
    return {torch.float32: "fp32", torch.bfloat16: "bf16"}[dt]


def run():
    if not lowp.fp8_supported():
        print("fp8 not supported on this GPU, nothing to test")
        return 0
    torch.manual_seed(0)
    rows = []

    def check(name, err, tol):
        rows.append((name, err, tol))

    if lowp.fp4_supported():
        x = torch.randn(256, 512, device=DEV)
        packed, bscale, g = lowp._fp4_quant_impl(x)
        deq = lowp.fp4_dequant(packed, bscale, g)
        check("nvfp4 round trip (round to nearest)", rel(deq, x), 0.12)
        grid = torch.tensor(lowp.FP4_GRID, device=DEV)
        vals = deq.view(256, -1, 16) / (bscale.float() * g)[..., None]
        on_grid = (vals.abs()[..., None] - grid).abs().min(-1).values.max().item()
        check("nvfp4 values on the e2m1 grid (max distance)", on_grid, 1e-3)

        xs = torch.randn(128, 64, device=DEV)
        acc = torch.zeros_like(xs)
        draws = 400
        for _ in range(draws):
            p, s, gg = lowp._fp4_quant_impl(xs, torch.rand_like(xs))
            acc += lowp.fp4_dequant(p, s, gg)
        check("nvfp4 stochastic rounding is unbiased (mean of 400 draws)", rel(acc / draws, xs), 0.02)

        # The Triton quantizer must produce exactly the reference bits, normal and transposed.
        for dt in (torch.float32, torch.bfloat16):
            x = torch.randn(512, 768, device=DEV).to(dt)
            for transpose in (False, True):
                src = x.t().contiguous() if transpose else x
                rp, rs, rg = lowp._fp4_quant_blocked_impl(src)
                tp, ts, tg = lowp.nvfp4_quantize(x, transpose=transpose)
                mism = (rp != tp).float().mean().item() + (rs.view(torch.uint8) != ts.view(torch.uint8)).float().mean().item()
                check(f"nvfp4 Triton quantizer = reference bits ({_n(dt)}, transpose={transpose})", mism, 1e-4)
        xs = torch.randn(256, 128, device=DEV)
        acc = torch.zeros_like(xs)
        for _ in range(draws):
            p, s, gg = lowp.nvfp4_quantize(xs, stochastic=True)
            pr, sr, _ = lowp._fp4_quant_impl(xs)
            acc += lowp.fp4_dequant(p, sr, gg)
        check("nvfp4 Triton stochastic rounding is unbiased (400 draws)", rel(acc / draws, xs), 0.02)

        for M, N, K in ((256, 384, 512), (1024, 768, 2048)):
            a = torch.randn(M, K, device=DEV)
            b = torch.randn(N, K, device=DEV)
            qa, sa, ga = lowp._fp4_quant_blocked_impl(a)
            qb, sb, gb = lowp._fp4_quant_blocked_impl(b)
            out = lowp._mm4(qa, qb, sa, sb, ga, gb, torch.float32)
            pa, psa, _ = lowp._fp4_quant_impl(a)
            pb, psb, _ = lowp._fp4_quant_impl(b)
            ref = lowp.fp4_dequant(pa, psa, ga) @ lowp.fp4_dequant(pb, psb, gb).t()
            check(f"nvfp4 GEMM {M}x{N}x{K} vs dequantized fp32 matmul", rel(out, ref), 1e-3)

    tol = {"fp8": (0.06, 0.08), "fp4": (0.18, 0.25)}     # (forward, gradients)
    for fmt in ("fp8", "fp4"):
        if fmt == "fp4" and not lowp.fp4_supported():
            continue
        for M, K, N in ((2048, 768, 2304), (4096, 1024, 2816)):
            lin = lowp.LowPrecisionLinear(K, N, fmt).to(DEV)
            torch.nn.init.normal_(lin.weight, std=0.02)
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, requires_grad=True)
            dy = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)
            y = lin(x)
            y.backward(dy)
            xr = x.detach().float().requires_grad_(True)
            wr = lin.weight.detach().clone().requires_grad_(True)
            yr = xr @ wr.t()
            yr.backward(dy.float())
            check(f"{fmt} linear {M}x{K}->{N} forward", rel(y, yr), tol[fmt][0])
            check(f"{fmt} linear {M}x{K}->{N} grad input", rel(x.grad, xr.grad), tol[fmt][1])
            check(f"{fmt} linear {M}x{K}->{N} grad weight", rel(lin.weight.grad, wr.grad), tol[fmt][1])

    fails = 0
    width = max(len(r[0]) for r in rows)
    for name, err, t in rows:
        ok = err <= t
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {err:.2e}  (tol {t:.0e})")
    print(f"\n{len(rows) - fails} of {len(rows)} checks passed")
    return 1 if fails else 0


def test_lowp():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
