"""Correctness tests for the fused kernels, forward and every gradient.

Each kernel is compared against a float32 PyTorch reference computed from the same
inputs. Error is the relative norm ||out - ref|| / ||ref||. Tolerances: 1e-4 for
float32, 3e-2 for bf16, 5e-2 for the cross entropy in bf16.

Run as a script (exits non zero if any check fails) or with pytest:
  python kernels/test_kernels.py
  pytest kernels/test_kernels.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernels import triton_ops as T  # noqa: E402

TOL = {torch.float32: 1e-4, torch.bfloat16: 3e-2}
TOL_CE = {torch.float32: 1e-4, torch.bfloat16: 5e-2}
DEV = "cuda"


def _cuda_add_rmsnorm():
    """The CUDA C++ version, if it builds on this machine (section 9)."""
    try:
        from kernels.cuda_ext.add_rmsnorm_ext import add_rmsnorm
        return add_rmsnorm
    except Exception as e:  # not built yet or no compiler
        return e


# --------------------------------------------------------------------------
# float32 references
# --------------------------------------------------------------------------

def rmsnorm_ref(x, w, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def add_rmsnorm_ref(a, r, w, eps=1e-5):
    s = a.float() + r.float()
    return rmsnorm_ref(s, w, eps), s


def swiglu_ref(a, b):
    return F.silu(a.float()) * b.float()


def rope_ref(qkv, cos, sin, H):
    B, Tn, C3 = qkv.shape
    HD = C3 // (3 * H)
    q, k, v = qkv.float().view(B, Tn, 3, H, HD).permute(2, 0, 3, 1, 4)

    def rot(x):
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

    return rot(q), rot(k), v


def flce_ref(h, W, tgt):
    return F.cross_entropy(h.float() @ W.float().t(), tgt)


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

def rel_err(out, ref):
    out, ref = out.float(), ref.float()
    return ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def leaf(t, dtype):
    return t.detach().to(dtype).clone().requires_grad_(True)


def compare(name, kernel_fn, ref_fn, inputs, grad_idx, tol, out_dtypes=None):
    """Run kernel and reference, backprop the same random output gradients through both,
    and compare every output and every input gradient. Returns a list of result rows."""
    k_in = [leaf(x, x.dtype) if i in grad_idx else x for i, x in enumerate(inputs)]
    r_in = [leaf(x, torch.float32) if i in grad_idx else x for i, x in enumerate(inputs)]
    k_out = kernel_fn(*k_in)
    r_out = ref_fn(*r_in)
    k_out = k_out if isinstance(k_out, tuple) else (k_out,)
    r_out = r_out if isinstance(r_out, tuple) else (r_out,)
    g = torch.Generator(device=DEV).manual_seed(1)
    grads = [torch.randn(ro.shape, device=DEV, generator=g) if ro.dim() > 0 else torch.ones((), device=DEV)
             for ro in r_out]
    torch.autograd.backward(list(k_out), [gr.to(ko.dtype) for gr, ko in zip(grads, k_out)])
    torch.autograd.backward(list(r_out), grads)
    rows = []
    for j, (ko, ro) in enumerate(zip(k_out, r_out)):
        rows.append((name, f"out{j}" if len(k_out) > 1 else "out", rel_err(ko, ro), tol))
    for i in grad_idx:
        rows.append((name, f"grad{i}", rel_err(k_in[i].grad, r_in[i].grad), tol))
    return rows


def all_checks():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    rows = []
    cuda_impl = _cuda_add_rmsnorm()
    impls = {"triton": T.add_rmsnorm}
    if callable(cuda_impl):
        impls["cuda"] = cuda_impl

    # 8.1 RMSNorm and residual add plus RMSNorm
    for D in (768, 2048):
        for dt in (torch.float32, torch.bfloat16):
            x = torch.randn(4096, D, device=DEV)
            w = 1 + 0.1 * torch.randn(D, device=DEV)
            rows += compare(f"rmsnorm D={D} {_n(dt)}", lambda x, w: T.rmsnorm(x, w), rmsnorm_ref,
                            [x.to(dt), w], [0, 1], TOL[dt])
            a = torch.randn(4096, D, device=DEV)
            r = torch.randn(4096, D, device=DEV)
            for impl, fn in impls.items():
                rows += compare(f"add_rmsnorm[{impl}] D={D} {_n(dt)}", lambda a, r, w, fn=fn: fn(a, r, w),
                                add_rmsnorm_ref, [a.to(dt), r.to(dt), w], [0, 1, 2], TOL[dt])
        # Training case under bf16 autocast: a is bf16, the residual r is float32, y is bf16.
        a = torch.randn(4096, D, device=DEV).bfloat16()
        r = torch.randn(4096, D, device=DEV)
        for impl, fn in impls.items():
            rows += compare(f"add_rmsnorm[{impl}] D={D} mixed a=bf16 r=fp32",
                            lambda a, r, w, fn=fn: fn(a, r, w, out_dtype=torch.bfloat16),
                            add_rmsnorm_ref, [a, r, w], [0, 1, 2], TOL[torch.bfloat16])

    # 8.2 SwiGLU (second shape has a ragged tail)
    for shape in ((4096, 2048), (1000, 1375)):
        for dt in (torch.float32, torch.bfloat16):
            a = torch.randn(*shape, device=DEV).to(dt)
            b = torch.randn(*shape, device=DEV).to(dt)
            rows += compare(f"swiglu {shape} {_n(dt)}", T.swiglu, swiglu_ref, [a, b], [0, 1], TOL[dt])

    # 8.3 RoPE on packed qkv, two head configurations
    from bench_train import rope_tables
    for H, HD in ((12, 64), (16, 128)):
        Bn, Tn = 2, 256
        cos, sin = rope_tables(Tn, HD, 10000.0, DEV)
        for dt in (torch.float32, torch.bfloat16):
            qkv = torch.randn(Bn, Tn, 3 * H * HD, device=DEV).to(dt)
            rows += compare(f"rope_qkv H={H} HD={HD} {_n(dt)}", lambda q: T.rope_qkv(q, cos, sin, H),
                            lambda q: rope_ref(q, cos, sin, H), [qkv], [0], TOL[dt])

    # 8.4 Chunked linear plus cross entropy. N is not a multiple of the chunk size.
    N, D, V = 4096 + 37, 768, 32000
    tgt = torch.randint(0, V, (N,), device=DEV)
    for dt in (torch.float32, torch.bfloat16):
        h = torch.randn(N, D, device=DEV).to(dt)
        W = 0.02 * torch.randn(V, D, device=DEV)          # weight stays float32, as in the model
        rows += compare(f"linear_cross_entropy N={N} V={V} h={_n(dt)}",
                        lambda h, W: T.fused_linear_cross_entropy(h, W, tgt),
                        lambda h, W: flce_ref(h, W, tgt), [h, W], [0, 1], TOL_CE[dt])
    return rows, cuda_impl


def _n(dt):
    return {torch.float32: "fp32", torch.bfloat16: "bf16"}[dt]


def run():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    rows, cuda_impl = all_checks()
    fails = 0
    width = max(len(r[0]) for r in rows)
    for name, what, err, tol in rows:
        ok = err <= tol
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {what:<6} rel err {err:.2e}  (tol {tol:.0e})")
    if not callable(cuda_impl):
        print(f"SKIP  CUDA C++ add_rmsnorm not available: {str(cuda_impl).splitlines()[0][:120]}")
    print(f"\n{len(rows) - fails} of {len(rows)} checks passed")
    return 1 if fails else 0


def test_all_kernels():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
