"""Microbenchmarks for the fused kernels, forward plus backward.

For each op: eager PyTorch (the same code the torch model runs), torch.compile of that
reference, and the hand written kernel (Triton, plus the CUDA C++ add_rmsnorm when it
builds). Shapes are the real training shapes for --size, --batch and --seq under bf16
autocast. Memory bound ops report achieved bandwidth against the calibrated peak.
The cross entropy is compute bound, so it reports TFLOPs and peak memory instead.

  python kernels/bench_kernels.py --size small --batch 8 --seq 1024 --machine-label final
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from triton.testing import do_bench

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_train import RESULTS, SIZES, apply_rope, env_info, rope_tables  # noqa: E402
from kernels import triton_ops as T  # noqa: E402

DEV = "cuda"
BF16 = torch.bfloat16
EPS = 1e-5


# Eager references, written the way the torch model computes them under bf16 autocast.
def rmsnorm_eager(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w).to(BF16)


def add_rmsnorm_eager(a, r, w):
    s = r + a
    return rmsnorm_eager(s, w), s


def swiglu_eager(a, b):
    return F.silu(a) * b


def rope_eager(qkv, cos, sin, H):
    B, Tn, C3 = qkv.shape
    q, k, v = qkv.view(B, Tn, 3, H, C3 // (3 * H)).permute(2, 0, 3, 1, 4)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v


def ce_eager(h, W, tgt):
    with torch.autocast("cuda", dtype=BF16):
        logits = h @ W.t()
    return F.cross_entropy(logits.float(), tgt)


def ce_fused(h, W, tgt):
    return T.fused_linear_cross_entropy(h, W, tgt)


def make_step(fn, inputs, grad_idx):
    """A closure that runs forward plus backward once with fixed output gradients."""
    leaves = [x.detach().clone().requires_grad_(True) if i in grad_idx else x for i, x in enumerate(inputs)]
    out = fn(*leaves)
    outs = out if isinstance(out, tuple) else (out,)
    grads = [torch.randn_like(o) if o.dim() > 0 else torch.ones_like(o) for o in outs]

    def step():
        for i in grad_idx:
            leaves[i].grad = None
        o = fn(*leaves)
        o = o if isinstance(o, tuple) else (o,)
        torch.autograd.backward(list(o), grads)

    return step


def peak_mem_gb(step):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small", choices=list(SIZES))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--machine-label", default="dev")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    info = env_info(args.machine_label)
    cfg = SIZES[args.size]
    N, D, H, HD, FF, V = args.batch * args.seq, cfg.d, cfg.H, cfg.hd, cfg.ffn, cfg.vocab
    cal_path = RESULTS / f"calibration_{args.machine_label}.json"
    peak_bw = json.loads(cal_path.read_text())["bandwidth_gbs"] if cal_path.exists() else None
    peak_bf16 = json.loads(cal_path.read_text())["matmul"]["bf16"]["tflops"] if cal_path.exists() else None
    print(f"{info['arch']} ({args.machine_label}) | {args.size}: N={N} tokens, d={D}, ffn={FF}, H={H}, V={V}"
          f" | calibrated bandwidth {peak_bw} GB/s")

    try:
        from kernels.cuda_ext.add_rmsnorm_ext import add_rmsnorm as cuda_add_rmsnorm
    except Exception as e:
        cuda_add_rmsnorm = None
        print(f"CUDA C++ add_rmsnorm not available: {str(e).splitlines()[0][:100]}")

    g = torch.Generator(device=DEV).manual_seed(0)
    rn = lambda *s, dt=BF16: torch.randn(*s, device=DEV, generator=g).to(dt)
    w = 1 + 0.1 * torch.randn(D, device=DEV, generator=g)
    cos, sin = rope_tables(args.seq, HD, 10000.0, DEV)
    tgt = torch.randint(0, V, (N,), device=DEV, generator=g)

    # name, impls, inputs, grad_idx, bytes moved fwd+bwd (minimum, ignoring the tiny weight), flops
    ops = [
        ("rmsnorm (x fp32 -> bf16)",
         {"eager": rmsnorm_eager, "triton": lambda x, w: T.rmsnorm(x, w, EPS, BF16)},
         [rn(N, D, dt=torch.float32), w], [0, 1],
         N * D * (4 + 2 + 2 + 4 + 4), None),
        ("add_rmsnorm (a bf16 + r fp32)",
         {"eager": add_rmsnorm_eager, "triton": lambda a, r, w: T.add_rmsnorm(a, r, w, EPS, BF16),
          **({"cuda": lambda a, r, w: cuda_add_rmsnorm(a, r, w, EPS, BF16)} if cuda_add_rmsnorm else {})},
         [rn(N, D), rn(N, D, dt=torch.float32), w], [0, 1, 2],
         N * D * (2 + 4 + 2 + 4 + 2 + 4 + 4 + 2 + 4), None),
        ("swiglu",
         {"eager": swiglu_eager, "triton": T.swiglu},
         [rn(N, FF), rn(N, FF)], [0, 1],
         N * FF * 2 * (3 + 5), None),
        ("rope_qkv",
         {"eager": lambda q: rope_eager(q, cos, sin, H), "triton": lambda q: T.rope_qkv(q, cos, sin, H)},
         [rn(args.batch, args.seq, 3 * D)], [0],
         N * 3 * D * 2 * 4, None),
        ("linear + cross entropy",
         {"eager": ce_eager, "triton": ce_fused},
         [rn(N, D), 0.02 * torch.randn(V, D, device=DEV, generator=g), tgt], [0, 1],
         None, 6.0 * N * D * V),
    ]

    rows = []
    for name, impls, inputs, grad_idx, nbytes, flops in ops:
        fns = dict(impls)
        if not args.no_compile:
            fns = {"eager": impls["eager"], "torch.compile": torch.compile(impls["eager"]),
                   **{k: v for k, v in impls.items() if k != "eager"}}
        res = {}
        for impl, fn in fns.items():
            try:
                step = make_step(fn, inputs, grad_idx)
                # do_bench clears L2 before every timed call and returns the median, so results
                # reflect DRAM traffic, as in a real step where other layers evict these tensors.
                sec = do_bench(step, warmup=50, rep=300, return_mode="median") / 1e3
                r = {"ms": round(sec * 1e3, 4), "peak_mem_gb": round(peak_mem_gb(step), 3)}
                if nbytes:
                    r["gbs"] = round(nbytes / sec / 1e9, 1)
                    r["pct_peak_bw"] = round(100 * nbytes / sec / 1e9 / peak_bw, 1) if peak_bw else None
                if flops:
                    r["tflops"] = round(flops / sec / 1e12, 1)
                    r["pct_peak_bf16"] = round(100 * flops / sec / 1e12 / peak_bf16, 1) if peak_bf16 else None
                res[impl] = r
            except Exception as e:
                res[impl] = {"error": f"{type(e).__name__}: {e}".splitlines()[0][:200]}
            torch.cuda.empty_cache()
        eager_ms = res["eager"].get("ms")
        for impl, r in res.items():
            if "ms" in r and eager_ms:
                r["speedup_vs_eager"] = round(eager_ms / r["ms"], 2)
            if "ms" in r and res.get("torch.compile", {}).get("ms"):
                r["speedup_vs_compile"] = round(res["torch.compile"]["ms"] / r["ms"], 2)
        rows.append({"op": name, "bytes": nbytes, "flops": flops, "results": res})

    from tabulate import tabulate
    table = []
    for row in rows:
        for impl, r in row["results"].items():
            if "error" in r:
                table.append([row["op"], impl, "error", r["error"][:50], "", "", "", ""])
                continue
            perf = (f"{r['gbs']} GB/s ({r['pct_peak_bw']}%)" if "gbs" in r
                    else f"{r['tflops']} TFLOPs ({r['pct_peak_bf16']}%)")
            table.append([row["op"], impl, r["ms"], perf, r.get("speedup_vs_eager", ""),
                          r.get("speedup_vs_compile", ""), r["peak_mem_gb"]])
    print(tabulate(table, headers=["op (fwd+bwd)", "impl", "ms", "achieved (of peak)", "vs eager",
                                   "vs compile", "extra mem GB"]))

    out = {**info, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "size": args.size, "batch": args.batch,
           "seq": args.seq, "tokens": N, "peak_bandwidth_gbs": peak_bw, "peak_bf16_tflops": peak_bf16, "ops": rows}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"bench_kernels_{args.machine_label}_{info['arch']}_{args.size}_b{args.batch}_s{args.seq}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
