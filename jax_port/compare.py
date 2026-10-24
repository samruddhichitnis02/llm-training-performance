"""Compare the JAX port with the PyTorch model on the same shapes.

Runs in the PyTorch venv and calls the JAX venv as a subprocess (the two need different
NVIDIA library pins). Three parts:

1. Parity: one transformer block from bench_train.py, bf16 autocast, random fp32 input.
   Its weights, input, output gradient, output and all gradients are saved to an npz;
   the JAX block (jax_port/block_jax.py --parity) loads the same weights and must match
   the output and every gradient to bf16 tolerance (3e-2 relative).
2. One block, forward plus backward on a (B, T, d) input, timed the same way in both
   frameworks (median of synchronized steps): PyTorch eager, torch.compile, the Triton
   kernels, and JAX jit.
3. Full training step (forward, backward, AdamW): PyTorch ladder rows 4 (eager, fused
   Adam), 5 (torch.compile) and 6 (Triton kernels), measured here in this session, and
   the JAX step from block_jax.py. Plus the kernel bucket breakdown of each, from the
   PyTorch profiler and the JAX profiler.

  python jax_port/compare.py --size medium --batch 4 --seq 1024 --machine-label final
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench_train import (RESULTS, SIZES, Block, cap_gpu_memory, env_info, load_ops,  # noqa: E402
                         measure_matmul_tflops, rope_tables, run_one)

JAX_PY = os.environ.get("JAX_PYTHON", str(Path.home() / "venvs/jax/bin/python"))
TOL = 3e-2


def jax_run(args, timeout=1800):
    p = subprocess.run([JAX_PY, str(ROOT / "jax_port" / "block_jax.py")] + args, capture_output=True, text=True,
                       timeout=timeout, cwd=ROOT)
    if p.returncode != 0:
        raise RuntimeError(f"JAX run failed: {p.stderr.strip().splitlines()[-1] if p.stderr else p.returncode}")
    return p.stdout


def parity(cfg, B, T, attn):
    torch.manual_seed(0)
    blk = Block(cfg, "sdpa", None).cuda()
    for m in blk.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
    with torch.no_grad():   # non trivial norm weights so their gradients are tested too
        blk.norm1.weight.copy_(1 + 0.1 * torch.randn(cfg.d, device="cuda"))
        blk.norm2.weight.copy_(1 + 0.1 * torch.randn(cfg.d, device="cuda"))
    x = torch.randn(B, T, cfg.d, device="cuda", requires_grad=True)
    dy = torch.randn(B, T, cfg.d, device="cuda")
    cos, sin = rope_tables(T, cfg.hd, cfg.rope_base, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = blk(x, cos, sin)
    y.backward(dy)
    lin = {"qkv": blk.attn.qkv, "wo": blk.attn.wo, "w1": blk.mlp.w1, "w3": blk.mlp.w3, "w2": blk.mlp.w2}
    arrays = {"H": np.array(cfg.H), "x": x.detach().cpu().numpy(), "dy": dy.cpu().numpy(),
              "y": y.detach().float().cpu().numpy(), "g_x": x.grad.cpu().numpy()}
    for k, m in lin.items():            # PyTorch stores (out, in); the JAX port uses (in, out)
        arrays[f"w_{k}"] = m.weight.detach().t().contiguous().cpu().numpy()
        arrays[f"g_{k}"] = m.weight.grad.t().contiguous().cpu().numpy()
    for k, m in (("norm1", blk.norm1), ("norm2", blk.norm2)):
        arrays[f"w_{k}"] = m.weight.detach().cpu().numpy()
        arrays[f"g_{k}"] = m.weight.grad.cpu().numpy()
    path = Path(tempfile.gettempdir()) / "jax_parity.npz"
    np.savez(path, **arrays)
    rows = json.loads(jax_run(["--parity", str(path), "--attn", attn]).strip().splitlines()[-1])
    return {k: {"rel_err": v, "pass": v <= TOL} for k, v in rows.items()}


def time_torch_block(cfg, B, T, kernels, compile_, warmup=5, steps=20):
    torch.manual_seed(0)
    ops = load_ops(kernels)
    blk = Block(cfg, "sdpa", ops).cuda()
    f = torch.compile(blk) if compile_ else blk
    x = torch.randn(B, T, cfg.d, device="cuda", requires_grad=True)
    dy = torch.randn(B, T, cfg.d, device="cuda")
    cos, sin = rope_tables(T, cfg.hd, cfg.rope_base, "cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = f(x, cos, sin)
        torch.autograd.backward(y, dy)

    for _ in range(warmup):
        step()
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    torch._dynamo.reset()
    return statistics.median(times) * 1e3


def gemm_compare(cfg, B, T):
    """The model's GEMM shapes, timed in PyTorch (cuBLAS through torch.matmul) and in XLA
    (inside jit), both without clearing L2, median of synchronized calls."""
    N, d, f = B * T, cfg.d, cfg.ffn
    shapes = [(N, 3 * d, d, "qkv forward"), (N, d, d, "wo forward"), (N, f, d, "w1/w3 forward"),
              (N, d, f, "w2 forward"), (N, d, f, "w1/w3 input gradient"), (f, d, N, "w1/w3 weight gradient"),
              (3 * d, d, N, "qkv weight gradient"), (N, cfg.vocab, d, "lm head")]
    rows = []
    for M, Nn, K, name in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, Nn, device="cuda", dtype=torch.bfloat16)
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(5):          # same method as the XLA side: 50 queued calls, one sync
            t0 = time.perf_counter()
            for _ in range(50):
                a @ b
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / 50)
        rows.append({"gemm": name, "M": M, "N": Nn, "K": K, "torch_us": round(best * 1e6, 2)})
        del a, b
    jx = json.loads(jax_run(["--gemm", json.dumps([[r["M"], r["N"], r["K"], False] for r in rows])]).strip().splitlines()[-1])
    for r, j in zip(rows, jx):
        r["xla_us"] = j["us"]
        r["xla_kernel"] = j["kernel"]
        fl = 2 * r["M"] * r["N"] * r["K"]
        r["torch_tflops"] = round(fl / r["torch_us"] / 1e6, 1)
        r["xla_tflops"] = round(fl / r["xla_us"] / 1e6, 1)
        r["xla_speedup"] = round(r["torch_us"] / r["xla_us"], 2)
    return rows


def torch_profile_buckets(size, B, T, kernels, compile_, label):
    tag = (f"{size}_b{B}_s{T}_bf16_sdpa_fusedadam" + ("_compile" if compile_ else "") + f"_{kernels}")
    p = RESULTS / f"profile_{label}_sm_120_{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"launches_per_step": d["launches_per_step"], "gpu_busy_ms": d["gpu_busy_ms_per_step"],
            "buckets": {k: v["ms_per_step"] for k, v in d["buckets"].items()}, "source": p.name}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="medium", choices=list(SIZES))
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--machine-label", default="final")
    p.add_argument("--jax-attn", default="cudnn,xla")
    args = p.parse_args()

    cap_gpu_memory()
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = SIZES[args.size]
    B, T = args.batch, args.seq
    info = env_info(args.machine_label)
    print(f"{info['arch']} ({args.machine_label}) | {args.size} batch {B} seq {T}")

    # 1. parity
    par = parity(cfg, 2, 256, "cudnn")
    ok = all(v["pass"] for v in par.values())
    print(f"parity, one block, bf16, JAX vs PyTorch: {'PASS' if ok else 'FAIL'}  "
          + ", ".join(f"{k} {v['rel_err']:.1e}" for k, v in par.items()))

    # 1b. the model's GEMMs alone, cuBLAS against XLA
    gemms = gemm_compare(cfg, B, T)
    torch.cuda.empty_cache()

    # 2. one block, forward plus backward
    blocks = {"PyTorch eager": time_torch_block(cfg, B, T, "torch", False),
              "PyTorch torch.compile": time_torch_block(cfg, B, T, "torch", True),
              "PyTorch Triton kernels": time_torch_block(cfg, B, T, "triton", False)}
    torch.cuda.empty_cache()

    # 3. full step: PyTorch rows 4, 5, 6 now, then JAX in its own process
    bf16_ref = measure_matmul_tflops("bf16")["tflops"]
    base = dict(size=args.size, batch=B, seq=T, precision="bf16", attn="sdpa", fused_adam=True, ckpt=False, nvtx=False)
    rows = {}
    for name, extra in (("PyTorch eager (row 4)", dict(compile=False, kernels="torch")),
                        ("PyTorch torch.compile (row 5)", dict(compile=True, kernels="torch")),
                        ("PyTorch Triton kernels (row 6)", dict(compile=False, kernels="triton"))):
        r = run_one(cfg, dict(base, name=name, **extra), 5, 20, bf16_ref, None)
        rows[name] = {k: r.get(k) for k in ("status", "ms_per_step", "tokens_per_s", "mfu", "peak_mem_gb", "final_loss")}
        k = extra["kernels"]
        rows[name]["profile"] = torch_profile_buckets(args.size, B, T, k, extra["compile"], args.machine_label)
    torch.cuda.empty_cache()

    jax_rows = {}
    for attn in args.jax_attn.split(","):
        jax_run(["--size", args.size, "--batch", str(B), "--seq", str(T), "--attn", attn,
                 "--machine-label", args.machine_label])
        j = json.loads((RESULTS / f"jax_{args.machine_label}_{info['arch']}_{args.size}_b{B}_s{T}_{attn}.json").read_text())
        name = f"JAX jit, attention {attn}"
        jax_rows[name] = j
        rows[name] = {"status": "ok", **{k: j["model"][k] for k in ("ms_per_step", "tokens_per_s", "mfu", "peak_mem_gb",
                                                                    "final_loss")},
                      "profile": {"launches_per_step": j["profile"].get("kernels_per_step"),
                                  "gpu_busy_ms": j["profile"].get("gpu_ms_per_step"),
                                  "buckets": {k: v["ms_per_step"] for k, v in j["profile"].get("buckets", {}).items()}},
                      "hlo": {"fusions": j["model_hlo"]["fusions"], "fusion_kinds": j["model_hlo"]["fusion_kinds"],
                              "custom_calls": j["model_hlo"]["custom_calls"]}}
        blocks[f"JAX jit, attention {attn}"] = j["block"]["ms"]

    from tabulate import tabulate
    print("\nGEMMs alone, bf16")
    print(tabulate([[g["gemm"], f"{g['M']}x{g['N']}x{g['K']}", g["torch_us"], g["torch_tflops"], g["xla_us"],
                     g["xla_tflops"], g["xla_kernel"], g["xla_speedup"]] for g in gemms],
                   headers=["GEMM", "MxNxK", "torch us", "TFLOPs", "XLA us", "TFLOPs", "XLA kernel", "XLA speedup"]))
    print("\nOne block, forward plus backward")
    print(tabulate([[k, round(v, 3)] for k, v in blocks.items()], headers=["implementation", "ms"]))
    print("\nFull training step")
    print(tabulate([[k, r.get("ms_per_step"), r.get("tokens_per_s"), f"{100 * r['mfu']:.1f}%" if r.get("mfu") else "",
                     r.get("peak_mem_gb"), (r.get("profile") or {}).get("launches_per_step")] for k, r in rows.items()],
                   headers=["implementation", "ms/step", "tok/s", "MFU", "peak GB", "kernels/step"]))

    out = {**info, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "size": args.size, "batch": B, "seq": T,
           "bf16_ref_tflops": bf16_ref, "parity": par, "parity_pass": ok, "gemms": gemms, "block_fwd_bwd_ms": blocks,
           "step": rows}
    path = RESULTS / f"jax_vs_pytorch_{args.machine_label}_{info['arch']}_{args.size}_b{B}_s{T}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
