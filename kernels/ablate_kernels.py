"""Measure each fused Triton kernel's gain on its own, inside the real training step.

All rows use the fused model structure (--kernels triton) with bf16 autocast, SDPA and
fused Adam. "none" runs every op in PyTorch; each following row switches on exactly one
fused kernel; "all" switches on all of them. Gain is measured against "none".

  python kernels/ablate_kernels.py --size medium --batch 4 --seq 1024 --machine-label final
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_train import FUSED_OPS, RESULTS, SIZES, env_info, measure_matmul_tflops, run_one  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="medium", choices=list(SIZES))
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--machine-label", default="dev")
    args = p.parse_args()

    info = env_info(args.machine_label)
    cfg = SIZES[args.size]
    bf16_ref = measure_matmul_tflops("bf16")["tflops"]
    base = dict(size=args.size, batch=args.batch, seq=args.seq, precision="bf16", attn="sdpa", fused_adam=True,
                compile=False, ckpt=False, nvtx=False, kernels="triton")
    configs = [("none (all ops in PyTorch)", [])] + [(f"only {op}", [op]) for op in FUSED_OPS] \
        + [("all five", list(FUSED_OPS))]
    print(f"{info['arch']} ({args.machine_label}) | {args.size} batch {args.batch} seq {args.seq}")

    rows = []
    for name, ops in configs:
        row = run_one(cfg, dict(base, name=name, fused_ops=ops), args.warmup, args.steps, bf16_ref, None)
        rows.append(row)
        if row["status"] == "ok":
            print(f"  {name:<40} {row['ms_per_step']:8.2f} ms  {row['peak_mem_gb']:6.2f} GB", flush=True)
        else:
            print(f"  {name:<40} {row['status']} {row.get('error', '')}", flush=True)

    ref = rows[0]
    for r in rows:
        if r["status"] == "ok" and ref["status"] == "ok":
            r["saved_ms"] = round(ref["ms_per_step"] - r["ms_per_step"], 2)
            r["saved_pct"] = round(100 * (ref["ms_per_step"] - r["ms_per_step"]) / ref["ms_per_step"], 2)
            r["mem_saved_gb"] = round(ref["peak_mem_gb"] - r["peak_mem_gb"], 2)

    from tabulate import tabulate
    print()
    print(tabulate([[r["name"], r.get("ms_per_step", r["status"]), r.get("saved_ms", ""), f"{r.get('saved_pct', '')}%",
                     r.get("peak_mem_gb", ""), r.get("mem_saved_gb", "")] for r in rows],
                   headers=["config", "ms/step", "saved ms", "saved", "peak GB", "GB saved"]))
    sum_single = sum(r.get("saved_ms", 0) for r in rows[1:-1])
    print(f"\nsum of single kernel savings: {sum_single:.2f} ms, all five together: {rows[-1].get('saved_ms')} ms")

    out = {**info, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "size": args.size, "batch": args.batch,
           "seq": args.seq, "bf16_ref_tflops": bf16_ref, "rows": rows, "sum_single_saved_ms": round(sum_single, 2)}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"ablation_kernels_{args.machine_label}_{info['arch']}_{args.size}_b{args.batch}_s{args.seq}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
