"""Profile a few training steps and show where GPU time goes.

Prints the top GPU kernels (calls per step, ms per step, percent), a bucket
breakdown (matmul, attention, optimizer, memory copies, elementwise and other)
and the GPU idle percent. Saves a small summary JSON to results/ and the full
Chrome trace to profiles/ (open it at https://ui.perfetto.dev or chrome://tracing).

The step is first timed WITHOUT the profiler, because the profiler slows the step
down. That clean time is what the idle estimate is measured against.

Examples
  python profile_step.py --size medium --batch 4 --seq 1024 --machine-label final
  python profile_step.py --size medium --batch 4 --seq 1024 --fused-adam --kernels triton --machine-label final

Nsight Systems trace (uses the --nvtx ranges: forward, backward, optimizer, block<i>, step<i>):
  nsys profile -t cuda,nvtx,osrt -o profiles/nsys_medium_torch --force-overwrite true \\
      python bench_train.py --size medium --batch 4 --seq 1024 --nvtx --warmup 3 --steps 5
  nsys stats --report cuda_gpu_kern_sum profiles/nsys_medium_torch.nsys-rep
Open the .nsys-rep file in the Nsight Systems GUI to see the timeline with the named ranges.
"""

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, schedule

from bench_train import RESULTS, ROOT, SIZES, build, cleanup, env_info

PROFILES = ROOT / "profiles"
warnings.filterwarnings("ignore", message=".*Profiler clears events.*")

# Profiler rows that wrap real kernels. Counting them would double count GPU time.
WRAPPER_PREFIXES = ("aten::", "autograd::", "Optimizer", "ProfilerStep", "cudaLaunch", "cuda")

# First matching bucket wins, so the order matters. Lowercase substrings of kernel names.
# Attention comes before matmul because flash attention kernel names contain "cutlass".
BUCKETS = [
    ("attention", ("flash", "fmha", "attention", "sdpa", "attn")),
    ("matmul", ("gemm", "cutlass", "cublas", "nvjet", "xmma", "triton_tem_", "matmul")),
    ("optimizer", ("adam", "multi_tensor", "multitensor")),
    ("memory copies", ("memcpy", "memset", "copy_kernel", "direct_copy", "copy")),
]
OTHER = "elementwise and other"
BUCKET_NAMES = ["matmul", "attention", "optimizer", "memory copies", OTHER]


def bucket_of(name, attn="sdpa", in_optimizer=False):
    """Bucket for one kernel. Kernels launched inside Optimizer.step always count as optimizer.

    Softmax counts as attention only with naive attention. With SDPA the softmax lives inside
    the flash kernel, and the remaining softmax kernels belong to the cross entropy loss.
    """
    if in_optimizer:
        return "optimizer"
    low = name.lower()
    if attn == "naive" and "softmax" in low:
        return "attention"
    for bucket, keys in BUCKETS:
        if any(k in low for k in keys):
            return bucket
    return OTHER


def optimizer_ranges(prof):
    """GPU side time ranges of the Optimizer.step annotation, so every optimizer kernel is caught
    even when its name does not say adam (for example the foreach lerp and sqrt kernels)."""
    return [(e.time_range.start, e.time_range.end) for e in prof.events()
            if e.device_type == DeviceType.CUDA and e.name.startswith("Optimizer.step")]


def inside(t, ranges):
    return any(s <= t <= e for s, e in ranges)


def is_wrapper(name):
    return name.startswith(WRAPPER_PREFIXES)


def gpu_events(prof):
    """Real GPU kernel and memcpy events, with profiler wrapper rows removed."""
    out = []
    for e in prof.events():
        if e.device_type != DeviceType.CUDA or is_wrapper(e.name):
            continue
        out.append(e)
    return out


def union_us(intervals):
    """Total length of the union of (start, end) intervals, so overlapping streams are not counted twice."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def time_clean(step, warmup, steps):
    """Median step time with no profiler attached."""
    for _ in range(warmup):
        step()
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="small", choices=list(SIZES))
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--attn", default="sdpa", choices=["naive", "sdpa"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--kernels", default="torch", choices=["torch", "triton", "cuda"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10, help="clean timed steps")
    p.add_argument("--prof-steps", type=int, default=3, help="steps recorded by the profiler")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--machine-label", default="dev")
    p.add_argument("--tag", default=None, help="extra name for the output files")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available.")

    info = env_info(args.machine_label)
    cfg = SIZES[args.size]
    run = dict(size=args.size, batch=args.batch, seq=args.seq, precision=args.precision, attn=args.attn,
               compile=args.compile, fused_adam=args.fused_adam, ckpt=args.ckpt, kernels=args.kernels, nvtx=False)
    desc = (f"{args.precision} {args.attn}" + (" fusedadam" if args.fused_adam else "")
            + (" compile" if args.compile else "") + (" ckpt" if args.ckpt else "") + f" kernels={args.kernels}")
    print(f"{info['arch']} ({args.machine_label}) | {args.size} batch {args.batch} seq {args.seq} | {desc}")

    cleanup()
    model, opt, step = build(cfg, run)

    # 1. Clean timing, no profiler.
    step_s = time_clean(step, args.warmup, args.steps)
    step_us = step_s * 1e6
    print(f"clean step time (no profiler): {step_s * 1e3:.2f} ms")

    # 2. Profiled steps. One warmup step inside the profiler so its own startup is not recorded.
    n = args.prof_steps
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=0, warmup=1, active=n, repeat=1)) as prof:
        for _ in range(n + 1):
            step()
            torch.cuda.synchronize()
            prof.step()

    events = gpu_events(prof)
    if not events:
        sys.exit("The profiler recorded no GPU kernels. Check that CUPTI works on this machine.")
    opt_ranges = optimizer_ranges(prof)

    # Per kernel totals. Key is (name, bucket) so the same kernel used by the model and by the
    # optimizer (for example a copy) is counted in the right bucket for each use.
    per_kernel = {}
    for e in events:
        b = bucket_of(e.name, args.attn, inside(e.time_range.start, opt_ranges))
        k = per_kernel.setdefault((e.name, b), {"us": 0.0, "calls": 0})
        k["us"] += e.time_range.end - e.time_range.start
        k["calls"] += 1
    busy_sum_us = sum(k["us"] for k in per_kernel.values())
    busy_union_us = union_us([(e.time_range.start, e.time_range.end) for e in events])

    busy_per_step = busy_union_us / n
    idle_pct = max(0.0, 100.0 * (1.0 - busy_per_step / step_us))
    launches_per_step = len(events) / n

    # Buckets.
    buckets = {b: {"us": 0.0, "calls": 0} for b in BUCKET_NAMES}
    for (name, bname), k in per_kernel.items():
        buckets[bname]["us"] += k["us"]
        buckets[bname]["calls"] += k["calls"]

    # Print top kernels.
    from tabulate import tabulate
    ranked = sorted(per_kernel.items(), key=lambda kv: -kv[1]["us"])
    rows = [[(name[:80] + "...") if len(name) > 83 else name, bname, k["calls"] / n,
             round(k["us"] / n / 1e3, 3), f"{100 * k['us'] / busy_sum_us:.1f}%"]
            for (name, bname), k in ranked[: args.top]]
    print(f"\nTop {len(rows)} GPU kernels ({n} profiled steps)")
    print(tabulate(rows, headers=["kernel", "bucket", "calls/step", "ms/step", "% GPU"]))

    brows = [[b, round(v["calls"] / n), round(v["us"] / n / 1e3, 3), f"{100 * v['us'] / busy_sum_us:.1f}%"]
             for b, v in buckets.items()]
    print("\nBucket breakdown")
    print(tabulate(brows, headers=["bucket", "launches/step", "ms/step", "% GPU"]))

    print(f"\nGPU kernel launches per step: {launches_per_step:.0f}")
    print(f"GPU busy per step: {busy_per_step / 1e3:.2f} ms of {step_s * 1e3:.2f} ms clean step time")
    print(f"GPU idle: {idle_pct:.1f}%")
    warning = None
    if busy_per_step > step_us:
        warning = ("GPU busy time measured under the profiler is larger than the clean step time. "
                   "The profiler inflated kernel times, so trust only the percentage shares, not the absolute ms.")
        print(f"WARNING: {warning}")

    # Save outputs.
    tag = args.tag or (f"{args.size}_b{args.batch}_s{args.seq}_{args.precision}_{args.attn}"
                       + ("_fusedadam" if args.fused_adam else "") + ("_compile" if args.compile else "")
                       + ("_ckpt" if args.ckpt else "") + f"_{args.kernels}")
    stem = f"profile_{args.machine_label}_{info['arch']}_{tag}"
    PROFILES.mkdir(exist_ok=True)
    trace_path = PROFILES / f"{stem}.json"
    prof.export_chrome_trace(str(trace_path))

    summary = {
        **info,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "size": args.size, "batch": args.batch, "seq": args.seq,
        "config": {k: v for k, v in run.items() if k != "nvtx"},
        "clean_step_ms": round(step_s * 1e3, 3),
        "profiled_steps": n,
        "gpu_busy_ms_per_step": round(busy_per_step / 1e3, 3),
        "gpu_idle_pct": round(idle_pct, 2),
        "launches_per_step": round(launches_per_step, 1),
        "warning": warning,
        "buckets": {b: {"ms_per_step": round(v["us"] / n / 1e3, 4), "launches_per_step": round(v["calls"] / n, 1),
                        "pct": round(100 * v["us"] / busy_sum_us, 2)} for b, v in buckets.items()},
        "top_kernels": [{"name": name, "bucket": bname, "calls_per_step": k["calls"] / n,
                         "ms_per_step": round(k["us"] / n / 1e3, 4), "pct": round(100 * k["us"] / busy_sum_us, 2)}
                        for (name, bname), k in ranked[:50]],
        "chrome_trace": str(trace_path.relative_to(ROOT)),
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{stem}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)} and {trace_path.relative_to(ROOT)}")

    del model, opt, step
    cleanup()


if __name__ == "__main__":
    main()
