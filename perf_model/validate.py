"""Validate the performance model against measured step times, and explain the MFU gap.

For each model size it measures the real training step (bench_train.run_one) for the
unfused model (ladder row 4: bf16, SDPA, fused Adam) and the fused model (row 6, Triton
kernels), then compares with two predictions from perf_model/roofline.py:

  roofline     peak matmul speed, peak bandwidth and launch overhead from calibration only
  calibrated   every GEMM and attention call timed at its real shape on this GPU,
               elementwise ops and launches still from the roofline

Error = (predicted - measured) / measured. Outputs:
  results/perf_model_validation_<label>.json
  results/perf_model_validation_<label>.png    predicted against measured
  results/mfu_gap_<label>.png                   where the time between ideal and measured goes

  python perf_model/validate.py --machine-label final
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perf_model.roofline import (CATEGORIES, RESULTS, SIZES, Machine, inventory,  # noqa: E402
                                 measure_machine_extras, measure_shape_times, predict)

CONFIGS = {"small": (8, 1024), "medium": (4, 1024), "large": (2, 1024)}
# Held out set: shapes the model was never compared against while it was being built
# (different batch, sequence length and one more model size).
HOLDOUT = [("tiny", 16, 1024), ("small", 4, 2048), ("medium", 8, 512), ("large", 4, 512), ("medium", 2, 2048)]
MODES = {"torch": "unfused (row 4)", "triton": "fused (row 6)"}


def measure(size, B, T, mode, warmup, steps, bf16_ref):
    from bench_train import SIZES as BSIZES, run_one
    run = dict(name=f"{size} {mode}", size=size, batch=B, seq=T, precision="bf16", attn="sdpa", fused_adam=True,
               compile=False, ckpt=False, nvtx=False, kernels=mode)
    return run_one(BSIZES[size], run, warmup, steps, bf16_ref, None)


def mfu_gap(pr_cal, measured_ms, ideal_ms, matmul_ideal_ms):
    """Split measured step time into ideal matmul time plus named losses.

    ideal: model FLOPs at the bf16 peak. matmul efficiency: calibrated GEMM time minus the
    same FLOPs at peak (tile shapes, wave quantization, the Ampere generation kernels cuBLAS
    picks here). The rest of the categories are their calibrated predicted time. Whatever
    the model does not explain (idle gaps, CPU launch stalls, cache effects) is the residual.
    """
    gap = {"ideal (model FLOPs at bf16 peak)": ideal_ms,
           "matmul below peak": pr_cal.by_cat["matmul"] - matmul_ideal_ms,
           "attention": pr_cal.by_cat["attention"],
           "memory bound elementwise, casts, loss": pr_cal.by_cat["elementwise"] + pr_cal.by_cat["copy/cast"]
           + pr_cal.by_cat["loss"],
           "optimizer": pr_cal.by_cat["optimizer"],
           "launch overhead": pr_cal.launch_ms}
    gap["idle and unmodeled (residual)"] = measured_ms - sum(gap.values())
    return {k: round(v, 3) for k, v in gap.items()}


def plot(results, label, arch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, meas, roof, cal = [], [], [], []
    for r in results:
        if r["measured_ms"] is None:
            continue
        names.append(f"{r['size']} b{r['batch']} s{r['seq']}\n{r['mode_name']}")
        meas.append(r["measured_ms"])
        roof.append(r["roofline_ms"])
        cal.append(r["calibrated_ms"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(names))
    w = 0.27
    ax.bar([i - w for i in x], meas, w, label="measured", color="#333333")
    ax.bar(list(x), cal, w, label="predicted, calibrated GEMMs", color="#4c78a8")
    ax.bar([i + w for i in x], roof, w, label="predicted, roofline only", color="#9ecae9")
    for i, (m, c) in enumerate(zip(meas, cal)):
        ax.text(i, c, f"{100 * (c - m) / m:+.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("ms per training step")
    ax.set_title(f"Performance model against measured step time ({arch}, {label})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"perf_model_validation_{label}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    rows = [r for r in results if r["measured_ms"] is not None]
    keys = list(rows[0]["mfu_gap"].keys())
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#8c564b", "#e377c2", "#7f7f7f"]
    bottoms = [0.0] * len(rows)
    for k, c in zip(keys, colors):
        vals = [max(0.0, r["mfu_gap"][k]) for r in rows]
        ax.bar(range(len(rows)), vals, bottom=bottoms, label=k, color=c)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    # The stacked categories are the calibrated prediction; when it is above the measured time
    # the residual is negative and not drawn, so mark the measured time on every bar.
    ax.scatter(range(len(rows)), [r["measured_ms"] for r in rows], marker="_", s=900, color="black",
               linewidths=2.5, zorder=5, label="measured step time")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['size']} b{r['batch']} s{r['seq']}\n{r['mode_name']}\nMFU {100 * r['measured_mfu']:.0f}%"
                        for r in rows], fontsize=7)
    ax.set_ylabel("ms per training step")
    ax.set_title(f"Where the step time goes beyond ideal matmul time ({arch}, {label})")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, max(max(bottoms), max(r["measured_ms"] for r in rows)) * 1.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"mfu_gap_{label}.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="small,medium,large")
    p.add_argument("--machine-label", default="dev")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--holdout", action="store_true", help="validate on the held out configurations")
    p.add_argument("--reuse-measured", action="store_true",
                   help="take measured step times from the previous validation JSON instead of rerunning")
    args = p.parse_args()

    import torch
    from bench_train import env_info, measure_matmul_tflops

    info = env_info(args.machine_label)
    cal_path = RESULTS / f"calibration_{args.machine_label}.json"
    shape_cache = RESULTS / f"shape_calibration_{args.machine_label}.json"
    extras = measure_machine_extras(shape_cache)
    m = Machine.from_calibration(cal_path, extras)
    bf16_ref = measure_matmul_tflops("bf16")["tflops"]
    peak = m.peak_tflops["bf16"]
    print(f"{info['arch']} ({args.machine_label}) | calibration: bf16 {peak} TFLOPs, DRAM {m.bandwidth_gbs} GB/s, "
          f"L2 {m.l2_gbs} GB/s ({m.l2_mb:.0f} MB), CPU issue {m.launch_us} us and GPU gap {m.gpu_gap_us} us per "
          f"kernel | bf16 reached this run: {bf16_ref} TFLOPs\n")
    stem = f"perf_model_validation_{args.machine_label}" + ("_holdout" if args.holdout else "")
    previous = {}
    prev_path = RESULTS / f"{stem}.json"
    if args.reuse_measured and prev_path.exists():
        for r in json.loads(prev_path.read_text())["results"]:
            previous[(r["size"], r["batch"], r["seq"], r["mode"])] = r

    results = []
    configs = HOLDOUT if args.holdout else [(size, *CONFIGS[size]) for size in args.sizes.split(",")]
    for size, B, T in configs:
        s = SIZES[size]
        model_flops = s.model_flops_per_token(T) * B * T
        ideal_ms = model_flops / (peak * 1e12) * 1e3
        for mode, mode_name in MODES.items():
            ops = inventory(s, B, T, "bf16", "sdpa", mode, "fused")
            shape_times = measure_shape_times(ops, shape_cache)
            pr_roof = predict(ops, m)
            pr_cal = predict(ops, m, shape_times=shape_times)
            if (size, B, T, mode) in previous and previous[(size, B, T, mode)]["measured_ms"]:
                row = {"status": "ok", "ms_per_step": previous[(size, B, T, mode)]["measured_ms"]}
            else:
                row = measure(size, B, T, mode, args.warmup, args.steps, bf16_ref)
            meas = row.get("ms_per_step") if row["status"] == "ok" else None
            matmul_flops = sum(o.total_flops for o in ops if o.cat == "matmul")
            r = {"size": size, "batch": B, "seq": T, "mode": mode, "mode_name": mode_name,
                 "measured_ms": meas, "measured_status": row["status"],
                 "measured_mfu": (model_flops / (meas / 1e3) / (peak * 1e12)) if meas else None,
                 "roofline_ms": round(pr_roof.total_ms, 3), "calibrated_ms": round(pr_cal.total_ms, 3),
                 "roofline_err_pct": round(100 * (pr_roof.total_ms - meas) / meas, 2) if meas else None,
                 "calibrated_err_pct": round(100 * (pr_cal.total_ms - meas) / meas, 2) if meas else None,
                 "predicted_launches": pr_cal.launches,
                 "roofline_by_cat_ms": {k: round(v, 3) for k, v in pr_roof.by_cat.items()},
                 "calibrated_by_cat_ms": {k: round(v, 3) for k, v in pr_cal.by_cat.items()},
                 "launch_ms": round(pr_cal.launch_ms, 3), "ideal_ms": round(ideal_ms, 3),
                 "predicted_gpu_ms": round(pr_cal.gpu_ms, 3), "predicted_cpu_issue_ms": round(pr_cal.cpu_ms, 3),
                 "predicted_cpu_bound": pr_cal.cpu_bound,
                 "matmul_efficiency": round(matmul_flops / (pr_cal.by_cat["matmul"] / 1e3) / (peak * 1e12), 4),
                 "mfu_gap": mfu_gap(pr_cal, meas, ideal_ms, matmul_flops / (peak * 1e12) * 1e3) if meas else None}
            results.append(r)
            if meas:
                print(f"{size:<7} {mode_name:<16} measured {meas:8.2f} ms | roofline {pr_roof.total_ms:8.2f} ms "
                      f"({r['roofline_err_pct']:+6.1f}%) | calibrated {pr_cal.total_ms:8.2f} ms "
                      f"({r['calibrated_err_pct']:+6.1f}%) | MFU {100 * r['measured_mfu']:.1f}% | "
                      f"GEMM efficiency {100 * r['matmul_efficiency']:.0f}%", flush=True)
            else:
                print(f"{size:<7} {mode_name:<16} measured {row['status']}", flush=True)
            torch.cuda.empty_cache()

    ok = [r for r in results if r["measured_ms"]]
    summary = {}
    for mode in MODES:
        for kind in ("roofline", "calibrated"):
            errs = [abs(r[f"{kind}_err_pct"]) for r in ok if r["mode"] == mode]
            if errs:
                summary[f"{mode} {kind} mean abs error pct"] = round(sum(errs) / len(errs), 2)
    print("\nmean absolute error:", json.dumps(summary, indent=1))

    print("\nMFU gap, ms per step (calibrated model; residual = measured minus everything modeled):")
    for r in ok:
        print(f"  {r['size']} {r['mode_name']}: " + ", ".join(f"{k} {v:.1f}" for k, v in r["mfu_gap"].items()))

    out = {**info, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "calibration": str(cal_path.name),
           "machine_model": {"bf16_tflops": peak, "dram_gbs": m.bandwidth_gbs, "l2_gbs": m.l2_gbs, "l2_mb": m.l2_mb,
                             "cpu_issue_us": m.launch_us, "gpu_gap_us": m.gpu_gap_us},
           "bf16_ref_tflops_this_run": bf16_ref, "results": results, "summary": summary}
    out["holdout"] = args.holdout
    path = RESULTS / f"{stem}.json"
    path.write_text(json.dumps(out, indent=2))
    tag = args.machine_label + ("_holdout" if args.holdout else "")
    plot(results, tag, info["arch"])
    print(f"\nSaved {path.name}, perf_model_validation_{tag}.png, mfu_gap_{tag}.png")


if __name__ == "__main__":
    main()
