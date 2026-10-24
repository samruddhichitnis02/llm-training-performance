"""Charts for the README, drawn from the saved result files (no GPU needed).

  results/ladder_tokens_per_s_<label>.png   tokens per second and MFU for every ladder row
  results/profile_buckets_<label>.png       GPU time per kernel bucket, before and after fusion

  python make_charts.py --machine-label final
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def ladder(label):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (size, b) in zip(axes, (("small", 8), ("medium", 4))):
        d = json.loads((RESULTS / "milestones" / f"m5_sweep_{label}_sm_120_{size}_b{b}_s1024.json").read_text())
        rows = [r for r in d["rows"] if r["status"] == "ok"]
        names = [r["name"].replace(" + ", "\n+ ", 1) for r in rows]
        tps = [r["tokens_per_s"] for r in rows]
        colors = ["#bdbdbd"] * 2 + ["#9ecae1"] * 3 + ["#3182bd"] * 2 + ["#e6550d"] * 2
        bars = ax.bar(range(len(rows)), tps, color=colors[:len(rows)])
        for i, (bar, r) in enumerate(zip(bars, rows)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{r['speedup']:.2f}x\n{100 * r['mfu']:.0f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("tokens per second")
        ax.set_ylim(0, max(tps) * 1.22)
        ax.set_title(f"{size}, batch {b}, seq 1024 (labels: speedup over row 1, MFU)", fontsize=9)
    fig.suptitle(f"Optimization ladder, one training step (sm_120, {label})", fontsize=10)
    fig.tight_layout()
    out = RESULTS / f"ladder_tokens_per_s_{label}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def buckets(label):
    configs = [("bf16 + SDPA", "bf16_sdpa_torch"), ("+ fused Adam", "bf16_sdpa_fusedadam_torch"),
               ("+ torch.compile", "bf16_sdpa_fusedadam_compile_torch"),
               ("Triton kernels", "bf16_sdpa_fusedadam_triton"), ("CUDA C++ add+RMSNorm", "bf16_sdpa_fusedadam_cuda"),
               ("+ FP8 linears", "bf16_sdpa_fusedadam_triton_fp8"), ("+ NVFP4 linears", "bf16_sdpa_fusedadam_triton_fp4")]
    keys = ["matmul", "attention", "optimizer", "memory copies", "elementwise and other"]
    colors = ["#3182bd", "#9467bd", "#8c564b", "#fdae6b", "#e6550d"]
    data = []
    for name, tag in configs:
        d = json.loads((RESULTS / f"profile_{label}_sm_120_medium_b4_s1024_{tag}.json").read_text())
        data.append((name, d))
    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = [0.0] * len(data)
    for k, c in zip(keys, colors):
        vals = [d["buckets"][k]["ms_per_step"] for _, d in data]
        ax.bar(range(len(data)), vals, bottom=bottoms, color=c, label=k)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    idle = [max(0.0, d["clean_step_ms"] - d["gpu_busy_ms_per_step"]) for _, d in data]
    ax.bar(range(len(data)), idle, bottom=bottoms, color="#d9d9d9", label="GPU idle")
    for i, (_, d) in enumerate(data):
        ax.text(i, d["clean_step_ms"] + 1, f"{d['clean_step_ms']:.0f} ms\n{d['launches_per_step']:.0f} kernels",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([n for n, _ in data], fontsize=8, rotation=15)
    ax.set_ylabel("ms per training step")
    ax.set_ylim(0, max(d["clean_step_ms"] for _, d in data) * 1.2)
    ax.set_title(f"Where GPU time goes, medium, batch 4, seq 1024 (sm_120, {label})", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    out = RESULTS / f"profile_buckets_{label}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine-label", default="final")
    a = p.parse_args()
    for f in (ladder(a.machine_label), buckets(a.machine_label)):
        print(f"Saved {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
