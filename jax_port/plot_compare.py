"""Charts for the JAX against PyTorch comparison, from the saved result files.

  results/jax_vs_pytorch_<label>.png
    left:  step time per model size for PyTorch torch.compile, the PyTorch Triton
           kernels and JAX jit (cuDNN attention)
    right: GPU time per kernel bucket for medium, PyTorch Triton kernels against JAX

  python jax_port/plot_compare.py --machine-label final
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIGS = [("small", 8), ("medium", 4), ("large", 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine-label", default="final")
    args = p.parse_args()
    lab = args.machine_label
    M = RESULTS / "milestones"

    sizes, torch_c, torch_t, jx = [], [], [], []
    cmp_ = json.loads((RESULTS / f"jax_vs_pytorch_{lab}_sm_120_medium_b4_s1024.json").read_text())
    for size, b in CONFIGS:
        j = json.loads((RESULTS / f"jax_{lab}_sm_120_{size}_b{b}_s1024_cudnn.json").read_text())
        if size == "medium":
            c = cmp_["step"]["PyTorch torch.compile (row 5)"]["ms_per_step"]
            t = cmp_["step"]["PyTorch Triton kernels (row 6)"]["ms_per_step"]
        else:
            c = json.loads((M / f"m9_pytorch_{size}_compile.json").read_text())["rows"][0]["ms_per_step"]
            t = json.loads((M / f"m9_pytorch_{size}_triton.json").read_text())["rows"][0]["ms_per_step"]
        sizes.append(f"{size}\nbatch {b}, seq 1024")
        torch_c.append(c)
        torch_t.append(t)
        jx.append(j["model"]["ms_per_step"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    w = 0.27
    x = range(len(sizes))
    for off, vals, lbl, col in ((-w, torch_c, "PyTorch torch.compile", "#9ecae1"),
                                (0, torch_t, "PyTorch Triton kernels", "#3182bd"),
                                (w, jx, "JAX jit (cuDNN attention)", "#e6550d")):
        ax.bar([i + off for i in x], vals, w, label=lbl, color=col)
        for i, v in enumerate(vals):
            ax.text(i + off, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sizes, fontsize=8)
    ax.set_ylabel("ms per training step")
    ax.set_title("Full training step (bf16, fused attention)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    tprof = cmp_["step"]["PyTorch Triton kernels (row 6)"]["profile"]["buckets"]
    jprof = json.loads((RESULTS / f"jax_{lab}_sm_120_medium_b4_s1024_cudnn.json").read_text())["profile"]["buckets"]
    jopt = json.loads((RESULTS / f"jax_{lab}_sm_120_medium_b4_s1024_cudnn.json").read_text())["optimizer"]["ms"]
    cats = ["matmul", "attention", "optimizer", "elementwise and copies"]
    tv = [tprof["matmul"], tprof["attention"], tprof["optimizer"],
          tprof["elementwise and other"] + tprof["memory copies"]]
    # The JAX trace has no optimizer boundary: its fusions bucket holds both, so the
    # optimizer time measured alone is split out of it.
    jv = [jprof["matmul"]["ms_per_step"], jprof["attention"]["ms_per_step"], jopt,
          jprof["fusions and other"]["ms_per_step"] - jopt + jprof["memory copies"]["ms_per_step"]]
    xs = range(len(cats))
    ax.bar([i - 0.2 for i in xs], tv, 0.4, label="PyTorch Triton kernels", color="#3182bd")
    ax.bar([i + 0.2 for i in xs], jv, 0.4, label="JAX jit", color="#e6550d")
    for i, (a, b) in enumerate(zip(tv, jv)):
        ax.text(i - 0.2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("GPU ms per step")
    ax.set_title("medium, GPU time by kernel bucket", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("JAX against PyTorch on the same model and shapes (sm_120, final)", fontsize=10)
    fig.tight_layout()
    out = RESULTS / f"jax_vs_pytorch_{lab}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out.name}")
    print("step ms:", dict(zip([s.split()[0] for s in sizes], zip(torch_c, torch_t, jx))))
    print("medium buckets, PyTorch Triton vs JAX:", dict(zip(cats, zip(tv, [round(v, 3) for v in jv]))))


if __name__ == "__main__":
    main()
