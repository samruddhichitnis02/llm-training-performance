"""Scaling model: step time, MFU and tokens per second for data, tensor and pipeline
parallel training, alone and combined, from 8 to 4096 GPUs.

PREDICTION ONLY. The compute side comes from the validated performance model in
roofline.py (calibrated on one sm_120 GPU, 4.6% mean error). The communication side uses
the link constants in links.json, which are not measured here: the only machine
available has one GPU, so no NCCL all reduce could be timed.

Model of one training step for a layout (dp, tp, pp, microbatch size mb):
  global batch G sequences of length T, m = G / (dp x mb) microbatches per step

  compute     per microbatch per stage = roofline prediction of the model with the block
              GEMMs sharded tp ways (Megatron style), divided over pp stages. The lm head
              and embedding stay on the stages that hold them; that imbalance is ignored.
  TP comm     2 all reduces of the (mb, T, d) bf16 activation per layer in the forward,
              2 in the backward, on the intra node link, not overlapped.
  PP          1F1B schedule: (m + p - 1) microbatch slots per stage, bubble fraction
              (p - 1) / (m + p - 1); one (mb, T, d) send per microbatch per direction
              between neighbour stages.
  DP comm     one gradient all reduce per step over this GPU's fp32 gradients
              (parameters / (tp x pp) x 4 bytes), hierarchical: ring inside the node, ring
              across nodes for the 1 / g share each GPU owns. It overlaps with the backward
              pass of the last microbatch; the rest is exposed.
  optimizer   fused Adam with ZeRO 1 (distributed optimizer): each GPU updates
              parameters / (tp x pp x dp) of the model.
  memory      fp32 weights and gradients, a bf16 weight copy, Adam state / dp, and
              activations for the microbatches 1F1B keeps in flight; must fit in 90% of
              the GPU.

Ring all reduce over n GPUs at bus bandwidth bw: 2 (n - 1) / n x bytes / bw + 2 (n - 1) x latency.

  python perf_model/scaling.py --machine-label final
  python perf_model/scaling.py --models llama3-70b --gpus 1024 --system nvlink_ib --show-top 5
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perf_model.roofline import RESULTS, SIZES, Machine, inventory, load_shape_times, predict  # noqa: E402

LINKS = Path(__file__).resolve().with_name("links.json")

# Workloads: model, sequence length, global batch in sequences.
WORKLOADS = {
    "small": ("small", 1024, 512),            # 0.5M tokens per step
    "medium": ("medium", 1024, 512),
    "large": ("large", 1024, 512),
    "llama3-8b": ("llama3-8b", 4096, 1024),   # 4M tokens per step
    "llama3-70b": ("llama3-70b", 4096, 1024),
}
GPU_COUNTS = [8, 64, 1024, 4096]


@dataclass
class Link:
    bw_gbs: float
    latency_us: float


def ring_allreduce_s(nbytes, n, link):
    if n <= 1:
        return 0.0
    return 2 * (n - 1) / n * nbytes / (link.bw_gbs * 1e9) + 2 * (n - 1) * link.latency_us * 1e-6


def hier_allreduce_s(nbytes, n, g, intra, inter):
    """All reduce over n GPUs with g of them per node: reduce scatter and all gather inside
    the node, and an all reduce across n / g nodes of the 1 / g share each GPU owns."""
    if n <= 1:
        return 0.0
    g = max(1, min(g, n))
    nodes = n // g
    t = 0.0
    if g > 1:
        t += ring_allreduce_s(nbytes, g, intra)
    if nodes > 1:
        t += ring_allreduce_s(nbytes / g, nodes, inter)
    return t


def p2p_s(nbytes, link):
    return nbytes / (link.bw_gbs * 1e9) + link.latency_us * 1e-6


class StageCompute:
    """Per microbatch compute time of one pipeline stage, from the roofline model, cached
    per (tp, mb). Optimizer time is taken out and modeled separately (ZeRO 1)."""

    def __init__(self, shape, seq, machine, shape_times):
        self.shape, self.seq, self.m, self.st = shape, seq, machine, shape_times
        self.cache = {}

    def get(self, tp, mb, ckpt=False):
        """(seconds per microbatch, of which backward). With activation checkpointing every
        block's forward runs twice; the second time is part of the backward pass."""
        key = (tp, mb)
        if key not in self.cache:
            ops = [o for o in inventory(self.shape, mb, self.seq, "bf16", "sdpa", "triton", "fused", tp=tp)
                   if o.cat != "optimizer"]
            p = predict(ops, self.m, "bf16", self.st)
            fwd = sum(o["ms"] for o in p.ops if o["phase"] == "fwd") / 1e3
            bwd = sum(o["ms"] for o in p.ops if o["phase"] == "bwd") / 1e3
            self.cache[key] = (p.total_ms / 1e3, fwd, bwd)
        total, fwd, bwd = self.cache[key]
        if ckpt:
            return total + fwd, bwd + fwd
        return total, bwd


def evaluate(shape, seq, G, n, tp, pp, mb, comp, links, system, cfg, ckpt=False, overlap=True):
    """Step time and breakdown for one layout, or None if it does not divide or fit."""
    if n % (tp * pp):
        return None
    dp = n // (tp * pp)
    if G % (dp * mb) or shape.L % pp or shape.H % tp or shape.kvh % tp or shape.ffn % tp:
        return None
    m = G // (dp * mb)
    if pp > 1 and m < pp:            # 1F1B needs at least p microbatches to fill the pipeline
        return None
    gpn = system["gpus_per_node"]
    if tp > gpn:
        return None
    intra = links[system["intra"]]
    inter = links[system["inter"]]

    total_p, _ = shape.params()
    p_gpu = total_p / (tp * pp)
    # memory: fp32 weight + fp32 grad + bf16 weight copy (10 B) + Adam (8 B) sharded over dp
    act_coeff = cfg["activation_bytes_per_token_per_layer_per_d_ckpt" if ckpt
                    else "activation_bytes_per_token_per_layer_per_d"]
    in_flight = min(pp, m)
    act = act_coeff * shape.d * seq * mb * (shape.L / pp) * in_flight / tp
    if ckpt:   # plus the full activations of the one block being recomputed
        act += cfg["activation_bytes_per_token_per_layer_per_d"] * shape.d * seq * mb / tp
    mem = p_gpu * 10 + p_gpu * 8 / dp + act
    if mem > 0.9 * cfg["gpu_memory_gb"] * 1e9:
        return None

    t_mb, t_mb_bwd = comp.get(tp, mb, ckpt)
    t_stage = t_mb / pp
    t_stage_bwd = t_mb_bwd / pp

    act_bytes = mb * seq * shape.d * 2
    tp_link = intra
    t_tp = 4 * (shape.L / pp) * ring_allreduce_s(act_bytes, tp, tp_link) if tp > 1 else 0.0

    # pipeline stages: consecutive stages share a node while tp * pp fits in it
    pp_link = intra if tp * pp <= gpn else inter
    t_send = p2p_s(act_bytes, pp_link) if pp > 1 else 0.0

    slots = m + pp - 1
    t_pipe = slots * (t_stage + t_tp + 2 * t_send)
    bubble = (pp - 1) / slots

    grad_bytes = p_gpu * 4
    g_dp = max(1, gpn // (tp * pp)) if tp * pp < gpn else 1
    t_dp = hier_allreduce_s(grad_bytes, dp, g_dp, intra, inter)
    hidden = min(t_dp, t_stage_bwd) if overlap else 0.0
    t_dp_exposed = t_dp - hidden

    bw = comp.m.bandwidth_gbs * 1e9
    t_opt = (p_gpu / dp) * 28 / bw + p_gpu * 6 / bw     # Adam on the shard, then the bf16 weight cast

    step = t_pipe + t_dp_exposed + t_opt
    tokens = G * seq
    mfu = shape.model_flops_per_token(seq) * tokens / step / n / (comp.m.peak_tflops["bf16"] * 1e12)
    return {
        "gpus": n, "dp": dp, "tp": tp, "pp": pp, "mb": mb, "microbatches": m, "ckpt": ckpt,
        "step_s": step, "tokens_per_s": tokens / step, "tokens_per_s_per_gpu": tokens / step / n, "mfu": mfu,
        "mem_gb": mem / 1e9, "bubble_frac": bubble,
        "breakdown_s": {"compute": m * t_stage, "pipeline bubble": (pp - 1) * t_stage,
                        "tensor parallel comm": slots * t_tp, "pipeline sends": slots * 2 * t_send,
                        "data parallel comm (exposed)": t_dp_exposed, "optimizer": t_opt},
        "dp_allreduce_s": t_dp, "dp_hidden_s": hidden,
    }


def layouts(shape, n, restrict=None):
    """Candidate (tp, pp, mb, ckpt) for n GPUs. restrict: 'dp', 'tp+dp', 'pp+dp' or None."""
    for tp in (1, 2, 4, 8):
        for pp in divisors(shape.L, shape.L):
            if tp * pp > n:
                continue
            if restrict == "dp" and (tp > 1 or pp > 1):
                continue
            if restrict == "tp+dp" and pp > 1:
                continue
            if restrict == "pp+dp" and tp > 1:
                continue
            for mb in (1, 2, 4, 8, 16):
                for ckpt in (False, True):
                    yield tp, pp, mb, ckpt


def divisors(x, cap):
    return [d for d in range(1, min(x, cap) + 1) if x % d == 0]


def all_layouts(shape, seq, G, n, comp, links, system, cfg, restrict=None):
    out = []
    for tp, pp, mb, ckpt in layouts(shape, n, restrict):
        r = evaluate(shape, seq, G, n, tp, pp, mb, comp, links, system, cfg, ckpt)
        if r:
            out.append(r)
    return sorted(out, key=lambda r: -r["tokens_per_s"])


def search(shape, seq, G, n, comp, links, system, cfg, restrict=None):
    """Best layout by tokens per second, or None."""
    rows = all_layouts(shape, seq, G, n, comp, links, system, cfg, restrict)
    return rows[0] if rows else None


def why_none(shape, seq, G, n, cfg):
    """Short reason no layout works at this GPU count."""
    max_mp = 8 * shape.L
    if G * 8 * shape.L < n:
        return f"global batch of {G} sequences is too small for {n} GPUs"
    total_p, _ = shape.params()
    if total_p * 10 / min(max_mp, n) > 0.9 * cfg["gpu_memory_gb"] * 1e9:
        return "weights and gradients do not fit even with the widest tp x pp split"
    return "no dp x tp x pp split both divides the batch and fits in memory"


def lay(r):
    return f"dp{r['dp']} tp{r['tp']} pp{r['pp']} mb{r['mb']}" + (" ckpt" if r["ckpt"] else "")


def plot(table, label, arch, systems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    styles = {"nvlink_ib": "-", "pcie_ib": "--"}
    colors = {"small": "#9ecae1", "medium": "#4292c6", "large": "#08519c", "llama3-8b": "#fd8d3c",
              "llama3-70b": "#d94801"}
    for model, per_sys in table.items():
        for sysname in systems:
            rows = [r for r in per_sys[sysname]["best"] if r]
            if not rows:
                continue
            xs = [r["gpus"] for r in rows]
            axes[0].plot(xs, [r["tokens_per_s"] for r in rows], styles[sysname], marker="o", color=colors[model],
                         label=f"{model}, {sysname}")
            axes[1].plot(xs, [100 * r["mfu"] for r in rows], styles[sysname], marker="o", color=colors[model],
                         label=f"{model}, {sysname}")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(GPU_COUNTS)
        ax.set_xticklabels([str(g) for g in GPU_COUNTS])
        ax.set_xlabel("GPUs")
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("tokens per second (best layout)")
    axes[1].set_ylabel("MFU % (best layout)")
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle(f"PREDICTED scaling, best dp x tp x pp layout per point ({arch} compute, {label} calibration; "
                 f"links not measured)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / f"scaling_curves_{label}.png", dpi=130)
    plt.close(fig)


def plot_breakdown(rows, label, title, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r]
    if not rows:
        print(f"  (no layouts to draw for {fname})")
        return
    keys = list(rows[0]["breakdown_s"].keys())
    colors = ["#2ca02c", "#bcbd22", "#9467bd", "#17becf", "#d62728", "#8c564b"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # Shares of the step, so GPU counts with very different step times compare on one axis.
    bottoms = [0.0] * len(rows)
    for k, c in zip(keys, colors):
        vals = [100 * r["breakdown_s"][k] / r["step_s"] for r in rows]
        ax.bar(range(len(rows)), vals, bottom=bottoms, label=k, color=c)
        for i, v in enumerate(vals):
            if v >= 4:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['gpus']} GPUs, step {r['step_s']:.1f} s\n{lay(r)}\nMFU {100 * r['mfu']:.0f}%"
                        for r in rows], fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of step time")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(WORKLOADS))
    p.add_argument("--gpus", default=",".join(map(str, GPU_COUNTS)))
    p.add_argument("--systems", default="nvlink_ib,pcie_ib")
    p.add_argument("--machine-label", default="final")
    p.add_argument("--links", default=str(LINKS))
    p.add_argument("--show-top", type=int, default=0, help="also print the N best layouts per point")
    p.add_argument("--measure-shapes", action="store_true",
                   help="time all sharded GEMM and attention shapes on this GPU first (cached)")
    args = p.parse_args()

    cfg = json.loads(Path(args.links).read_text())
    links = {k: Link(v["bw_gbs"], v["latency_us"]) for k, v in cfg["links"].items()}
    cal = RESULTS / f"calibration_{args.machine_label}.json"
    shape_cache = RESULTS / f"shape_calibration_{args.machine_label}.json"
    extras = json.loads(shape_cache.read_text()).get("extras") if shape_cache.exists() else None
    machine = Machine.from_calibration(cal, extras)
    if args.measure_shapes:
        # Time every sharded GEMM and attention shape the search can use, so no prediction
        # relies on the efficiency fallback (tp sharded GEMMs are smaller and less efficient).
        from perf_model.roofline import measure_shape_times
        for model in args.models.split(","):
            size, seq, _ = WORKLOADS[model]
            ops = []
            for tp in (1, 2, 4, 8):
                for mb in (1, 2, 4, 8, 16):
                    if SIZES[size].H % tp == 0 and SIZES[size].kvh % tp == 0:
                        ops += inventory(SIZES[size], mb, seq, "bf16", "sdpa", "triton", "fused", tp=tp)
            n_new = len({o.key for o in ops if o.key} - set(load_shape_times(shape_cache)))
            print(f"timing {n_new} new GEMM and attention shapes for {model} ...", flush=True)
            measure_shape_times(ops, shape_cache)
    shape_times = load_shape_times(shape_cache)
    systems = args.systems.split(",")
    gpus = [int(g) for g in args.gpus.split(",")]
    print("PREDICTION: compute from the validated roofline model on this GPU's calibration "
          f"({machine.arch}, {args.machine_label}); links from {Path(args.links).name}, not measured.\n")

    table = {}
    for model in args.models.split(","):
        size, seq, G = WORKLOADS[model]
        shape = SIZES[size]
        comp = StageCompute(shape, seq, machine, shape_times)
        total_p, _ = shape.params()
        table[model] = {}
        for sysname in systems:
            system = cfg["systems"][sysname]
            best, pure = [], {}
            print(f"{model} ({total_p / 1e9:.2f}B params, seq {seq}, global batch {G} = {G * seq / 1e6:.1f}M tokens), "
                  f"{sysname}")
            for n in gpus:
                b = search(shape, seq, G, n, comp, links, system, cfg)
                best.append(b)
                for r in ("dp", "tp+dp", "pp+dp"):
                    pure.setdefault(r, []).append(search(shape, seq, G, n, comp, links, system, cfg, restrict=r))
                if b:
                    bd = b["breakdown_s"]
                    worst = max((k for k in bd if k != "compute"), key=lambda k: bd[k])
                    print(f"  {n:>5} GPUs: {lay(b):<26} step {b['step_s'] * 1e3:9.1f} ms  "
                          f"{b['tokens_per_s']:>12,.0f} tok/s  MFU {100 * b['mfu']:5.1f}%  mem {b['mem_gb']:5.1f} GB  "
                          f"biggest overhead: {worst} {1e3 * bd[worst]:.1f} ms")
                else:
                    print(f"  {n:>5} GPUs: none ({why_none(shape, seq, G, n, cfg)})")
                alone = "  ".join(f"{r} only: " + (f"{100 * pure[r][-1]['mfu']:.1f}% ({lay(pure[r][-1])})"
                                                   if pure[r][-1] else "does not fit")
                                  for r in ("dp", "tp+dp", "pp+dp"))
                print(f"         MFU with {alone}")
                if args.show_top:
                    for r in all_layouts(shape, seq, G, n, comp, links, system, cfg)[: args.show_top]:
                        print(f"        {lay(r)}: {r['tokens_per_s']:,.0f} tok/s, MFU {100 * r['mfu']:.1f}%")
            table[model][sysname] = {"best": best, "restricted": pure}
            print()

    # Scaling efficiency against perfect linear scaling from the smallest GPU count.
    for model, per_sys in table.items():
        for sysname, d in per_sys.items():
            rows = [r for r in d["best"] if r]
            if rows:
                base = rows[0]
                for r in rows:
                    r["scaling_efficiency"] = (r["tokens_per_s"] / base["tokens_per_s"]) / (r["gpus"] / base["gpus"])

    out = {"prediction": True, "arch": machine.arch, "machine_label": args.machine_label,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "links": cfg, "workloads": WORKLOADS,
           "gpu_counts": gpus, "results": table}
    path = RESULTS / f"scaling_{args.machine_label}.json"
    path.write_text(json.dumps(out, indent=1))
    plot(table, args.machine_label, machine.arch, systems)
    for model in ("llama3-8b", "llama3-70b"):
        if model in table and "nvlink_ib" in table[model]:
            plot_breakdown(table[model]["nvlink_ib"]["best"], args.machine_label,
                           f"PREDICTED step time breakdown, {model} shape, nvlink_ib, best layout per GPU count",
                           f"scaling_breakdown_{model}_{args.machine_label}.png")
    print(f"Saved {path.name}, scaling_curves_{args.machine_label}.png")


if __name__ == "__main__":
    main()
