"""Roofline performance model of one training step.

Builds an inventory of every GPU operation in the forward pass, backward pass and
optimizer step of the Llama style model in bench_train.py, with its FLOPs, bytes moved
and kernel launches, and predicts the step time from the calibration numbers:

    time per op = max(FLOPs / peak matmul speed, bytes / memory bandwidth) + launches x launch overhead

Two inventories match the two code paths of the benchmark:
  kernels="torch"   the PyTorch model (ladder rows 1 to 4): every eager kernel, including
                    autocast casts, the multi pass RMSNorm and RoPE, and the full logits.
  kernels="triton"  the fused model (ladder row 6): the Triton kernels from section 8.

Two prediction levels:
  roofline     peak numbers only (calibration JSON): what the hardware allows.
  calibrated   each GEMM and attention call timed at its real shape on this GPU (cached
               in results/shape_calibration_<label>.json); elementwise ops stay roofline.

The inventory takes any model shape, including grouped query attention (kv_heads) and a
tensor parallel degree (tp), so the scaling model in scaling.py can reuse it.

Examples
  python perf_model/roofline.py --size medium --batch 4 --seq 1024 --kernels triton
  python perf_model/roofline.py --size medium --batch 4 --seq 1024 --what-if bw=2 --what-if launch=0
  python perf_model/roofline.py --d 4096 --L 32 --H 32 --kv-heads 8 --ffn 14336 --vocab 128256 --batch 1 --seq 8192
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

CATEGORIES = ["matmul", "attention", "elementwise", "copy/cast", "loss", "optimizer"]
CE_CHUNK = 2048


# --------------------------------------------------------------------------
# Model shape
# --------------------------------------------------------------------------

@dataclass
class Shape:
    d: int
    L: int
    H: int
    ffn: int
    vocab: int = 32000
    kv_heads: int = None      # grouped query attention; None means kv_heads == H

    @property
    def hd(self):
        return self.d // self.H

    @property
    def kvh(self):
        return self.kv_heads or self.H

    def params(self):
        """Total and matmul parameters (all but the embedding), as in bench_train.count_params."""
        qkv = self.d * (self.H + 2 * self.kvh) * self.hd
        per_block = qkv + self.d * self.d + 3 * self.d * self.ffn + 2 * self.d
        total = self.L * per_block + self.d + 2 * self.vocab * self.d
        return total, total - self.vocab * self.d

    def model_flops_per_token(self, seq):
        """Spec definition: 6 x matmul params + 12 x L x d x seq."""
        return 6 * self.params()[1] + 12 * self.L * self.d * seq


SIZES = {
    "tiny": Shape(d=512, L=8, H=8, ffn=1408),
    "small": Shape(d=768, L=12, H=12, ffn=2048),
    "medium": Shape(d=1024, L=24, H=16, ffn=2816),
    "large": Shape(d=2048, L=16, H=16, ffn=5504),
    # Public Llama 3 shapes (grouped query attention, 128K vocabulary).
    "llama3-8b": Shape(d=4096, L=32, H=32, ffn=14336, vocab=128256, kv_heads=8),
    "llama3-70b": Shape(d=8192, L=80, H=64, ffn=28672, vocab=128256, kv_heads=8),
}


# --------------------------------------------------------------------------
# Operation inventory
# --------------------------------------------------------------------------

@dataclass
class Op:
    name: str
    cat: str
    phase: str              # fwd, bwd, opt
    flops: float = 0.0      # per call
    bytes: float = 0.0      # per call, all traffic of all its kernels
    launches: int = 1       # kernels per call
    count: int = 1          # calls per step
    key: tuple = None       # shape key for calibrated timing (GEMMs and attention)
    dram_bytes: float = None  # per call, traffic that must reach DRAM (first read, last write)
    l2_set: float = 0.0     # size of the intermediates passed between its kernels

    @property
    def total_flops(self):
        return self.flops * self.count

    @property
    def total_bytes(self):
        return self.bytes * self.count

    @property
    def total_launches(self):
        return self.launches * self.count


def _gemm(name, M, N, K, a, out_bytes, phase, count, layout, batch=1, cat="matmul"):
    """C (M, N) = A (M, K) @ B (K, N). layout says whether A and B are plain or transposed
    views (n or t), which is how the real call reads them; it only matters for timing."""
    return Op(name, cat, phase, flops=2.0 * M * N * K * batch,
              bytes=batch * ((M * K + K * N) * a + M * N * out_bytes), launches=1, count=count,
              key=("gemm", batch, M, N, K, layout, a))


def _ew(name, elems, bytes_per_elem, launches, phase, count, cat="elementwise", unique=None):
    """Memory bound op over elems values. bytes_per_elem counts the traffic of all its kernels.
    unique (bytes per value) is what has to come from or go to DRAM: the op's real inputs
    and outputs. The rest is intermediates between its kernels, written by one and read by
    the next; if they fit in L2 they never reach DRAM. One fused kernel has unique = all."""
    unique = bytes_per_elem if unique is None else unique
    return Op(name, cat, phase, bytes=float(elems) * bytes_per_elem, launches=launches, count=count,
              dram_bytes=float(elems) * unique, l2_set=float(elems) * 4)


def inventory(s, B, T, precision="bf16", attn="sdpa", kernels="torch", optimizer="fused", tp=1):
    """Every GPU op of one training step, as bench_train.py runs it.

    tp > 1 gives the per GPU work under Megatron style tensor parallelism: heads and the
    MLP width are split across tp GPUs; norms, residual adds and the embedding are not.
    (The lm head is vocab parallel.) Communication is not included here; scaling.py adds it.
    """
    bf = precision == "bf16"
    a = 2 if bf else 4                         # activation bytes
    fused = kernels != "torch"
    if fused and not bf:
        raise ValueError("the fused kernels path is modeled for bf16 only")
    if attn == "naive" and s.kvh != s.H:
        raise ValueError("naive attention is modeled without grouped query attention")

    N = B * T
    d, L, V = s.d, s.L, s.vocab // tp
    H, hd, kv = s.H // tp, s.hd, max(1, s.kvh // tp)
    f = s.ffn // tp
    dq, dkv = H * hd, kv * hd
    dqkv = dq + 2 * dkv
    n = N * d
    ops = []
    add = ops.append

    linears = [("qkv", dqkv, d), ("wo", d, dq), ("w1", f, d), ("w3", f, d), ("w2", d, f)]   # (out, in)

    # ---------------- forward, per layer ----------------
    if bf:
        for nm, o, i in linears:
            add(_ew(f"autocast cast {nm} weight fp32 to bf16", o * i, 6, 1, "fwd", L, "copy/cast"))

    def norm_fwd(name, count):
        if fused:
            add(_ew(f"{name} rmsnorm (Triton)", n, 4 + a, 1, "fwd", count))
        else:   # pow, mean, add, rsqrt, mul, mul (and the cast to bf16)
            add(_ew(f"{name} rmsnorm (eager, {7 if bf else 6} kernels)", n, 34 if bf else 28, 7 if bf else 6, "fwd",
                    count, unique=4 + a))   # in: x fp32, out: y

    norm_fwd("attn", L)
    add(_gemm("qkv projection", N, dqkv, d, a, a, "fwd", L, "nt"))
    if fused:
        add(_ew("rope_qkv (Triton, splits q k v)", N * dqkv, 2 * a, 1, "fwd", L))
    else:      # per q and k: cast, neg, cat, mul cos, mul sin, add, cast back
        add(_ew("rope on q (eager, 7 kernels)", N * dq, 52 if bf else 40, 7 if bf else 5, "fwd", L, unique=2 * a))
        add(_ew("rope on k (eager, 7 kernels)", N * dkv, 52 if bf else 40, 7 if bf else 5, "fwd", L, unique=2 * a))

    if attn == "sdpa":
        add(Op("flash attention forward", "attention", "fwd", flops=2.0 * B * T * T * dq,
               bytes=(2 * N * dq + 2 * N * dkv) * a + B * H * T * 4, launches=1, count=L,
               key=("sdpa_fwd", B, H, kv, T, hd)))
    else:
        S = B * H * T * T                     # score matrix elements per layer
        add(_gemm("scores q @ k^T", T, T, hd, a, a, "fwd", L, "nt", batch=B * H, cat="attention"))
        add(_ew("scale scores", S, 2 * a, 1, "fwd", L, "attention"))
        add(_ew("causal mask fill", S, 2 * a, 1, "fwd", L, "attention"))
        add(_ew("softmax (with fp32 casts)" if bf else "softmax", S, 20 if bf else 8, 3 if bf else 1, "fwd", L,
                "attention", unique=2 * a))
        add(_gemm("probs @ v", T, hd, T, a, a, "fwd", L, "nn", batch=B * H, cat="attention"))
    add(_ew("attention output transpose copy", N * dq, 2 * a, 1, "fwd", L, "copy/cast"))
    add(_gemm("output projection", N, d, dq, a, a, "fwd", L, "nt"))
    if fused:
        add(_ew("add_rmsnorm (Triton, residual add + norm)", n, 8 + 2 * a, 1, "fwd", L))
    else:
        add(_ew("residual add (fp32 + bf16)" if bf else "residual add", n, 10 if bf else 12, 1, "fwd", L))
        norm_fwd("mlp", L)
    add(_gemm("w1 (gate)", N, f, d, a, a, "fwd", L, "nt"))
    add(_gemm("w3 (up)", N, f, d, a, a, "fwd", L, "nt"))
    if fused:
        add(_ew("swiglu (Triton)", N * f, 3 * a, 1, "fwd", L))
    else:
        add(_ew("silu, multiply (eager, 2 kernels)", N * f, 5 * a, 2, "fwd", L, unique=3 * a))
    add(_gemm("w2 (down)", N, d, f, a, a, "fwd", L, "nt"))
    add(_ew("residual add (fp32 + bf16)" if bf else "residual add", n, 10 if bf else 12, 1, "fwd", L))

    # ---------------- backward, per layer ----------------
    for nm, o, i in linears:
        add(_gemm(f"{nm} input gradient", N, i, o, a, a, "bwd", L, "nn"))
        add(_gemm(f"{nm} weight gradient", o, i, N, a, a, "bwd", L, "tn"))
        if bf:
            add(_ew(f"{nm} weight gradient cast bf16 to fp32", o * i, 6, 1, "bwd", L, "copy/cast"))
    if bf:
        add(_ew("residual branch gradient cast fp32 to bf16", n, 6, 1, "bwd", (1 if fused else 2) * L, "copy/cast"))
    add(_ew("residual stream gradient accumulate", n, 12, 1, "bwd", (1 if fused else 2) * L))
    if fused:
        add(_ew("rmsnorm backward (Triton + dw sum)", n, 8 + a, 2, "bwd", L))
        add(_ew("add_rmsnorm backward (Triton + dw sum)", n, 12 + 2 * a, 2, "bwd", L))
        add(_ew("swiglu backward (Triton)", N * f, 5 * a, 1, "bwd", L))
        add(_ew("rope_qkv backward (Triton, packs dqkv)", N * dqkv, 2 * a, 1, "bwd", L))
    else:
        # in: dy, saved x (fp32); out: dx (fp32)
        add(_ew("rmsnorm backward (eager, about 12 kernels)", n, 90 if bf else 84, 12, "bwd", 2 * L, unique=a + 8))
        add(_ew("silu, multiply backward (eager, 3 kernels)", N * f, 9 * a, 3, "bwd", L, unique=5 * a))
        add(_ew("rope backward on q (eager)", N * dq, 60 if bf else 48, 8, "bwd", L, unique=2 * a))
        add(_ew("rope backward on k (eager)", N * dkv, 60 if bf else 48, 8, "bwd", L, unique=2 * a))
        add(_ew("q k v gradients packed into qkv layout", N * dqkv, 2 * a, 3, "bwd", L, "copy/cast"))
    add(_ew("attention output transpose copy backward", N * dq, 2 * a, 1, "bwd", L, "copy/cast"))
    if attn == "sdpa":
        add(Op("flash attention backward", "attention", "bwd", flops=5.0 * B * T * T * dq,
               bytes=(4 * N * dq + 4 * N * dkv) * a + N * dq * a * 2 + N * dq * 8, launches=3, count=L,
               key=("sdpa_bwd", B, H, kv, T, hd)))
    else:
        S = B * H * T * T
        add(_gemm("d probs = dy @ v^T", T, T, hd, a, a, "bwd", L, "nt", batch=B * H, cat="attention"))
        add(_gemm("dv = probs^T @ dy", T, hd, T, a, a, "bwd", L, "tn", batch=B * H, cat="attention"))
        add(_ew("softmax backward (with fp32 casts)" if bf else "softmax backward", S, 24 if bf else 12,
                3 if bf else 1, "bwd", L, "attention", unique=3 * a))
        add(_ew("mask fill and scale backward", S, 4 * a, 2, "bwd", L, "attention"))
        add(_gemm("dq = dscores @ k", T, hd, T, a, a, "bwd", L, "nn", batch=B * H, cat="attention"))
        add(_gemm("dk = dscores^T @ q", T, hd, T, a, a, "bwd", L, "tn", batch=B * H, cat="attention"))

    # ---------------- embedding, final norm, lm head, loss ----------------
    add(_ew("embedding gather", n, 8, 1, "fwd", 1, "loss"))
    if fused:
        add(_ew("final rmsnorm (Triton)", n, 4 + a, 1, "fwd", 1))
        add(_ew("final rmsnorm backward (Triton)", n, 8 + a, 2, "bwd", 1))
    else:
        add(_ew("final rmsnorm (eager)", n, 34 if bf else 28, 7 if bf else 6, "fwd", 1, unique=4 + a))
        add(_ew("final rmsnorm backward (eager)", n, 90 if bf else 84, 12, "bwd", 1, unique=a + 8))
    if bf:
        add(_ew("lm head weight cast fp32 to bf16", V * d, 6, 1, "fwd", 1, "copy/cast"))
    if fused:
        chunks = math.ceil(N / CE_CHUNK)
        Nc = N // chunks
        add(_gemm("lm head logits (per chunk)", Nc, V, d, a, a, "fwd", chunks, "nt"))
        add(_ew("cross entropy and logit gradient in place (Triton)", Nc * V, 2 * a, 1, "fwd", chunks, "loss"))
        add(_gemm("lm head input gradient (per chunk)", Nc, d, V, a, a, "fwd", chunks, "nn"))
        add(_gemm("lm head weight gradient, fp32 accumulate (per chunk)", V, d, Nc, a, 8, "fwd", chunks, "tn"))
        add(_ew("scale saved gradients by the incoming gradient", n, 2 * a, 1, "bwd", 1, "loss"))
        add(_ew("lm head gradient scale and cast", V * d, 8, 1, "bwd", 1, "copy/cast"))
    else:
        add(_gemm("lm head logits", N, V, d, a, a, "fwd", 1, "nt"))
        if bf:
            add(_ew("logits cast bf16 to fp32", N * V, 6, 1, "fwd", 1, "copy/cast"))
        add(_ew("log softmax and nll loss", N * V, 8, 2, "fwd", 1, "loss"))
        add(_ew("nll backward (zero fill and scatter)", N * V, 4, 2, "bwd", 1, "loss"))
        add(_ew("log softmax backward", N * V, 12, 1, "bwd", 1, "loss"))
        if bf:
            add(_ew("logit gradient cast fp32 to bf16", N * V, 6, 1, "bwd", 1, "copy/cast"))
        add(_gemm("lm head input gradient", N, d, V, a, a, "bwd", 1, "nn"))
        add(_gemm("lm head weight gradient", V, d, N, a, a, "bwd", 1, "tn"))
        if bf:
            add(_ew("lm head weight gradient cast bf16 to fp32", V * d, 6, 1, "bwd", 1, "copy/cast"))
    add(_ew("embedding backward (zero fill, sort, index add)", s.vocab * d // tp, 4, 1, "bwd", 1, "loss"))
    add(_ew("embedding backward index add", n, 8, 6, "bwd", 1, "loss"))

    # ---------------- optimizer ----------------
    P = sum(o * i for _, o, i in linears) * L + 2 * d * L + d + V * d + s.vocab * d // tp
    tensors = 7 * L + 3
    if optimizer == "fused":
        # one pass: read param, grad, exp_avg, exp_avg_sq (16 B), write param and both moments (12 B)
        add(_ew("fused AdamW", P, 28, max(1, tensors // 8), "opt", 1, "optimizer"))
    else:
        # foreach: weight decay, lerp, mul, addcmul, sqrt, div, add, addcdiv as separate passes
        add(_ew("multi tensor AdamW (8 passes)", P, 84, int(0.85 * tensors), "opt", 1, "optimizer"))
    return ops


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

@dataclass
class Machine:
    peak_tflops: dict            # per precision, measured
    bandwidth_gbs: float         # DRAM, measured with a 1 GB copy
    launch_us: float             # CPU time to issue one kernel from PyTorch eager (calibration)
    label: str = ""
    arch: str = ""
    gpu_gap_us: float = 1.0      # GPU time between back to back kernels (measured with a CUDA graph)
    l2_gbs: float = None         # bandwidth for data that stays in L2 (None: same as DRAM)
    l2_mb: float = 0.0           # L2 capacity

    @classmethod
    def from_calibration(cls, path, extra=None):
        """extra: optional dict with gpu_gap_us, l2_gbs and l2_mb (from measure_machine_extras)."""
        cal = json.loads(Path(path).read_text())
        peaks = {k: v["tflops"] for k, v in cal["matmul"].items() if v.get("status") == "ok"}
        m = cls(peaks, cal["bandwidth_gbs"], cal.get("launch_overhead_us", 0.0),
                cal.get("machine_label", ""), cal.get("arch", ""))
        for k, v in (extra or {}).items():
            setattr(m, k, v)
        return m

    def scaled(self, bw=1.0, peak=1.0, launch=1.0):
        """What if: multiply memory bandwidth (DRAM and L2), matmul peak, or launch costs."""
        return Machine({k: v * peak for k, v in self.peak_tflops.items()}, self.bandwidth_gbs * bw,
                       self.launch_us * launch, self.label, self.arch, self.gpu_gap_us * launch,
                       (self.l2_gbs * bw) if self.l2_gbs else None, self.l2_mb)


@dataclass
class Prediction:
    total_ms: float               # max(GPU time, CPU issue time)
    by_cat: dict = field(default_factory=dict)        # GPU ms per category
    gpu_ms: float = 0.0           # sum of op times plus kernel gaps
    gap_ms: float = 0.0           # launches x GPU gap per kernel
    cpu_ms: float = 0.0           # launches x CPU issue time per kernel
    launches: int = 0
    flops: float = 0.0
    bytes: float = 0.0
    dram_bytes: float = 0.0
    ops: list = field(default_factory=list)

    @property
    def cpu_bound(self):
        return self.cpu_ms > self.gpu_ms

    @property
    def launch_ms(self):
        """Time the step loses to kernel launches: the GPU gaps, plus any CPU bound excess."""
        return self.total_ms - sum(self.by_cat.values())


def predict(ops, machine, precision="bf16", shape_times=None):
    """Predicted step time.

    Per op: max(FLOPs / peak, memory time). Memory time counts the op's DRAM traffic at DRAM
    bandwidth and, for multi kernel eager ops whose intermediates fit in half of L2, the
    intermediate traffic at L2 bandwidth. shape_times (op key -> measured seconds) replaces
    the roofline for GEMMs and attention when given.

    Launches: the GPU spends gpu_gap_us between kernels; the CPU needs launch_us to issue
    each one. The CPU runs ahead of the GPU, so the step takes the longer of the two.
    """
    peak = machine.peak_tflops["bf16" if precision == "bf16" else "fp32"] * 1e12
    gemm_eff = gemm_efficiency(shape_times, peak) if shape_times else 1.0
    bw = machine.bandwidth_gbs * 1e9
    l2bw = (machine.l2_gbs or machine.bandwidth_gbs) * 1e9
    l2cap = machine.l2_mb * 2 ** 20 * 0.5
    p = Prediction(0.0, {c: 0.0 for c in CATEGORIES})
    for op in ops:
        t_c = op.flops / peak
        dram = op.bytes if op.dram_bytes is None else op.dram_bytes
        inter = op.bytes - dram
        if inter > 0 and op.l2_set > l2cap:     # intermediates do not fit, they go to DRAM too
            dram, inter = op.bytes, 0.0
        t_m = dram / bw + inter / l2bw
        if shape_times and op.key in shape_times:
            t = shape_times[op.key]
            bound = "measured"
        elif op.key is not None and gemm_eff < 1.0:
            # GEMM or attention shape that was not timed: peak scaled by the efficiency
            # measured on the largest timed GEMMs.
            t = max(t_c / gemm_eff, t_m)
            bound = "estimated"
        else:
            t = max(t_c, t_m)
            bound = "compute" if t_c >= t_m else "memory"
        t_total = t * op.count
        p.by_cat[op.cat] += t_total * 1e3
        p.launches += op.total_launches
        p.flops += op.total_flops
        p.bytes += op.total_bytes
        p.dram_bytes += dram * op.count
        ai = op.flops / op.bytes if op.bytes else float("inf")
        p.ops.append(dict(name=op.name, cat=op.cat, phase=op.phase, count=op.count, ms=round(t_total * 1e3, 4),
                          gflops=round(op.total_flops / 1e9, 3), mbytes=round(op.total_bytes / 1e6, 3),
                          dram_mbytes=round(dram * op.count / 1e6, 3),
                          intensity=round(ai, 2) if math.isfinite(ai) else None, bound=bound,
                          launches=op.total_launches))
    p.gap_ms = p.launches * machine.gpu_gap_us / 1e3
    p.gpu_ms = sum(p.by_cat.values()) + p.gap_ms
    p.cpu_ms = p.launches * machine.launch_us / 1e3
    p.total_ms = max(p.gpu_ms, p.cpu_ms)
    return p


def gemm_efficiency(shape_times, peak_flops, min_gflop=20.0):
    """FLOP weighted efficiency (achieved / peak) of the timed bf16 GEMMs of at least min_gflop.
    Used for GEMM shapes that were not timed, such as Llama 3 shapes too big for one GPU."""
    flops = secs = 0.0
    for key, t in shape_times.items():
        if key[0] != "gemm" or key[6] != 2:
            continue
        _, batch, M, N, K = key[:5]
        f = 2.0 * batch * M * N * K
        if f >= min_gflop * 1e9:
            flops += f
            secs += t
    return min(1.0, flops / secs / peak_flops) if secs else 1.0


def measure_machine_extras(cache_path=None):
    """GPU side gap between kernels (1000 tiny kernels replayed from a CUDA graph, so there is
    no CPU cost), L2 bandwidth (copy of a 16 MB buffer that stays in L2) and L2 size."""
    if cache_path and Path(cache_path).exists():
        data = json.loads(Path(cache_path).read_text())
        if "extras" in data:
            return data["extras"]
    import torch
    x = torch.zeros(1, device="cuda")
    n = 1000
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x.add_(1.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            x.add_(1.0)
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gap = 1e9
    for _ in range(5):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        gap = min(gap, e0.elapsed_time(e1) * 1e3 / n)
    nbytes = 16 * 2 ** 20
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        e0.record()
        for _ in range(100):
            dst.copy_(src)
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / 1e3 / 100)
    extras = {"gpu_gap_us": round(gap, 3), "l2_gbs": round(2 * nbytes / best / 1e9, 1),
              "l2_mb": torch.cuda.get_device_properties(0).L2_cache_size / 2 ** 20}
    if cache_path:
        data = json.loads(Path(cache_path).read_text()) if Path(cache_path).exists() else {"times": {}}
        data["extras"] = extras
        Path(cache_path).write_text(json.dumps(data, indent=1))
    return extras


# --------------------------------------------------------------------------
# Shape calibration: time each GEMM and attention call at its real shape
# --------------------------------------------------------------------------

def measure_shape_times(ops, cache_path=None):
    """Seconds per call for every distinct GEMM and attention key in ops, measured on this GPU
    with triton.testing.do_bench (L2 cleared before each call, median). Results are cached."""
    import torch
    import torch.nn.functional as F
    from triton.testing import do_bench

    cache = {}
    if cache_path and Path(cache_path).exists():
        cache = {tuple(json.loads(k)): v for k, v in json.loads(Path(cache_path).read_text())["times"].items()}
    keys = sorted({op.key for op in ops if op.key is not None and op.key not in cache}, key=str)
    torch.backends.cuda.matmul.allow_tf32 = False
    for key in keys:
        try:
            if key[0] == "gemm":
                _, batch, M, N, K, layout, a = key
                dt = torch.bfloat16 if a == 2 else torch.float32
                A = torch.randn(batch, K, M, device="cuda", dtype=dt).transpose(1, 2) if layout[0] == "t" \
                    else torch.randn(batch, M, K, device="cuda", dtype=dt)
                Bm = torch.randn(batch, N, K, device="cuda", dtype=dt).transpose(1, 2) if layout[1] == "t" \
                    else torch.randn(batch, K, N, device="cuda", dtype=dt)
                if batch == 1:
                    A, Bm = A[0], Bm[0]
                fn = lambda: torch.matmul(A, Bm)
                cache[key] = do_bench(fn, warmup=10, rep=50, return_mode="median") / 1e3
            elif key[0] in ("sdpa_fwd", "sdpa_bwd"):
                _, B, H, kv, T, hd = key
                q = torch.randn(B, H, T, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                k = torch.randn(B, kv, T, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                v = torch.randn(B, kv, T, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                gqa = kv != H
                fwd = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
                t_f = do_bench(fwd, warmup=10, rep=50, return_mode="median") / 1e3
                o = fwd()
                do = torch.randn_like(o)
                bwd = lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True)
                t_b = do_bench(bwd, warmup=10, rep=50, return_mode="median") / 1e3
                cache[("sdpa_fwd", B, H, kv, T, hd)] = t_f
                cache[("sdpa_bwd", B, H, kv, T, hd)] = t_b
        except Exception as e:  # out of memory for a huge shape: fall back to the roofline for it
            print(f"  shape timing skipped for {key}: {type(e).__name__}")
        torch.cuda.empty_cache()
    if cache_path:
        data = json.loads(Path(cache_path).read_text()) if Path(cache_path).exists() else {}
        data["times"] = {json.dumps(list(k)): v for k, v in cache.items()}
        Path(cache_path).write_text(json.dumps(data, indent=1))
    return cache


def load_shape_times(cache_path):
    """Cached shape timings without touching the GPU (for runs on a machine without one)."""
    if not cache_path or not Path(cache_path).exists():
        return {}
    data = json.loads(Path(cache_path).read_text())
    return {tuple(json.loads(k)): v for k, v in data.get("times", {}).items()}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def summarize(p, title, model_flops=None, peak_tflops=None):
    lines = [f"{title}: predicted {p.total_ms:.2f} ms per step "
             f"({'CPU launch bound' if p.cpu_bound else 'GPU bound'}: GPU {p.gpu_ms:.2f} ms, CPU issue {p.cpu_ms:.2f} ms)"]
    for c in CATEGORIES:
        if p.by_cat.get(c):
            lines.append(f"  {c:<12} {p.by_cat[c]:9.2f} ms  ({100 * p.by_cat[c] / p.total_ms:4.1f}%)")
    lines.append(f"  {'launches':<12} {p.launch_ms:9.2f} ms  ({p.launches} kernels)")
    if model_flops and peak_tflops:
        mfu = model_flops / (p.total_ms / 1e3) / (peak_tflops * 1e12)
        lines.append(f"  predicted MFU {100 * mfu:.1f}%  (model FLOPs {model_flops / 1e12:.2f} T per step)")
    return "\n".join(lines)


def _scale_shape_times(shape_times, kw):
    """For a what if on the matmul peak, measured GEMM and attention times shrink by the same
    factor (same efficiency on faster hardware). Bandwidth and launch what ifs leave them alone."""
    if not shape_times or "peak" not in kw:
        return shape_times
    f = kw["peak"]
    if f <= 0:
        raise SystemExit("peak multiplier must be positive")
    return {k: v / f for k, v in shape_times.items()}


def parse_what_if(items):
    """['bw=2', 'peak=1.5', 'launch=0'] -> list of (label, kwargs) for Machine.scaled."""
    out = []
    for item in items:
        k, _, v = item.partition("=")
        if k not in ("bw", "peak", "launch"):
            raise SystemExit(f"--what-if takes bw=, peak= or launch=, not {item}")
        out.append((item, {k: float(v)}))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="medium", choices=list(SIZES))
    for k in ("d", "L", "H", "ffn", "vocab", "kv-heads"):
        p.add_argument(f"--{k}", type=int, default=None, help="override one dimension of --size")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "naive"])
    p.add_argument("--kernels", default="both", choices=["torch", "triton", "both"])
    p.add_argument("--optimizer", default="fused", choices=["fused", "foreach"])
    p.add_argument("--calibration", default=str(RESULTS / "calibration_final.json"))
    p.add_argument("--what-if", action="append", default=[], help="bw=X, peak=X or launch=X (multipliers)")
    p.add_argument("--measure-shapes", action="store_true",
                   help="time this config's GEMM and attention shapes on the GPU (else use the cache, if any)")
    p.add_argument("--top", type=int, default=12, help="print the N slowest ops")
    p.add_argument("--json", default=None, help="write the full prediction to this file")
    args = p.parse_args()

    s = SIZES[args.size]
    over = {k: getattr(args, k.replace("-", "_")) for k in ("d", "L", "H", "ffn", "vocab", "kv-heads")}
    s = Shape(**{**asdict(s), **{k.replace("-", "_"): v for k, v in over.items() if v is not None}})
    cal_label = json.loads(Path(args.calibration).read_text()).get("machine_label", "final")
    shape_cache = RESULTS / f"shape_calibration_{cal_label}.json"
    extras = json.loads(shape_cache.read_text()).get("extras") if shape_cache.exists() else None
    m = Machine.from_calibration(args.calibration, extras)
    shape_times = None
    if args.measure_shapes:
        shape_times = measure_shape_times(
            sum((inventory(s, args.batch, args.seq, args.precision, args.attn, k, args.optimizer)
                 for k in ("torch", "triton")), []), shape_cache)
    elif shape_cache.exists():
        shape_times = load_shape_times(shape_cache)
    total, matmul = s.params()
    mf = s.model_flops_per_token(args.seq) * args.batch * args.seq
    print(f"shape {asdict(s)} ({total / 1e6:.0f}M params), batch {args.batch}, seq {args.seq}, {args.precision}")
    print(f"machine {m.arch} ({m.label}): bf16 {m.peak_tflops.get('bf16')} TFLOPs, fp32 {m.peak_tflops.get('fp32')} "
          f"TFLOPs, DRAM {m.bandwidth_gbs} GB/s, L2 {m.l2_gbs} GB/s ({m.l2_mb} MB), CPU issue {m.launch_us} us "
          f"and GPU gap {m.gpu_gap_us} us per kernel")
    if shape_times:
        keys = {o.key for k in ("torch", "triton")
                for o in inventory(s, args.batch, args.seq, args.precision, args.attn, k, args.optimizer) if o.key}
        hit = len(keys & set(shape_times))
        eff = gemm_efficiency(shape_times, m.peak_tflops["bf16"] * 1e12)
        print(f"GEMM and attention timing: {hit} of {len(keys)} shapes measured on this GPU; the rest use the peak "
              f"times the measured large GEMM efficiency ({100 * eff:.0f}%)\n")
    else:
        print("GEMM and attention timing: roofline at peak (no shape calibration found)\n")

    modes = ["torch", "triton"] if args.kernels == "both" else [args.kernels]
    out = {"shape": asdict(s), "batch": args.batch, "seq": args.seq, "machine": asdict(m), "modes": {}}
    for mode in modes:
        ops = inventory(s, args.batch, args.seq, args.precision, args.attn, mode, args.optimizer)
        pr = predict(ops, m, args.precision, shape_times)
        name = "unfused (PyTorch ops)" if mode == "torch" else "fused (Triton kernels)"
        print(summarize(pr, name, mf, m.peak_tflops["bf16"]))
        top = sorted(pr.ops, key=lambda o: -o["ms"])[: args.top]
        for o in top:
            print(f"    {o['ms']:8.3f} ms  {o['bound']:<7}  AI {o['intensity'] if o['intensity'] is not None else '-':>8}"
                  f"  {o['name']}")
        for label, kw in parse_what_if(args.what_if):
            # Measured GEMM times would not respond to a faster peak, so what if runs use the roofline
            # for GEMMs scaled by the measured efficiency of this config.
            alt = predict(ops, m.scaled(**kw), args.precision, _scale_shape_times(shape_times, kw))
            print(f"  what if {label}: {alt.total_ms:.2f} ms ({100 * (alt.total_ms / pr.total_ms - 1):+.1f}%)")
        print()
        out["modes"][mode] = {"total_ms": pr.total_ms, "gpu_ms": pr.gpu_ms, "cpu_ms": pr.cpu_ms,
                              "by_cat": pr.by_cat, "launch_ms": pr.launch_ms, "launches": pr.launches,
                              "flops": pr.flops, "bytes": pr.bytes, "dram_bytes": pr.dram_bytes, "ops": pr.ops}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
