"""JAX port of the Llama style model in bench_train.py, with jax.jit.

The model is a plain function of a parameter pytree, written to match the PyTorch model:
pre norm blocks with RMSNorm, packed qkv linear, RoPE (rotate half, base 10000), causal
attention, SwiGLU MLP, untied embedding and lm head, vocab 32000, no biases, weights
normal with std 0.02. Mixed precision matches torch.autocast with bf16: fp32 master
weights, matmuls and attention in bf16, norms, residual stream and loss in fp32.
AdamW (betas 0.9 and 0.95, weight decay 0.1, lr 3e-4, eps 1e-8) is written out here, so
no optimizer library is needed.

Measured, each with warmup and the median of timed steps (block_until_ready):
  model   full training step (forward, backward, AdamW), one jit, buffers donated
  block   one transformer block, forward plus backward, on a (B, T, d) input
Plus the XLA view of the compiled step: fusion and custom call counts from the optimized
HLO, and kernel level GPU time per bucket from a jax.profiler trace.

Runs in its own venv (JAX's CUDA wheels pin their own NVIDIA libraries):
  python3 -m venv ~/venvs/jax && source ~/venvs/jax/bin/activate
  pip install "jax[cuda12]==0.9.2" numpy==2.5.4 tabulate==0.10.0 matplotlib==3.11.2
  XLA_PYTHON_CLIENT_PREALLOCATE=false python jax_port/block_jax.py --size medium --batch 4 --seq 1024
"""

import argparse
import glob
import gzip
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# JAX is pinned to 0.9.2. On WSL2, JAX 0.10 and newer cannot allocate any single buffer
# above about 4 GB, even on an empty 32 GB card, with every allocator setting tried
# (jax-ml/jax issue 41448). 0.9.2 allocates normally and supports sm_120.
# Preallocation is off so the reported peak memory is what the step really uses.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PROFILES = ROOT / "profiles"

SIZES = {
    "tiny": dict(d=512, L=8, H=8, ffn=1408),
    "small": dict(d=768, L=12, H=12, ffn=2048),
    "medium": dict(d=1024, L=24, H=16, ffn=2816),
    "large": dict(d=2048, L=16, H=16, ffn=5504),
}
VOCAB = 32000
EPS = 1e-5
BF16 = jnp.bfloat16


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def init_params(key, cfg):
    d, L, ffn = cfg["d"], cfg["L"], cfg["ffn"]
    keys = iter(jax.random.split(key, 3 + 5 * L))
    nrm = lambda shape: 0.02 * jax.random.normal(next(keys), shape, jnp.float32)
    blocks = []
    for _ in range(L):
        blocks.append({
            "norm1": jnp.ones((d,), jnp.float32), "norm2": jnp.ones((d,), jnp.float32),
            # weights stored (in, out) so x @ w needs no transpose
            "qkv": nrm((d, 3 * d)), "wo": nrm((d, d)),
            "w1": nrm((d, ffn)), "w3": nrm((d, ffn)), "w2": nrm((ffn, d)),
        })
    return {"emb": nrm((VOCAB, d)), "blocks": blocks, "norm_f": jnp.ones((d,), jnp.float32),
            "head": nrm((d, VOCAB))}


def rope_tables(T, hd):
    inv = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    freqs = jnp.outer(jnp.arange(T, dtype=jnp.float32), inv)
    emb = jnp.concatenate([freqs, freqs], axis=-1)            # (T, hd), halves duplicated
    return jnp.cos(emb), jnp.sin(emb)


def rmsnorm(x, w):
    x = x.astype(jnp.float32)
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def apply_rope(x, cos, sin):
    """x: (B, T, H, hd) bf16. Math in fp32, result bf16."""
    xf = x.astype(jnp.float32)
    x1, x2 = jnp.split(xf, 2, axis=-1)
    rot = jnp.concatenate([-x2, x1], axis=-1)
    return (xf * cos[None, :, None, :] + rot * sin[None, :, None, :]).astype(BF16)


def attention(q, k, v, impl):
    """q, k, v: (B, T, H, hd) bf16, causal."""
    if impl == "naive":
        hd = q.shape[-1]
        s = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) / math.sqrt(hd)
        T = q.shape[1]
        mask = jnp.tril(jnp.ones((T, T), bool))
        s = jnp.where(mask, s, -jnp.inf)
        p = jax.nn.softmax(s, axis=-1).astype(BF16)
        return jnp.einsum("bhqk,bkhd->bqhd", p, v)
    return jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation=impl)


def block(p, x, cos, sin, H, attn_impl):
    """One pre norm transformer block. x is the fp32 residual stream (B, T, d)."""
    B, T, d = x.shape
    hd = d // H
    h = rmsnorm(x, p["norm1"]).astype(BF16)
    qkv = h @ p["qkv"].astype(BF16)                           # (B, T, 3d), packed (3, H, hd)
    q, k, v = jnp.split(qkv.reshape(B, T, 3, H, hd), 3, axis=2)
    q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    o = attention(q, k, v, attn_impl).reshape(B, T, d)
    x = x + (o @ p["wo"].astype(BF16)).astype(jnp.float32)
    h = rmsnorm(x, p["norm2"]).astype(BF16)
    g = h @ p["w1"].astype(BF16)
    u = h @ p["w3"].astype(BF16)
    m = (jax.nn.silu(g.astype(jnp.float32)) * u.astype(jnp.float32)).astype(BF16)
    return x + (m @ p["w2"].astype(BF16)).astype(jnp.float32)


def loss_fn(params, tokens, targets, cfg, attn_impl):
    B, T = tokens.shape
    cos, sin = rope_tables(T, cfg["d"] // cfg["H"])
    x = params["emb"][tokens]                                 # fp32
    for p in params["blocks"]:
        x = block(p, x, cos, sin, cfg["H"], attn_impl)
    h = rmsnorm(x, params["norm_f"]).astype(BF16)
    logits = (h @ params["head"].astype(BF16)).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return jnp.mean(nll)


def adamw_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": jnp.zeros((), jnp.int32)}


def adamw_update(params, grads, state, lr=3e-4, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    c1 = 1 - b1 ** t.astype(jnp.float32)
    c2 = 1 - b2 ** t.astype(jnp.float32)
    params = jax.tree_util.tree_map(
        lambda p, m, v: p * (1 - lr * wd) - lr * (m / c1) / (jnp.sqrt(v / c2) + eps), params, m, v)
    return params, {"m": m, "v": v, "t": t}


def make_train_step(cfg, attn_impl):
    def step(params, opt, tokens, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets, cfg, attn_impl)
        params, opt = adamw_update(params, grads, opt)
        return params, opt, loss
    return jax.jit(step, donate_argnums=(0, 1))


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def count_params(cfg):
    d, L, ffn = cfg["d"], cfg["L"], cfg["ffn"]
    total = L * (4 * d * d + 3 * d * ffn + 2 * d) + d + 2 * VOCAB * d
    return total, total - VOCAB * d


def model_flops_per_token(cfg, T):
    return 6 * count_params(cfg)[1] + 12 * cfg["L"] * cfg["d"] * T


def bf16_peak_tflops(n=8192):
    a = jnp.ones((n, n), BF16)
    f = jax.jit(lambda a: a @ a)
    f(a).block_until_ready()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(20):
            out = f(a)
        out.block_until_ready()
        best = min(best, (time.perf_counter() - t0) / 20)
    return 2 * n ** 3 / best / 1e12


def time_fn(fn, warmup, steps):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def hlo_stats(compiled_text):
    """Fusions and custom calls in the optimized HLO of the compiled step, and how many
    GEMM fusions also do work besides the dot (an fp32 to bf16 convert of an operand, or a
    convert or elementwise op on the output) that would be a separate kernel in eager."""
    kinds = Counter(re.findall(r"kind=k(\w+)", compiled_text))
    customs = Counter(re.findall(r'custom_call_target="([^"]+)"', compiled_text))
    # Split into computations: a header line ending in "{" up to the closing "}".
    comps, cur = {}, None
    for line in compiled_text.splitlines():
        if line.rstrip().endswith("{") and line.startswith(("%", "ENTRY")):
            cur = line.split()[0].lstrip("%") if line.startswith("%") else "ENTRY"
            comps[cur] = []
        elif line.startswith("}"):
            cur = None
        elif cur:
            comps[cur].append(line)
    gemm = {k: v for k, v in comps.items() if k.startswith("gemm_fusion")}
    merged = sum(1 for v in gemm.values() if any("concatenate(" in l for l in v))
    with_convert = sum(1 for v in gemm.values() if any(" convert(" in l for l in v))
    entry = "\n".join(comps.get("ENTRY", []))
    gemm_calls = len(re.findall(r"kind=kCustom, calls=%gemm_fusion", compiled_text))
    return {"fusion_kinds": dict(kinds), "fusions": sum(kinds.values()), "custom_calls": dict(customs),
            "gemm_fusion_calls": gemm_calls, "gemm_fusion_computations": len(gemm),
            "gemm_fusions_merging_two_weights": merged, "gemm_fusions_with_convert_inside": with_convert,
            "standalone_converts": len(re.findall(r"%wrapped_convert[.\d]* = ", compiled_text)),
            "entry_instructions": entry.count("\n") + 1 if entry else 0, "hlo_chars": len(compiled_text)}


def time_optimizer(state, warmup, steps):
    """The AdamW update alone, jitted, on the training state's own parameters and moments
    (donated, so no extra copy), with constant gradients. Runs last: it changes the state."""
    params = state["params"]
    grads = jax.tree_util.tree_map(lambda p: jnp.full_like(p, 1e-3), params)
    f = jax.jit(adamw_update, donate_argnums=(0, 2))
    c = f.lower(params, grads, state["opt"]).compile()

    def run():
        state["params"], state["opt"] = c(state["params"], grads, state["opt"])
        return state["opt"]["t"]

    sec = time_fn(run, warmup, steps)
    n = sum(x.size for x in jax.tree_util.tree_leaves(state["params"]))
    kernels = len(re.findall(r"kind=k\w+", c.as_text()))
    return {"ms": round(sec * 1e3, 3), "params": n, "fusions": kernels,
            "gbs": round(n * 28 / sec / 1e9, 1)}


BUCKETS = [
    ("attention", ("flash", "fmha", "attention", "sdpa", "cudnn_sdp", "softmax")),
    ("matmul", ("gemm", "cutlass", "cublas", "nvjet", "xmma", "matmul", "dot")),
    ("memory copies", ("memcpy", "memset", "copy", "transpose")),
]


def bucket_of(name):
    low = name.lower()
    for b, keys in BUCKETS:
        if any(k in low for k in keys):
            return b
    return "fusions and other"


def profile_step(run_step, tag, steps=3):
    """GPU kernel time per bucket from a jax.profiler trace of a few steps."""
    out_dir = PROFILES / f"jax_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(out_dir))
    for _ in range(steps):
        jax.block_until_ready(run_step())
    jax.profiler.stop_trace()
    files = sorted(glob.glob(str(out_dir / "**" / "*.trace.json.gz"), recursive=True), key=os.path.getmtime)
    if not files:
        return {"error": "no trace.json.gz written"}
    trace = json.loads(gzip.open(files[-1]).read())
    events = trace.get("traceEvents", [])
    # GPU kernels: complete events on a stream thread of a GPU process
    pids = {e["pid"]: e.get("args", {}).get("name", "") for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu_pids = {p for p, n in pids.items() if "gpu" in n.lower() or "/device:" in n.lower()}
    per = defaultdict(lambda: [0.0, 0])
    kernels = defaultdict(lambda: [0.0, 0])
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu_pids:
            continue
        name = e.get("name", "")
        if name.startswith(("Thunk", "XLA", "$")):
            continue
        b = bucket_of(name)
        per[b][0] += e.get("dur", 0.0)
        per[b][1] += 1
        kernels[name][0] += e.get("dur", 0.0)
        kernels[name][1] += 1
    total = sum(v[0] for v in per.values()) or 1.0
    # The trace file is named after the host; record only the folder.
    return {"trace_dir": str(out_dir.relative_to(ROOT)), "steps": steps,
            "gpu_ms_per_step": round(total / steps / 1e3, 3),
            "kernels_per_step": round(sum(v[1] for v in per.values()) / steps, 1),
            "buckets": {b: {"ms_per_step": round(v[0] / steps / 1e3, 3), "launches_per_step": round(v[1] / steps, 1),
                            "pct": round(100 * v[0] / total, 2)} for b, v in sorted(per.items())},
            "top_kernels": [{"name": n[:160], "bucket": bucket_of(n), "calls_per_step": c / steps,
                             "ms_per_step": round(t / steps / 1e3, 4)}
                            for n, (t, c) in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:25]]}


def parity(npz_path, attn_impl):
    """Run the JAX block on inputs and weights written by jax_port/compare.py (from the
    PyTorch block) and compare its output and gradients with PyTorch's."""
    import numpy as np
    z = np.load(npz_path)
    H = int(z["H"])
    p = {k: jnp.asarray(z[f"w_{k}"]) for k in ("norm1", "norm2", "qkv", "wo", "w1", "w3", "w2")}
    x = jnp.asarray(z["x"])
    dy = jnp.asarray(z["dy"])
    T, hd = x.shape[1], x.shape[2] // H
    cos, sin = rope_tables(T, hd)

    def f(p, x):
        return block(p, x, cos, sin, H, attn_impl)

    y, vjp = jax.vjp(jax.jit(f), p, x)
    gp, gx = vjp(dy)
    rel = lambda a, b: float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(b), 1e-30))
    rows = {"output": rel(y, z["y"]), "grad_x": rel(gx, z["g_x"])}
    for k in p:
        rows[f"grad_{k}"] = rel(gp[k], z[f"g_{k}"])
    return rows


def gemm_times(spec):
    """spec: JSON list of [M, N, K, a_fp32]. Times C = A @ B in bf16 under jit (A converted
    from fp32 inside the same jit when a_fp32, as the model's weight casts are) and returns
    microseconds per call plus the kernel XLA picked."""
    out = []
    for M, N, K, a32 in json.loads(spec):
        ka, kb = jax.random.split(jax.random.PRNGKey(0))
        A = jax.random.normal(ka, (M, K), jnp.float32 if a32 else BF16)
        Bm = jax.random.normal(kb, (K, N), BF16)
        f = jax.jit(lambda A, Bm: A.astype(BF16) @ Bm)
        c = f.lower(A, Bm).compile()
        kind = "cublas" if "__cublas$gemm" in c.as_text() else ("triton gemm fusion" if "__triton" in c.as_text()
                                                               or "gemm_fusion" in c.as_text() else "other")
        # Queue 50 calls and sync once: a sync per call costs about 1 ms under WSL2,
        # which is longer than these GEMMs.
        for _ in range(10):
            c(A, Bm)
        jax.block_until_ready(c(A, Bm))
        best = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(50):
                r = c(A, Bm)
            r.block_until_ready()
            best = min(best, (time.perf_counter() - t0) / 50)
        sec = best
        out.append({"M": M, "N": N, "K": K, "a_fp32": a32, "us": round(sec * 1e6, 2), "kernel": kind,
                    "tflops": round(2 * M * N * K / sec / 1e12, 1)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parity", default=None, help="npz from compare.py: check the JAX block against PyTorch")
    p.add_argument("--gemm", default=None, help="JSON list of [M, N, K, a_fp32]: time GEMMs under XLA")
    p.add_argument("--size", default="medium", choices=list(SIZES))
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--attn", default="cudnn", choices=["cudnn", "xla", "naive"],
                   help="cudnn: fused flash style kernel; xla: XLA's own lowering of the same op")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--machine-label", default="final")
    args = p.parse_args()

    if args.parity:
        print(json.dumps(parity(args.parity, args.attn)))
        return
    if args.gemm:
        print(json.dumps(gemm_times(args.gemm)))
        return

    dev = jax.devices()[0]
    arch = f"sm_{dev.compute_capability.replace('.', '')}"
    cfg = SIZES[args.size]
    B, T = args.batch, args.seq
    tag = f"{args.size}_b{B}_s{T}_{args.attn}"
    print(f"jax {jax.__version__} on {arch} ({args.machine_label}) | {tag}")

    peak = bf16_peak_tflops()
    print(f"bf16 matmul peak reached this run: {peak:.1f} TFLOPs")

    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    opt = adamw_init(params)
    k1, _ = jax.random.split(jax.random.PRNGKey(1))
    data = jax.random.randint(k1, (B, T + 1), 0, VOCAB)
    tokens, targets = data[:, :-1], data[:, 1:]
    step = make_train_step(cfg, args.attn)
    initial_loss = float(jax.jit(lambda p: loss_fn(p, tokens, targets, cfg, args.attn))(params))

    t0 = time.perf_counter()
    lowered = step.lower(params, opt, tokens, targets)
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    hlo = hlo_stats(compiled.as_text())

    state = {"params": params, "opt": opt, "loss": None}

    def run_step():
        state["params"], state["opt"], state["loss"] = compiled(state["params"], state["opt"], tokens, targets)
        return state["loss"]

    sec = time_fn(run_step, args.warmup, args.steps)
    loss = float(state["loss"])
    tokens_per_step = B * T
    tflops = model_flops_per_token(cfg, T) * tokens_per_step / sec / 1e12
    mem = dev.memory_stats() or {}
    model_row = {"ms_per_step": round(sec * 1e3, 2), "tokens_per_s": round(tokens_per_step / sec), "tflops": round(tflops, 2),
                 "mfu": round(tflops / peak, 4), "initial_loss": round(initial_loss, 4), "final_loss": round(loss, 4),
                 "steps_trained": args.warmup + args.steps + 3, "loss_finite": math.isfinite(loss),
                 "compile_s": round(compile_s, 1), "peak_mem_gb": round(mem.get("peak_bytes_in_use", 0) / 1e9, 2)}
    print(f"model step: {model_row['ms_per_step']} ms, {model_row['tokens_per_s']:,} tok/s, MFU {100 * model_row['mfu']:.1f}%, "
          f"loss {loss:.4f}, compile {compile_s:.1f} s, {hlo['fusions']} fusions, custom calls {hlo['custom_calls']}")

    prof = profile_step(run_step, f"{args.machine_label}_{tag}")
    if "buckets" in prof:
        print(f"profile: {prof['kernels_per_step']} kernels per step, {prof['gpu_ms_per_step']} ms GPU per step")
        for b, v in prof["buckets"].items():
            print(f"  {b:<18} {v['ms_per_step']:8.3f} ms  {v['launches_per_step']:6.0f} launches  {v['pct']:5.1f}%")
    else:
        print(f"profile: {prof}")

    # One transformer block, forward plus backward. (The step donates its inputs, so the
    # original params are gone; use the current ones.)
    bp = state["params"]["blocks"][0]
    x = jax.random.normal(jax.random.PRNGKey(2), (B, T, cfg["d"]), jnp.float32)
    cos, sin = rope_tables(T, cfg["d"] // cfg["H"])
    dy = jax.random.normal(jax.random.PRNGKey(3), (B, T, cfg["d"]), jnp.float32)

    @jax.jit
    def block_fb(bp, x):
        y, vjp = jax.vjp(lambda bp, x: block(bp, x, cos, sin, cfg["H"], args.attn), bp, x)
        return vjp(dy)

    block_c = block_fb.lower(bp, x).compile()
    block_s = time_fn(lambda: block_c(bp, x), args.warmup, args.steps)
    block_hlo = hlo_stats(block_c.as_text())
    print(f"block fwd+bwd: {block_s * 1e3:.3f} ms, {block_hlo['fusions']} fusions, custom calls {block_hlo['custom_calls']}")

    del block_c, x, dy
    opt_t = time_optimizer(state, args.warmup, args.steps)
    print(f"AdamW alone: {opt_t['ms']} ms for {opt_t['params'] / 1e6:.0f}M params in {opt_t['fusions']} fusions "
          f"({opt_t['gbs']} GB/s at 28 bytes per param)")
    print(f"GEMM fusion calls: {hlo['gemm_fusion_calls']} ({hlo['gemm_fusion_computations']} distinct), merging two "
          f"weight matrices into one GEMM: {hlo['gemm_fusions_merging_two_weights']}, with a convert inside: "
          f"{hlo['gemm_fusions_with_convert_inside']}, standalone convert kernels: {hlo['standalone_converts']}")

    out = {"framework": "jax", "jax": jax.__version__, "arch": arch, "machine_label": args.machine_label,
           "os": platform.platform(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "size": args.size, "batch": B,
           "seq": T, "attn": args.attn, "bf16_ref_tflops": round(peak, 1), "model": model_row, "model_hlo": hlo,
           "profile": prof, "block": {"ms": round(block_s * 1e3, 4), "hlo": block_hlo}, "optimizer": opt_t}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"jax_{args.machine_label}_{arch}_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
