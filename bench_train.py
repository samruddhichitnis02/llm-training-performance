"""Training step benchmark for a Llama style decoder.

Runs real training steps (forward, backward, AdamW update) on one fixed batch of
random tokens and reports ms per step, tokens per second, achieved TFLOPs, MFU,
peak memory and the final loss. Step speed does not depend on the data, so no
dataset is needed.

Examples
  python bench_train.py --calibrate-only --machine-label final
  python bench_train.py --size small --batch 8 --seq 1024 --precision bf16 --attn sdpa
  python bench_train.py --sweep --size small --batch 8 --seq 1024 --machine-label final
"""

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VOCAB = 32000


# --------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    d: int
    L: int
    H: int
    ffn: int
    vocab: int = VOCAB
    eps: float = 1e-5
    rope_base: float = 10000.0

    @property
    def hd(self):
        return self.d // self.H


SIZES = {
    "tiny": ModelConfig(d=512, L=8, H=8, ffn=1408),
    "small": ModelConfig(d=768, L=12, H=12, ffn=2048),
    "medium": ModelConfig(d=1024, L=24, H=16, ffn=2816),
    "large": ModelConfig(d=2048, L=16, H=16, ffn=5504),
}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def act_dtype():
    """Dtype the next matmul will run in: the autocast dtype if autocast is on."""
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return None


def rope_tables(T, hd, base, device):
    """cos and sin tables of shape (T, hd), float32, halves duplicated (rotate half layout)."""
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)                 # (T, hd/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (T, hd)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd). Math in float32, result in the input dtype."""
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        dt = act_dtype()
        return y.to(dt) if dt is not None else y.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg, attn_impl):
        super().__init__()
        self.cfg = cfg
        self.attn_impl = attn_impl
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)   # packed layout (3, H, hd)
        self.wo = nn.Linear(cfg.d, cfg.d, bias=False)
        self._mask = None

    def _causal_mask(self, T, device):
        if self._mask is None or self._mask.shape[0] != T or self._mask.device != device:
            self._mask = torch.ones(T, T, dtype=torch.bool, device=device).triu(1)
        return self._mask

    def forward(self, x, cos, sin, ops=None):
        B, T, C = x.shape
        H, hd = self.cfg.H, self.cfg.hd
        qkv = self.qkv(x)                                   # (B, T, 3*H*hd)
        if ops is not None:
            q, k, v = ops.rope_qkv(qkv, cos, sin, H)        # each (B, H, T, hd), contiguous
        else:
            q, k, v = qkv.view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Naive attention: materializes the full T by T score matrix.
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
            scores = scores.masked_fill(self._causal_mask(T, x.device), float("-inf"))
            probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
            y = probs @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d, cfg.ffn, bias=False)
        self.w3 = nn.Linear(cfg.d, cfg.ffn, bias=False)
        self.w2 = nn.Linear(cfg.ffn, cfg.d, bias=False)

    def forward(self, x, ops=None):
        if ops is not None:
            return self.w2(ops.swiglu(self.w1(x), self.w3(x)))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg, attn_impl, ops):
        super().__init__()
        self.ops = ops
        self.norm1 = RMSNorm(cfg.d, cfg.eps)
        self.attn = Attention(cfg, attn_impl)
        self.norm2 = RMSNorm(cfg.d, cfg.eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        ops = self.ops
        if ops is None:
            x = x + self.attn(self.norm1(x), cos, sin)
            return x + self.mlp(self.norm2(x))
        # Fused path. The residual stream x stays in its own dtype (float32 under autocast).
        n1 = ops.rmsnorm(x, self.norm1.weight, self.norm1.eps, out_dtype=act_dtype())
        a = self.attn(n1, cos, sin, ops)
        n2, s = ops.add_rmsnorm(a, x, self.norm2.weight, self.norm2.eps, out_dtype=act_dtype())
        return s + self.mlp(n2, ops)


class LlamaModel(nn.Module):
    def __init__(self, cfg, max_seq, attn_impl="sdpa", ops=None, ckpt=False, nvtx=False):
        super().__init__()
        self.cfg = cfg
        self.ops = ops
        self.ckpt = ckpt
        self.nvtx = nvtx
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList([Block(cfg, attn_impl, ops) for _ in range(cfg.L)])
        self.norm_f = RMSNorm(cfg.d, cfg.eps)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)   # untied from emb
        cos, sin = rope_tables(max_seq, cfg.hd, cfg.rope_base, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets):
        B, T = idx.shape
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.emb(idx)
        for i, blk in enumerate(self.blocks):
            if self.nvtx:
                torch.cuda.nvtx.range_push(f"block{i}")
            if self.ckpt and self.training:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
            if self.nvtx:
                torch.cuda.nvtx.range_pop()

        if self.ops is None:
            logits = self.head(self.norm_f(x))
            return F.cross_entropy(logits.float().view(-1, self.cfg.vocab), targets.reshape(-1))
        h = self.ops.rmsnorm(x, self.norm_f.weight, self.norm_f.eps, out_dtype=act_dtype())
        return self.ops.fused_linear_cross_entropy(h.view(B * T, -1), self.head.weight, targets.reshape(-1))


def load_ops(kernels):
    """Import the fused kernels lazily so the torch baseline works without them."""
    if kernels == "torch":
        return None
    from kernels import triton_ops as t
    ops = SimpleNamespace(
        rmsnorm=t.rmsnorm,
        add_rmsnorm=t.add_rmsnorm,
        swiglu=t.swiglu,
        rope_qkv=t.rope_qkv,
        fused_linear_cross_entropy=t.fused_linear_cross_entropy,
    )
    if kernels == "cuda":
        from kernels.cuda_ext.add_rmsnorm_ext import add_rmsnorm as cuda_add_rmsnorm
        ops.add_rmsnorm = cuda_add_rmsnorm
    elif kernels != "triton":
        raise ValueError(f"unknown kernels mode {kernels}")
    return ops


def count_params(cfg):
    """Total parameters and matmul parameters (everything except the embedding table)."""
    per_block = 4 * cfg.d * cfg.d + 3 * cfg.d * cfg.ffn + 2 * cfg.d
    total = cfg.L * per_block + cfg.d + 2 * cfg.vocab * cfg.d
    return total, total - cfg.vocab * cfg.d


def model_flops_per_token(cfg, seq):
    """6 x matmul params + 12 x L x d x seq. Checkpoint recompute is not counted."""
    _, n_matmul = count_params(cfg)
    return 6 * n_matmul + 12 * cfg.L * cfg.d * seq


# --------------------------------------------------------------------------
# Environment and calibration
# --------------------------------------------------------------------------

def arch_tag():
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def env_info(label):
    """Machine description. The GPU model name is never recorded, only the architecture tag."""
    props = torch.cuda.get_device_properties(0)
    return {
        "arch": arch_tag(),
        "machine_label": label,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu_count": torch.cuda.device_count(),
        "sm_count": props.multi_processor_count,
        "gpu_mem_gb": round(props.total_memory / 1e9, 1),
    }


def _time_gpu(fn, warmup=3, iters=20, repeats=3):
    """Best average seconds per call over a few repeats, timed with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1000.0 / iters)
    return best


def measure_matmul_tflops(precision, n=8192):
    """Peak dense matmul TFLOPs for one precision on n by n by n matrices."""
    dev = "cuda"
    flops = 2.0 * n * n * n
    cc = torch.cuda.get_device_capability()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if precision in ("fp32", "tf32"):
            torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
            a = torch.randn(n, n, device=dev)
            b = torch.randn(n, n, device=dev)
            fn = lambda: a @ b
        elif precision in ("fp16", "bf16"):
            dt = torch.float16 if precision == "fp16" else torch.bfloat16
            a = torch.randn(n, n, device=dev, dtype=dt)
            b = torch.randn(n, n, device=dev, dtype=dt)
            fn = lambda: a @ b
        elif precision == "fp8":
            if cc < (8, 9):
                return {"status": "unsupported", "reason": "fp8 tensor cores need compute capability 8.9+"}
            a = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
            b = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn).t()   # column major
            one = torch.tensor(1.0, device=dev)
            fn = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        elif precision == "fp4":
            # NVFP4: e2m1 values packed two per byte, one e4m3 scale per 16 values.
            if cc < (10, 0) or not hasattr(torch, "float4_e2m1fn_x2"):
                return {"status": "unsupported", "reason": "fp4 tensor cores need Blackwell (compute capability 10.0+)"}
            a = torch.randint(0, 256, (n, n // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            b = torch.randint(0, 256, (n, n // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2).t()
            sa = torch.ones(n * (n // 16), device=dev).to(torch.float8_e4m3fn)
            sb = torch.ones(n * (n // 16), device=dev).to(torch.float8_e4m3fn)
            fn = lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        else:
            raise ValueError(precision)
        sec = _time_gpu(fn)
        out = {"status": "ok", "tflops": round(flops / sec / 1e12, 1), "ms": round(sec * 1e3, 3), "n": n}
        if precision == "fp4":
            out["format"] = "nvfp4 (e2m1 values, e4m3 scale per 16)"
        if precision == "fp8":
            out["format"] = "e4m3 x e4m3, tensorwise scale, bf16 out"
        return out
    except Exception as e:  # keep calibration going if one precision fails
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.cuda.empty_cache()


def measure_bandwidth_gbs():
    """Device to device copy bandwidth (read plus write bytes per second)."""
    free, _ = torch.cuda.mem_get_info()
    nbytes = int(min(1 << 30, free // 8))
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    sec = _time_gpu(lambda: dst.copy_(src), warmup=3, iters=20)
    del src, dst
    torch.cuda.empty_cache()
    return {"status": "ok", "gbs": round(2 * nbytes / sec / 1e9, 1), "copy_bytes": nbytes}


def measure_launch_overhead_us():
    """Average time per tiny kernel when many are queued back to back."""
    x = torch.zeros(1, device="cuda")
    iters = 2000
    sec = _time_gpu(lambda: x.add_(1.0), warmup=50, iters=iters)
    return round(sec * 1e6, 2)


def calibrate(label):
    info = env_info(label)
    print(f"Calibrating on {info['arch']} ({label})")
    matmul = {}
    for p in ["fp32", "tf32", "fp16", "bf16", "fp8", "fp4"]:
        matmul[p] = measure_matmul_tflops(p)
        r = matmul[p]
        print(f"  matmul {p:>5}: " + (f"{r['tflops']:8.1f} TFLOPs" if r["status"] == "ok" else r["status"] + " " + r.get("reason", r.get("error", ""))))
    bw = measure_bandwidth_gbs()
    print(f"  memory bandwidth: {bw['gbs']:.1f} GB/s")
    launch = measure_launch_overhead_us()
    print(f"  launch overhead: {launch:.2f} us per kernel")
    out = {**info, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "matmul": matmul,
           "bandwidth_gbs": bw["gbs"], "launch_overhead_us": launch}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"calibration_{label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path.relative_to(ROOT)}")
    return out


# --------------------------------------------------------------------------
# Training step benchmark
# --------------------------------------------------------------------------

def set_precision(precision):
    # TF32 is off everywhere so the fp32 baseline is true fp32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp32":
        return torch.autocast("cuda", enabled=False)
    raise ValueError(precision)


def make_batch(B, T, vocab, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    data = torch.randint(0, vocab, (B, T + 1), generator=g).to(device)
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


def build(cfg, run, device="cuda"):
    """Model, optimizer and a step function for one configuration."""
    ops = load_ops(run["kernels"])
    torch.manual_seed(run.get("seed", 0))   # same initial weights for every row, so losses compare
    model = LlamaModel(cfg, run["seq"], run["attn"], ops, run["ckpt"], run["nvtx"] and not run["compile"]).to(device)
    model.train()
    # Non-fused rows use the multi tensor (foreach) Adam, set explicitly. With only fused=False this
    # PyTorch build ran the per parameter loop instead (checked with the profiler), a much slower baseline.
    adam_impl = dict(fused=True) if run["fused_adam"] else dict(foreach=True)
    opt = torch.optim.AdamW(model.parameters(), lr=run.get("lr", 3e-4), betas=(0.9, 0.95),
                            weight_decay=0.1, **adam_impl)
    fwd = torch.compile(model) if run["compile"] else model
    idx, tgt = make_batch(run["batch"], run["seq"], cfg.vocab, device)
    nvtx = run["nvtx"]

    def step():
        if nvtx:
            torch.cuda.nvtx.range_push("forward")
        with set_precision(run["precision"]):
            loss = fwd(idx, tgt)
        if nvtx:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("backward")
        loss.backward()
        if nvtx:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("optimizer")
        opt.step()
        opt.zero_grad(set_to_none=True)
        if nvtx:
            torch.cuda.nvtx.range_pop()
        return loss

    return model, opt, step


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch._dynamo.reset()
    except Exception:
        pass


def run_one(cfg, run, warmup, steps, bf16_ref, peak_spec):
    """Benchmark one configuration. Never raises: oom and errors are recorded in the row."""
    row = {"name": run.get("name", "run"), "config": {k: v for k, v in run.items() if k != "name"}}
    model = opt = step = None
    try:
        cleanup()
        torch.cuda.reset_peak_memory_stats()
        model, opt, step = build(cfg, run)
        loss = None
        for i in range(warmup):
            if run["nvtx"]:
                torch.cuda.nvtx.range_push(f"warmup{i}")
            loss = step()
            if run["nvtx"]:
                torch.cuda.nvtx.range_pop()
        times = []
        for i in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if run["nvtx"]:
                torch.cuda.nvtx.range_push(f"step{i}")
            loss = step()
            if run["nvtx"]:
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        final_loss = float(loss.item())
        sec = statistics.median(times)
        tokens = run["batch"] * run["seq"]
        tflops = model_flops_per_token(cfg, run["seq"]) * tokens / sec / 1e12
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        row.update({
            "status": "ok",
            "ms_per_step": round(sec * 1e3, 2),
            "ms_min": round(min(times) * 1e3, 2),
            "ms_max": round(max(times) * 1e3, 2),
            "tokens_per_s": round(tokens / sec),
            "tflops": round(tflops, 2),
            "mfu": round(tflops / bf16_ref, 4),
            "mfu_spec": round(tflops / peak_spec, 4) if peak_spec else None,
            "peak_mem_gb": round(peak_alloc, 2),
            "peak_reserved_gb": round(peak_reserved, 2),
            # Past the card's memory the driver can spill to system RAM and timings become meaningless.
            "spill_suspected": peak_reserved > 0.98 * total_mem,
            "final_loss": round(final_loss, 4),
            "loss_finite": math.isfinite(final_loss),
        })
    except torch.cuda.OutOfMemoryError:
        row.update({"status": "oom"})
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".strip().splitlines()
        row.update({"status": "error", "error": " ".join(msg)[:400]})
    finally:
        del model, opt, step
        cleanup()
    return row


LADDER = [
    ("1 fp32 baseline, naive attn", dict(precision="fp32", attn="naive", fused_adam=False, compile=False, kernels="torch")),
    ("2 bf16 autocast", dict(precision="bf16", attn="naive", fused_adam=False, compile=False, kernels="torch")),
    ("3 + fused attention (SDPA)", dict(precision="bf16", attn="sdpa", fused_adam=False, compile=False, kernels="torch")),
    ("4 + fused Adam", dict(precision="bf16", attn="sdpa", fused_adam=True, compile=False, kernels="torch")),
    ("5 + torch.compile", dict(precision="bf16", attn="sdpa", fused_adam=True, compile=True, kernels="torch")),
    ("6 fused Triton kernels", dict(precision="bf16", attn="sdpa", fused_adam=True, compile=False, kernels="triton")),
    ("7 CUDA C++ add+RMSNorm", dict(precision="bf16", attn="sdpa", fused_adam=True, compile=False, kernels="cuda")),
]


def print_rows(rows):
    from tabulate import tabulate
    table = []
    for r in rows:
        if r["status"] == "ok":
            table.append([r["name"], r["ms_per_step"], f"{r['tokens_per_s']:,}", r.get("speedup", ""),
                          r["tflops"], f"{100 * r['mfu']:.1f}%", r["peak_mem_gb"], r["final_loss"],
                          "yes" if r["loss_finite"] else "NO"])
        else:
            table.append([r["name"], r["status"], r.get("error", "")[:60], "", "", "", "", "", ""])
    print(tabulate(table, headers=["row", "ms/step", "tok/s", "speedup", "TFLOPs", "MFU", "peak GB", "loss", "finite"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="small", choices=list(SIZES))
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--attn", default="sdpa", choices=["naive", "sdpa"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--ckpt", action="store_true", help="activation checkpointing per block")
    p.add_argument("--kernels", default="torch", choices=["torch", "triton", "cuda"])
    p.add_argument("--nvtx", action="store_true", help="named NVTX ranges for Nsight Systems")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--machine-label", default="dev")
    p.add_argument("--peak-tflops", type=float, default=None, help="spec sheet bf16 dense peak, optional")
    p.add_argument("--sweep", action="store_true", help="run the full optimization ladder")
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--out", default=None, help="output JSON path (default under results/)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available. Install the +cu128 PyTorch build.")

    if args.calibrate_only:
        calibrate(args.machine_label)
        return

    cfg = SIZES[args.size]
    info = env_info(args.machine_label)
    total, n_matmul = count_params(cfg)
    print(f"{info['arch']} ({args.machine_label}) | {args.size}: {total / 1e6:.0f}M params, "
          f"batch {args.batch}, seq {args.seq}")

    bref = measure_matmul_tflops("bf16")
    if bref["status"] != "ok":
        sys.exit(f"bf16 reference matmul failed: {bref}")
    bf16_ref = bref["tflops"]
    print(f"bf16 matmul peak reached this run (MFU reference): {bf16_ref:.1f} TFLOPs")

    base = dict(size=args.size, batch=args.batch, seq=args.seq, ckpt=args.ckpt, nvtx=args.nvtx, lr=args.lr)
    if args.sweep:
        runs = [dict(name=name, **base, **cfg_row) for name, cfg_row in LADDER]
    else:
        name = (f"{args.precision} {args.attn}" + (" fusedadam" if args.fused_adam else "")
                + (" compile" if args.compile else "") + (" ckpt" if args.ckpt else "")
                + (f" kernels={args.kernels}" if args.kernels != "torch" else ""))
        runs = [dict(name=name, **base, precision=args.precision, attn=args.attn,
                     fused_adam=args.fused_adam, compile=args.compile, kernels=args.kernels)]

    rows = []
    for run in runs:
        print(f"running: {run['name']} ...", flush=True)
        row = run_one(cfg, run, args.warmup, args.steps, bf16_ref, args.peak_tflops)
        rows.append(row)
        if row["status"] == "ok":
            print(f"  {row['ms_per_step']} ms/step, {row['tokens_per_s']:,} tok/s, MFU {100 * row['mfu']:.1f}%, "
                  f"{row['peak_mem_gb']} GB, loss {row['final_loss']}", flush=True)
            if row["spill_suspected"]:
                print("  WARNING: memory reserved is at the card limit, timings may include spill to system RAM")
        else:
            print(f"  {row['status']} {row.get('error', '')}", flush=True)

    first = rows[0]
    for r in rows:
        if r["status"] == "ok" and first["status"] == "ok":
            r["speedup"] = round(first["ms_per_step"] / r["ms_per_step"], 2)

    print()
    print_rows(rows)

    out = {
        **info,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "size": args.size,
        "model": asdict(cfg),
        "params_total": total,
        "params_matmul": n_matmul,
        "flops_per_token": model_flops_per_token(cfg, args.seq),
        "batch": args.batch,
        "seq": args.seq,
        "warmup": args.warmup,
        "steps": args.steps,
        "bf16_ref_tflops": bf16_ref,
        "peak_tflops_spec": args.peak_tflops,
        "sweep": args.sweep,
        "rows": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    kind = "sweep" if args.sweep else "run"
    path = Path(args.out) if args.out else RESULTS / f"bench_{kind}_{args.machine_label}_{info['arch']}_{args.size}_b{args.batch}_s{args.seq}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
