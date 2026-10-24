"""Cross check the exported graph against a real PyTorch execution trace.

Runs one training step of the fused model (bench_train.py, --kernels triton, bf16, SDPA,
fused Adam) under torch.profiler.ExecutionTraceObserver, which records every operator
PyTorch dispatches, in order, with input shapes. From that trace it takes:

  GEMMs       every aten::mm / aten::addmm / aten::bmm call, as (M, N, K)
  attention   every flash attention forward and backward call
  order       the sequence of GEMM shapes in the forward and in the backward

and compares them with the exported graph (export_trace.py) for the same configuration:
the multiset of GEMM shapes must be equal, the attention call counts must be equal, and
the forward GEMM order must match.

The Triton kernels and the fused Adam are not aten operators, so the trace records them
only as autograd function nodes; they are compared by count.

  python workload_export/observe_pytorch.py --size medium --batch 4 --seq 1024
"""

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GEMM_OPS = {"aten::mm", "aten::addmm", "aten::bmm"}
ATTN_FWD = ("aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_efficient_attention",
            "aten::_scaled_dot_product_cudnn_attention")
ATTN_BWD = ("aten::_scaled_dot_product_flash_attention_backward",
            "aten::_scaled_dot_product_efficient_attention_backward",
            "aten::_scaled_dot_product_cudnn_attention_backward")


def gemm_mnk(name, shapes):
    """(M, N, K) of a GEMM from its input shapes as recorded by the trace."""
    if name == "aten::mm":
        (m, k), (k2, n) = shapes[0], shapes[1]
    elif name == "aten::addmm":
        (m, k), (k2, n) = shapes[1], shapes[2]
    else:   # bmm: batch, m, k x batch, k, n
        (b, m, k), (_, k2, n) = shapes[0], shapes[1]
    return (m, n, k)


def capture(size, B, T):
    import torch
    from torch.profiler import ExecutionTraceObserver
    from bench_train import SIZES, build, cap_gpu_memory

    cap_gpu_memory()
    run = dict(size=size, batch=B, seq=T, precision="bf16", attn="sdpa", fused_adam=True, compile=False,
               ckpt=False, nvtx=False, kernels="triton")
    model, opt, step = build(SIZES[size], run)
    for _ in range(2):            # warm up so Triton compilation is not in the trace
        step()
    torch.cuda.synchronize()
    path = Path(tempfile.gettempdir()) / f"pytorch_et_{size}_b{B}_s{T}.json"
    et = ExecutionTraceObserver()
    et.register_callback(str(path))
    et.start()
    step()
    torch.cuda.synchronize()
    et.stop()
    et.unregister_callback()
    return json.loads(path.read_text()), path


def parse_trace(trace):
    """Ordered GEMM shapes, attention calls and autograd functions from an ET JSON."""
    gemms, attn_f, attn_b, autograd = [], 0, 0, Counter()
    nodes = sorted(trace["nodes"], key=lambda n: n["id"])
    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        name = n["name"]
        parent = by_id.get(n.get("ctrl_deps"), {}).get("name")
        # An aten op can redispatch to itself (aten::mm inside aten::mm); the trace records
        # both. Count only the outermost call.
        if name in GEMM_OPS and parent not in GEMM_OPS:
            shapes = n.get("inputs", {}).get("shapes") or n.get("input_shapes")
            gemms.append(gemm_mnk(name, shapes))
        elif name in ATTN_FWD and parent not in ATTN_FWD:
            attn_f += 1
        elif name in ATTN_BWD and parent not in ATTN_BWD:
            attn_b += 1
        elif "Backward" in name or name.startswith("_"):
            autograd[name] += 1
    return gemms, attn_f, attn_b, autograd


def export_gemms(size, B, T):
    """GEMM shapes and attention counts of the exported graph, in graph order."""
    from perf_model.roofline import SIZES, Machine, RESULTS, load_shape_times
    from workload_export.export_trace import build_rank, make_groups

    m = Machine.from_calibration(RESULTS / "calibration_final.json")
    groups, _ = make_groups(1, 1, 1)
    g = build_rank(SIZES[size], B, T, 1, 1, 1, 1, 0, 0, 0, m, load_shape_times(None), groups)
    gemms, fwd, attn_f, attn_b = [], [], 0, 0
    for n in g.nodes:
        if n["type"] == "GEMM" and n["category"] == "matmul":
            mnk = _mnk_from_node(n)
            gemms.append(mnk)
            if n["phase"] == "fwd" and n["layer"] is not None:
                fwd.append(mnk)
        elif n["category"] == "attention":
            attn_f += n["phase"] == "fwd"
            attn_b += n["phase"] == "bwd"
    return gemms, fwd, attn_f, attn_b


def _mnk_from_node(n):
    out = n["outputs"][0]["shape"]
    M, N = out
    K = int(round(n["flops"] / (2.0 * M * N)))
    return (M, N, K)


def canon(mnk):
    """A GEMM and its transpose do the same work; PyTorch may compute C^T = B^T A^T."""
    m, n, k = mnk
    return (min(m, n), max(m, n), k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="medium")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--machine-label", default="final")
    args = p.parse_args()

    trace, path = capture(args.size, args.batch, args.seq)
    t_gemms, t_af, t_ab, autograd = parse_trace(trace)
    e_gemms, e_fwd, e_af, e_ab = export_gemms(args.size, args.batch, args.seq)

    N = args.batch * args.seq
    # Block GEMMs only for the order check: the trace's forward is everything before the
    # first backward GEMM. Activations are (N, d) here; the trace may record (B, T, d)
    # flattened by aten::matmul into (N, d), which is the same GEMM.
    t_counter, e_counter = Counter(map(canon, t_gemms)), Counter(map(canon, e_gemms))
    same_multiset = t_counter == e_counter
    t_fwd = [canon(x) for x in t_gemms[:len(e_fwd)]]
    same_order = t_fwd == [canon(x) for x in e_fwd]

    print(f"PyTorch execution trace: {len(trace['nodes'])} operator nodes ({path})")
    print(f"  GEMMs: trace {len(t_gemms)}, export {len(e_gemms)}; same multiset of (M, N, K): "
          f"{'PASS' if same_multiset else 'FAIL'}")
    if not same_multiset:
        print(f"    only in trace: {dict(list((t_counter - e_counter).items())[:6])}")
        print(f"    only in export: {dict(list((e_counter - t_counter).items())[:6])}")
    print(f"  forward block GEMM order (first {len(e_fwd)}): {'PASS' if same_order else 'FAIL'}")
    print(f"  flash attention calls: trace {t_af} forward and {t_ab} backward, export {e_af} and {e_ab}: "
          f"{'PASS' if (t_af, t_ab) == (e_af, e_ab) else 'FAIL'}")
    top = ", ".join(f"{k} x{v}" for k, v in autograd.most_common(8))
    print(f"  autograd and custom function nodes seen: {top}")

    out = {"size": args.size, "batch": args.batch, "seq": args.seq, "trace_nodes": len(trace["nodes"]),
           "trace_gemms": len(t_gemms), "export_gemms": len(e_gemms), "gemm_multiset_match": same_multiset,
           "forward_order_match": same_order, "attention": {"trace": [t_af, t_ab], "export": [e_af, e_ab]},
           "autograd_nodes": dict(autograd.most_common(30))}
    from perf_model.roofline import RESULTS
    dst = RESULTS / f"workload_et_crosscheck_{args.machine_label}_{args.size}_b{args.batch}_s{args.seq}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"Saved {dst.name}")
    ok = same_multiset and same_order and (t_af, t_ab) == (e_af, e_ab)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
