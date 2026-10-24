"""Export one training step as an operator level workload for architecture simulators.

Builds the graph of every GPU operation in one step of the fused model (the Triton
kernel path of bench_train.py, bf16 autocast, SDPA, fused Adam) for a chosen data,
tensor and pipeline parallel layout, one graph per rank. Every node carries:

  name, type (GEMM, elementwise, reduction, collective, memory copy, send, recv),
  input and output tensors (shape, dtype), FLOPs, bytes read and written, data and
  control dependencies, the phase (forward, backward, optimizer), its layer and
  microbatch, and a predicted duration from the validated performance model.

Collectives for the layout:
  tensor parallel   all reduce of the (mb, T, d) activation after the attention output
                    and the MLP down projection, and of the input gradient after the qkv
                    and the gate/up projections in the backward (Megatron style)
  pipeline          send and receive of activations and their gradients between stages,
                    1F1B order (warmup forwards, then one forward one backward, then cooldown)
  data parallel     all reduce of each layer's fp32 gradients as soon as its backward ends,
                    so it can overlap with the rest of the backward; the optimizer waits for all

Outputs (one directory per configuration):
  graph.rank<r>.json          the graph in plain JSON
  workload.<r>.et             the same graph in Chakra execution trace format (ASTRA-sim
                              reads workload.<r>.et for NPU r when given the prefix "workload")
  comm_group.json             communicator groups (tensor, pipeline and data parallel)
  summary.json                totals, and the cross check against perf_model/roofline.py

The cross check: summed FLOPs and bytes of the exported compute nodes, per category,
must equal the operation inventory of the performance model for the same configuration.
The graph is built here from tensor shapes, independently of that inventory, so a
mismatch points to a bug in one of the two.

NVIDIA's own processor and system simulators are proprietary and not available
outside NVIDIA. This export is an open format workload description of the same kind.

  python workload_export/export_trace.py --size medium --batch 4 --seq 1024
  python workload_export/export_trace.py --size medium --batch 2 --seq 1024 --dp 2 --tp 2 --pp 2 --microbatches 4
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perf_model.roofline import (CE_CHUNK, RESULTS, SIZES, Machine, inventory,  # noqa: E402
                                 load_shape_times)
from workload_export import chakra_writer as cw  # noqa: E402

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "int64": 8}
CAT_OF_TYPE = {}   # filled per node: perf model category, for the cross check

TYPE_TO_CHAKRA = {"GEMM": cw.COMP_NODE, "elementwise": cw.COMP_NODE, "reduction": cw.COMP_NODE,
                  "memory copy": cw.COMP_NODE, "collective": cw.COMM_COLL_NODE,
                  "send": cw.COMM_SEND_NODE, "recv": cw.COMM_RECV_NODE}


def numel(shape):
    return math.prod(shape)


class Graph:
    """Operator graph for one rank. Data dependencies are found from tensor names: an op
    depends on the last op that wrote each of its inputs."""

    def __init__(self, rank, machine, shape_times):
        self.rank = rank
        self.nodes = []
        self.producer = {}
        self.m = machine
        self.st = shape_times or {}
        self.peak = machine.peak_tflops["bf16"] * 1e12
        self.bw = machine.bandwidth_gbs * 1e9
        self.last_compute = None      # for control dependencies: one in order compute stream

    def add(self, name, typ, cat, phase, inputs, outputs, flops=0.0, layer=None, mb=None, key=None,
            extra_deps=(), comm=None, extra_read=0.0, extra_write=0.0, serial=True):
        """inputs/outputs: lists of (tensor name, shape, dtype). extra_read/extra_write: bytes
        an op moves beyond its named tensors (for example a flash attention accumulator)."""
        nid = len(self.nodes)
        rd = sum(numel(s) * DTYPE_BYTES[d] for _, s, d in inputs) + extra_read
        wr = sum(numel(s) * DTYPE_BYTES[d] for _, s, d in outputs) + extra_write
        deps = sorted({self.producer[t] for t, _, _ in inputs if t in self.producer} | set(extra_deps))
        ctrl = []
        is_comp = typ in ("GEMM", "elementwise", "reduction", "memory copy")
        if is_comp and serial and self.last_compute is not None and self.last_compute not in deps:
            ctrl = [self.last_compute]
        if is_comp:
            if key is not None and key in self.st:
                t = self.st[key]
            else:
                t = max(flops / self.peak, (rd + wr) / self.bw)
            t += self.m.gpu_gap_us * 1e-6
        else:
            t = 0.0
        node = {"id": nid, "name": name, "type": typ, "category": cat, "phase": phase, "layer": layer,
                "microbatch": mb, "inputs": [{"name": t_, "shape": list(s), "dtype": d} for t_, s, d in inputs],
                "outputs": [{"name": t_, "shape": list(s), "dtype": d} for t_, s, d in outputs],
                "flops": flops, "bytes_read": rd, "bytes_written": wr, "data_deps": deps, "ctrl_deps": ctrl,
                "predicted_us": round(t * 1e6, 3)}
        if comm:
            node["comm"] = comm
        self.nodes.append(node)
        for t_, _, _ in outputs:
            self.producer[t_] = nid
        if is_comp and serial:
            self.last_compute = nid
        return nid


# --------------------------------------------------------------------------
# The training step, rank by rank
# --------------------------------------------------------------------------

def rank_of(dp_i, pp_i, tp_i, tp, pp):
    """Rank numbering: tensor parallel innermost, then pipeline, then data parallel."""
    return dp_i * (tp * pp) + pp_i * tp + tp_i


def schedule_1f1b(stage, p, m):
    warm = min(p - stage - 1, m)
    order = [("F", i) for i in range(warm)]
    for i in range(m - warm):
        order += [("F", warm + i), ("B", i)]
    order += [("B", i) for i in range(m - warm, m)]
    return order


def build_rank(s, B, T, dp, tp, pp, m, dp_i, pp_i, tp_i, machine, shape_times, groups):
    """Graph of one rank: stage pp_i of a model with s.L layers, m microbatches of B sequences."""
    g = Graph(rank_of(dp_i, pp_i, tp_i, tp, pp), machine, shape_times)
    a = "bf16"
    N = B * T
    d, L, V = s.d, s.L, s.vocab // tp
    H, hd, kv = s.H // tp, s.hd, max(1, s.kvh // tp)
    f = s.ffn // tp
    dq, dkv = H * hd, kv * hd
    dqkv = dq + 2 * dkv
    Lp = L // pp
    layers = range(pp_i * Lp, (pp_i + 1) * Lp)
    first, last = pp_i == 0, pp_i == pp - 1
    linears = [("qkv", dqkv, d), ("wo", d, dq), ("w1", f, d), ("w3", f, d), ("w2", d, f)]
    tp_group = groups["tp"][(dp_i, pp_i)]
    dp_group = groups["dp"][(pp_i, tp_i)]
    act = (N, d)

    def gemm(name, phase, l, mb, x, w, out, M, Nn, K, layout, out_dtype=a, out_extra=0.0):
        """x (M, K) or its transpose, w (Nn, K) or its transpose, out (M, Nn)."""
        return g.add(name, "GEMM", "matmul", phase, [x, w], [(out, (M, Nn), out_dtype)], flops=2.0 * M * Nn * K,
                     layer=l, mb=mb, key=("gemm", 1, M, Nn, K, layout, 2), extra_write=out_extra)

    def tp_allreduce(name, phase, l, mb, tensor):
        if tp == 1:
            return
        g.add(name, "collective", "comm", phase, [(tensor, act, a)], [(tensor, act, a)], layer=l, mb=mb,
              comm={"kind": "all_reduce", "group": "tp", "pg": tp_group, "bytes": N * d * 2})

    def tag(mb, direction, boundary):
        return (mb * 2 + direction) * pp + boundary

    peer = lambda stage: rank_of(dp_i, stage, tp_i, tp, pp)

    def forward(mb):
        p = f"m{mb}."
        x = p + "x"
        if first:
            g.add("embedding gather", "memory copy", "loss", "fwd", [(p + "tokens", (N,), "int64")], [],
                  layer=None, mb=mb, extra_read=N * d * 4 - N * 8, extra_write=0)
            g.nodes[-1]["outputs"] = [{"name": x, "shape": [N, d], "dtype": "fp32"}]
            g.nodes[-1]["bytes_written"] = N * d * 4
            g.producer[x] = g.nodes[-1]["id"]
        else:
            g.add(f"recv activation from stage {pp_i - 1}", "recv", "comm", "fwd", [], [(x, act, "fp32")], mb=mb,
                  comm={"kind": "recv", "src": peer(pp_i - 1), "dst": g.rank, "bytes": N * d * 2,
                        "tag": tag(mb, 0, pp_i - 1)})
        for l in layers:
            q = f"{p}l{l}."
            for nm, o, i in linears:
                g.add(f"L{l} autocast cast {nm} weight fp32 to bf16", "memory copy", "copy/cast", "fwd",
                      [(f"l{l}.{nm}.w", (o, i), "fp32")], [(q + f"{nm}.w16", (o, i), a)], layer=l, mb=mb)
            g.add(f"L{l} attn rmsnorm", "reduction", "elementwise", "fwd", [(x, act, "fp32")], [(q + "n1", act, a)],
                  layer=l, mb=mb)
            gemm(f"L{l} qkv projection", "fwd", l, mb, (q + "n1", act, a), (q + "qkv.w16", (dqkv, d), a), q + "qkv",
                 N, dqkv, d, "nt")
            g.add(f"L{l} rope_qkv", "elementwise", "elementwise", "fwd", [(q + "qkv", (N, dqkv), a)],
                  [(q + "q", (N, dq), a), (q + "k", (N, dkv), a), (q + "v", (N, dkv), a)], layer=l, mb=mb)
            g.add(f"L{l} flash attention forward", "GEMM", "attention", "fwd",
                  [(q + "q", (N, dq), a), (q + "k", (N, dkv), a), (q + "v", (N, dkv), a)],
                  [(q + "o", (N, dq), a), (q + "lse", (B, H, T), "fp32")], flops=2.0 * B * T * T * dq,
                  layer=l, mb=mb, key=("sdpa_fwd", B, H, kv, T, hd))
            g.add(f"L{l} attention output transpose", "memory copy", "copy/cast", "fwd", [(q + "o", (N, dq), a)],
                  [(q + "ot", (N, dq), a)], layer=l, mb=mb)
            gemm(f"L{l} output projection", "fwd", l, mb, (q + "ot", (N, dq), a), (q + "wo.w16", (d, dq), a),
                 q + "attn", N, d, dq, "nt")
            tp_allreduce(f"L{l} tp all reduce (attention output)", "fwd", l, mb, q + "attn")
            g.add(f"L{l} add_rmsnorm", "reduction", "elementwise", "fwd", [(q + "attn", act, a), (x, act, "fp32")],
                  [(q + "n2", act, a), (q + "s", act, "fp32")], layer=l, mb=mb)
            gemm(f"L{l} w1 (gate)", "fwd", l, mb, (q + "n2", act, a), (q + "w1.w16", (f, d), a), q + "g", N, f, d, "nt")
            gemm(f"L{l} w3 (up)", "fwd", l, mb, (q + "n2", act, a), (q + "w3.w16", (f, d), a), q + "u", N, f, d, "nt")
            g.add(f"L{l} swiglu", "elementwise", "elementwise", "fwd", [(q + "g", (N, f), a), (q + "u", (N, f), a)],
                  [(q + "h", (N, f), a)], layer=l, mb=mb)
            gemm(f"L{l} w2 (down)", "fwd", l, mb, (q + "h", (N, f), a), (q + "w2.w16", (d, f), a), q + "mlp",
                 N, d, f, "nt")
            tp_allreduce(f"L{l} tp all reduce (mlp output)", "fwd", l, mb, q + "mlp")
            nx = q + "out"
            g.add(f"L{l} residual add", "elementwise", "elementwise", "fwd", [(q + "s", act, "fp32"),
                  (q + "mlp", act, a)], [(nx, act, "fp32")], layer=l, mb=mb)
            x = nx
        if last:
            g.add("final rmsnorm", "reduction", "elementwise", "fwd", [(x, act, "fp32")], [(p + "hf", act, a)], mb=mb)
            g.add("lm head weight cast fp32 to bf16", "memory copy", "copy/cast", "fwd",
                  [("head.w", (V, d), "fp32")], [(p + "head.w16", (V, d), a)], mb=mb)
            chunks = math.ceil(N / CE_CHUNK)
            Nc = N // chunks
            for c in range(chunks):
                cp = f"{p}c{c}."
                gemm(f"lm head logits chunk {c}", "fwd", None, mb, (p + "hf", (Nc, d), a),
                     (p + "head.w16", (V, d), a), cp + "logits", Nc, V, d, "nt")
                g.add(f"cross entropy and logit gradient chunk {c}", "reduction", "loss", "fwd",
                      [(cp + "logits", (Nc, V), a)], [(cp + "dlogits", (Nc, V), a)], mb=mb)
                gemm(f"lm head input gradient chunk {c}", "fwd", None, mb, (cp + "dlogits", (Nc, V), a),
                     (p + "head.w16", (V, d), a), cp + "dh", Nc, d, V, "nn")
                # dW (V, d) += dlogits^T @ h in fp32: the accumulator is read and written
                g.add(f"lm head weight gradient chunk {c}", "GEMM", "matmul", "fwd",
                      [(cp + "dlogits", (Nc, V), a), (p + "hf", (Nc, d), a)], [(p + "head.dw", (V, d), "fp32")],
                      flops=2.0 * V * d * Nc, mb=mb, key=("gemm", 1, V, d, Nc, "tn", 2), extra_read=V * d * 4)
        return x

    def backward(mb):
        p = f"m{mb}."
        if last:
            g.add("scale saved hidden gradient", "elementwise", "loss", "bwd",
                  [(f"{p}c{c}.dh", (N // math.ceil(N / CE_CHUNK), d), a) for c in range(math.ceil(N / CE_CHUNK))],
                  [(p + "dhf", act, a)], mb=mb)
            g.add("lm head gradient scale and cast", "memory copy", "copy/cast", "bwd",
                  [(p + "head.dw", (V, d), "fp32")], [("head.grad", (V, d), "fp32")], mb=mb)
            g.add("final rmsnorm backward", "reduction", "elementwise", "bwd",
                  [(p + "dhf", act, a), (f"{p}l{layers[-1]}.out", act, "fp32")], [(p + "dx", act, "fp32")], mb=mb)
        else:
            g.add(f"recv gradient from stage {pp_i + 1}", "recv", "comm", "bwd", [], [(p + "dx", act, "fp32")], mb=mb,
                  comm={"kind": "recv", "src": peer(pp_i + 1), "dst": g.rank, "bytes": N * d * 2,
                        "tag": tag(mb, 1, pp_i)})
        dx = p + "dx"
        for l in reversed(layers):
            q = f"{p}l{l}."
            xin = f"{p}l{l - 1}.out" if l != layers[0] else (p + "x")
            # residual branch: gradient of the MLP output in bf16
            g.add(f"L{l} residual branch gradient cast fp32 to bf16", "memory copy", "copy/cast", "bwd",
                  [(dx, act, "fp32")], [(q + "dmlp", act, a)], layer=l, mb=mb)
            for nm, M_, K_, o, i, xin_t, w_t, dout in (("w2", N, f, d, f, q + "h", q + "w2.w16", q + "dmlp"),):
                gemm(f"L{l} w2 input gradient", "bwd", l, mb, (dout, (N, d), a), (w_t, (d, f), a), q + "dh",
                     N, f, d, "nn")
                gemm(f"L{l} w2 weight gradient", "bwd", l, mb, (dout, (N, d), a), (xin_t, (N, f), a), q + "w2.dw16",
                     d, f, N, "tn")
            g.add(f"L{l} w2 weight gradient cast bf16 to fp32", "memory copy", "copy/cast", "bwd",
                  [(q + "w2.dw16", (d, f), a)], [(f"l{l}.w2.grad", (d, f), "fp32")], layer=l, mb=mb)
            g.add(f"L{l} swiglu backward", "elementwise", "elementwise", "bwd",
                  [(q + "dh", (N, f), a), (q + "g", (N, f), a), (q + "u", (N, f), a)],
                  [(q + "dg", (N, f), a), (q + "du", (N, f), a)], layer=l, mb=mb)
            for nm, dout in (("w1", q + "dg"), ("w3", q + "du")):
                gemm(f"L{l} {nm} input gradient", "bwd", l, mb, (dout, (N, f), a), (q + f"{nm}.w16", (f, d), a),
                     q + f"dn2.{nm}", N, d, f, "nn")
                gemm(f"L{l} {nm} weight gradient", "bwd", l, mb, (dout, (N, f), a), (q + "n2", act, a),
                     q + f"{nm}.dw16", f, d, N, "tn")
                g.add(f"L{l} {nm} weight gradient cast bf16 to fp32", "memory copy", "copy/cast", "bwd",
                      [(q + f"{nm}.dw16", (f, d), a)], [(f"l{l}.{nm}.grad", (f, d), "fp32")], layer=l, mb=mb)
            tp_allreduce(f"L{l} tp all reduce (mlp input gradient)", "bwd", l, mb, q + "dn2.w3")
            g.add(f"L{l} add_rmsnorm backward", "reduction", "elementwise", "bwd",
                  [(q + "dn2.w3", act, a), (dx, act, "fp32"), (q + "s", act, "fp32")],
                  [(q + "ds", act, "fp32"), (q + "da", act, a)], layer=l, mb=mb)
            gemm(f"L{l} wo input gradient", "bwd", l, mb, (q + "da", act, a), (q + "wo.w16", (d, dq), a), q + "dot",
                 N, dq, d, "nn")
            gemm(f"L{l} wo weight gradient", "bwd", l, mb, (q + "da", act, a), (q + "ot", (N, dq), a), q + "wo.dw16",
                 d, dq, N, "tn")
            g.add(f"L{l} wo weight gradient cast bf16 to fp32", "memory copy", "copy/cast", "bwd",
                  [(q + "wo.dw16", (d, dq), a)], [(f"l{l}.wo.grad", (d, dq), "fp32")], layer=l, mb=mb)
            g.add(f"L{l} attention output transpose backward", "memory copy", "copy/cast", "bwd",
                  [(q + "dot", (N, dq), a)], [(q + "do", (N, dq), a)], layer=l, mb=mb)
            # flash attention backward: reads q, k, v, o, do and writes dq, dk, dv; the dO.O
            # preprocessing rereads o and do, and the fp32 dq accumulator is written and read.
            g.add(f"L{l} flash attention backward", "GEMM", "attention", "bwd",
                  [(q + "q", (N, dq), a), (q + "k", (N, dkv), a), (q + "v", (N, dkv), a), (q + "o", (N, dq), a),
                   (q + "do", (N, dq), a)],
                  [(q + "dq", (N, dq), a), (q + "dk", (N, dkv), a), (q + "dv", (N, dkv), a)],
                  flops=5.0 * B * T * T * dq, layer=l, mb=mb, key=("sdpa_bwd", B, H, kv, T, hd),
                  extra_read=N * dq * 2 * 2 + N * dq * 4, extra_write=N * dq * 4)
            g.add(f"L{l} rope_qkv backward", "elementwise", "elementwise", "bwd",
                  [(q + "dq", (N, dq), a), (q + "dk", (N, dkv), a), (q + "dv", (N, dkv), a)],
                  [(q + "dqkv", (N, dqkv), a)], layer=l, mb=mb)
            gemm(f"L{l} qkv input gradient", "bwd", l, mb, (q + "dqkv", (N, dqkv), a), (q + "qkv.w16", (dqkv, d), a),
                 q + "dn1", N, d, dqkv, "nn")
            gemm(f"L{l} qkv weight gradient", "bwd", l, mb, (q + "dqkv", (N, dqkv), a), (q + "n1", act, a),
                 q + "qkv.dw16", dqkv, d, N, "tn")
            g.add(f"L{l} qkv weight gradient cast bf16 to fp32", "memory copy", "copy/cast", "bwd",
                  [(q + "qkv.dw16", (dqkv, d), a)], [(f"l{l}.qkv.grad", (dqkv, d), "fp32")], layer=l, mb=mb)
            tp_allreduce(f"L{l} tp all reduce (attention input gradient)", "bwd", l, mb, q + "dn1")
            g.add(f"L{l} rmsnorm backward", "reduction", "elementwise", "bwd",
                  [(q + "dn1", act, a), (xin, act, "fp32")], [(q + "dxn", act, "fp32")], layer=l, mb=mb)
            ndx = f"{p}l{l}.dxin"
            g.add(f"L{l} residual stream gradient accumulate", "elementwise", "elementwise", "bwd",
                  [(q + "ds", act, "fp32"), (q + "dxn", act, "fp32")], [(ndx, act, "fp32")], layer=l, mb=mb)
            dx = ndx
            if mb == m - 1 and dp > 1:
                # gradients of this layer are final after the last microbatch: all reduce now
                lp = sum(o * i for _, o, i in linears) + 2 * d
                g.add(f"L{l} dp gradient all reduce", "collective", "comm", "bwd",
                      [(f"l{l}.{nm}.grad", (o, i), "fp32") for nm, o, i in linears], [(f"l{l}.grad.reduced", (lp,),
                                                                                       "fp32")],
                      layer=l, mb=mb, comm={"kind": "all_reduce", "group": "dp", "pg": dp_group, "bytes": lp * 4})
        if first:
            g.add("embedding backward zero fill", "memory copy", "loss", "bwd", [], [("emb.grad", (s.vocab * d // tp,),
                                                                                      "fp32")], mb=mb)
            g.add("embedding backward index add", "elementwise", "loss", "bwd", [(dx, act, "fp32")],
                  [("emb.grad.rows", act, "fp32")], mb=mb)
        else:
            g.add(f"send gradient to stage {pp_i - 1}", "send", "comm", "bwd", [(dx, act, "fp32")], [], mb=mb,
                  comm={"kind": "send", "src": g.rank, "dst": peer(pp_i - 1), "bytes": N * d * 2,
                        "tag": tag(mb, 1, pp_i - 1)})

    for phase, mb in schedule_1f1b(pp_i, pp, m):
        if phase == "F":
            x = forward(mb)
            if not last:
                g.add(f"send activation to stage {pp_i + 1}", "send", "comm", "fwd", [(x, act, "fp32")], [], mb=mb,
                      comm={"kind": "send", "src": g.rank, "dst": peer(pp_i + 1), "bytes": N * d * 2,
                            "tag": tag(mb, 0, pp_i)})
        else:
            backward(mb)

    # extras (embedding, head, final norm) all reduced too, then the optimizer
    reduced = [n["id"] for n in g.nodes if n.get("comm", {}).get("group") == "dp"]
    extra_p = (d + V * d if last else 0) + (s.vocab * d // tp if first else 0)
    if dp > 1 and extra_p:
        reduced.append(g.add("dp gradient all reduce (embedding, final norm, lm head)", "collective", "comm", "bwd",
                             [], [], comm={"kind": "all_reduce", "group": "dp", "pg": dp_group, "bytes": extra_p * 4},
                             extra_deps=[g.last_compute]))
    P = (sum(o * i for _, o, i in linears) + 2 * d) * Lp + extra_p
    g.add("fused AdamW", "elementwise", "optimizer", "opt", [], [], extra_read=P * 16, extra_write=P * 12,
          extra_deps=reduced)
    return g


def make_groups(dp, tp, pp):
    """Communicator groups: id -> ranks. Ids start at 1 (0 means the default group)."""
    groups = {"tp": {}, "dp": {}, "pp": {}}
    table = {}
    nid = 1
    for dp_i in range(dp):
        for pp_i in range(pp):
            table[nid] = [rank_of(dp_i, pp_i, t, tp, pp) for t in range(tp)]
            groups["tp"][(dp_i, pp_i)] = nid
            nid += 1
    for pp_i in range(pp):
        for tp_i in range(tp):
            table[nid] = [rank_of(i, pp_i, tp_i, tp, pp) for i in range(dp)]
            groups["dp"][(pp_i, tp_i)] = nid
            nid += 1
    for dp_i in range(dp):
        for tp_i in range(tp):
            table[nid] = [rank_of(dp_i, i, tp_i, tp, pp) for i in range(pp)]
            groups["pp"][(dp_i, tp_i)] = nid
            nid += 1
    return groups, table


# --------------------------------------------------------------------------
# Chakra conversion
# --------------------------------------------------------------------------

def to_chakra(node):
    typ = TYPE_TO_CHAKRA[node["type"]]
    io = lambda ts: {"values": "", "shapes": json.dumps([t["shape"] for t in ts]),
                     "types": json.dumps([t["dtype"] for t in ts])}
    attrs = [("is_cpu_op", "bool_val", False), ("op_class", "string_val", node["type"]),
             ("phase", "string_val", node["phase"])]
    if typ == cw.COMP_NODE:
        # ASTRA-sim's roofline computes num_ops / min(peak, bw x num_ops / tensor_size); with
        # num_ops = 0 that is 0 / 0. One operation instead makes it tensor_size / bw, the
        # correct memory bound time. (The JSON keeps the exact FLOPs.)
        attrs += [("num_ops", "int64_val", max(1, int(node["flops"]))),
                  ("tensor_size", "uint64_val", int(node["bytes_read"] + node["bytes_written"]))]
    else:
        c = node["comm"]
        attrs.append(("comm_size", "int64_val", int(c["bytes"])))
        if typ == cw.COMM_COLL_NODE:
            attrs += [("comm_type", "int64_val", cw.ALL_REDUCE), ("pg_name", "string_val", str(c["pg"]))]
        else:
            attrs += [("comm_src", "int32_val", c["src"]), ("comm_dst", "int32_val", c["dst"]),
                      ("comm_tag", "int32_val", c["tag"])]
    return {"id": node["id"], "name": node["name"], "type": typ, "data_deps": node["data_deps"],
            "ctrl_deps": node["ctrl_deps"], "duration_micros": max(1, round(node["predicted_us"])) if typ == cw.COMP_NODE
            else 0, "inputs": io(node["inputs"]), "outputs": io(node["outputs"]), "attrs": attrs}


# --------------------------------------------------------------------------
# Cross check against the performance model inventory
# --------------------------------------------------------------------------

def cross_check(s, B, T, tp, m, graphs_by_stage):
    """Sum over one rank per pipeline stage (tp rank 0, dp rank 0) must equal the perf model:
    m x (the step without the optimizer) + the optimizer, per category."""
    ops = inventory(s, B, T, "bf16", "sdpa", "triton", "fused", tp=tp)
    ref = defaultdict(lambda: [0.0, 0.0])
    for o in ops:
        k = 1 if o.cat == "optimizer" else m
        ref[o.cat][0] += o.total_flops * k
        ref[o.cat][1] += o.total_bytes * k
    got = defaultdict(lambda: [0.0, 0.0])
    for g in graphs_by_stage:
        for n in g.nodes:
            if n["category"] == "comm":
                continue
            got[n["category"]][0] += n["flops"]
            got[n["category"]][1] += n["bytes_read"] + n["bytes_written"]
    rows, ok = [], True
    for cat in sorted(set(ref) | set(got)):
        rf, rb = ref[cat]
        gf, gb = got[cat]
        ef = abs(gf - rf) / rf if rf else (0.0 if gf == 0 else 1.0)
        eb = abs(gb - rb) / rb if rb else (0.0 if gb == 0 else 1.0)
        ok &= ef < 1e-9 and eb < 1e-9
        rows.append({"category": cat, "export_flops": gf, "model_flops": rf, "flops_rel_diff": ef,
                     "export_bytes": gb, "model_bytes": rb, "bytes_rel_diff": eb})
    tf = sum(r["export_flops"] for r in rows)
    mf = sum(r["model_flops"] for r in rows)
    tb = sum(r["export_bytes"] for r in rows)
    mb_ = sum(r["model_bytes"] for r in rows)
    return {"pass": ok, "categories": rows, "total_flops": [tf, mf], "total_bytes": [tb, mb_]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="medium", choices=list(SIZES))
    p.add_argument("--batch", type=int, default=4, help="microbatch size in sequences")
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--microbatches", type=int, default=1)
    p.add_argument("--machine-label", default="final")
    p.add_argument("--out", default=None, help="output directory (default profiles/workload/<config>)")
    args = p.parse_args()

    s = SIZES[args.size]
    if s.L % args.pp or s.H % args.tp or s.kvh % args.tp or s.ffn % args.tp:
        sys.exit("pp must divide the layers and tp must divide heads, kv heads and ffn")
    if args.pp > 1 and args.microbatches < args.pp:
        sys.exit("1F1B needs at least as many microbatches as pipeline stages")
    cal = RESULTS / f"calibration_{args.machine_label}.json"
    shape_cache = RESULTS / f"shape_calibration_{args.machine_label}.json"
    extras = json.loads(shape_cache.read_text()).get("extras") if shape_cache.exists() else None
    machine = Machine.from_calibration(cal, extras)
    shape_times = load_shape_times(shape_cache)

    tag = f"{args.size}_b{args.batch}_s{args.seq}_dp{args.dp}_tp{args.tp}_pp{args.pp}_m{args.microbatches}"
    out = Path(args.out) if args.out else ROOT / "profiles" / "workload" / tag
    out.mkdir(parents=True, exist_ok=True)
    groups, table = make_groups(args.dp, args.tp, args.pp)
    (out / "comm_group.json").write_text(json.dumps({str(k): v for k, v in table.items()}, indent=1))

    n_ranks = args.dp * args.tp * args.pp
    per_rank, stage_graphs = [], {}
    for dp_i in range(args.dp):
        for pp_i in range(args.pp):
            for tp_i in range(args.tp):
                g = build_rank(s, args.batch, args.seq, args.dp, args.tp, args.pp, args.microbatches,
                               dp_i, pp_i, tp_i, machine, shape_times, groups)
                if dp_i == 0 and tp_i == 0:
                    stage_graphs[pp_i] = g
                (out / f"graph.rank{g.rank}.json").write_text(json.dumps(
                    {"rank": g.rank, "dp_rank": dp_i, "pp_rank": pp_i, "tp_rank": tp_i, "nodes": g.nodes}))
                cw.write_et(out / f"workload.{g.rank}.et", [to_chakra(n) for n in g.nodes],
                            [("source", "string_val", "llm-training-perf workload_export"),
                             ("model", "string_val", args.size), ("layout", "string_val", tag)])
                types = defaultdict(int)
                for n in g.nodes:
                    types[n["type"]] += 1
                per_rank.append({"rank": g.rank, "nodes": len(g.nodes), "types": dict(types),
                                 "flops": sum(n["flops"] for n in g.nodes),
                                 "bytes": sum(n["bytes_read"] + n["bytes_written"] for n in g.nodes
                                              if n["category"] != "comm"),
                                 "comm_bytes": sum(n["comm"]["bytes"] for n in g.nodes if "comm" in n
                                                   and n["comm"]["kind"] != "recv"),
                                 "predicted_compute_ms": round(sum(n["predicted_us"] for n in g.nodes) / 1e3, 3)})

    check = cross_check(s, args.batch, args.seq, args.tp, args.microbatches, [stage_graphs[i] for i in range(args.pp)])
    summary = {"config": vars(args), "tag": tag, "ranks": n_ranks, "arch": machine.arch,
               "time": time.strftime("%Y-%m-%d %H:%M:%S"), "per_rank": per_rank, "cross_check": check,
               "files": {"json": "graph.rank<r>.json", "chakra": "workload.<r>.et", "comm_groups": "comm_group.json"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    r0 = per_rank[0]
    print(f"{tag}: {n_ranks} ranks, {r0['nodes']} nodes on rank 0 ({r0['types']}), "
          f"predicted compute {r0['predicted_compute_ms']} ms on rank 0")
    print(f"cross check against perf_model inventory: {'PASS' if check['pass'] else 'FAIL'}")
    for r in check["categories"]:
        print(f"  {r['category']:<12} FLOPs {r['export_flops'] / 1e12:10.4f} T vs {r['model_flops'] / 1e12:10.4f} T "
              f"({r['flops_rel_diff']:.1e})   bytes {r['export_bytes'] / 1e9:9.3f} GB vs {r['model_bytes'] / 1e9:9.3f} GB "
              f"({r['bytes_rel_diff']:.1e})")
    print(f"wrote {out}")
    return 0 if check["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
