"""Checks on an exported workload directory (from export_trace.py).

1. Round trip: every workload.<r>.et decodes (with chakra_writer's reader) to the same
   nodes, dependencies and attributes as graph.rank<r>.json.
2. Official decoder: if the MLCommons chakra package is installed, the .et files are also
   decoded with its generated protobuf classes (et_def_pb2) and compared.
3. Graph validity: ids are 0..n-1, every dependency points to an earlier node (so the
   graph is acyclic), every send has a matching receive on the peer rank with the same
   tag and size, and every rank in a communicator group issues the same collectives with
   the same sizes in the same order (otherwise a simulator would hang).

  python workload_export/check_export.py profiles/workload/medium_b2_s1024_dp2_tp2_pp2_m4
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from workload_export import chakra_writer as cw  # noqa: E402


def load(d):
    graphs = {}
    for p in sorted(d.glob("graph.rank*.json")):
        g = json.loads(p.read_text())
        graphs[g["rank"]] = g
    return graphs, json.loads((d / "comm_group.json").read_text())


def check_roundtrip(d, graphs):
    errors = []
    for r, g in graphs.items():
        meta, nodes = cw.read_et(d / f"workload.{r}.et")
        if len(nodes) != len(g["nodes"]):
            errors.append(f"rank {r}: {len(nodes)} nodes in .et, {len(g['nodes'])} in JSON")
            continue
        for a, b in zip(nodes, g["nodes"]):
            if a["id"] != b["id"] or a["name"] != b["name"] or sorted(a["data_deps"]) != b["data_deps"] \
                    or sorted(a["ctrl_deps"]) != b["ctrl_deps"]:
                errors.append(f"rank {r} node {b['id']}: id, name or deps differ")
                break
            at = a["attrs"]
            if "comm" in b:
                if at.get("comm_size") != b["comm"]["bytes"]:
                    errors.append(f"rank {r} node {b['id']}: comm_size differs")
                    break
            elif at.get("num_ops") != max(1, int(b["flops"])) or \
                    at.get("tensor_size") != int(b["bytes_read"] + b["bytes_written"]):
                errors.append(f"rank {r} node {b['id']}: num_ops or tensor_size differs")
                break
    return errors


def check_official(d, graphs):
    try:
        from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
        from chakra.src.third_party.utils.protolib import decodeMessage, openFileRd
    except Exception as e:  # package not installed
        return None, f"chakra package not installed ({type(e).__name__}); skipped"
    errors = []
    for r, g in graphs.items():
        fh = openFileRd(str(d / f"workload.{r}.et"))
        meta = GlobalMetadata()
        decodeMessage(fh, meta)
        count = 0
        node = Node()
        while decodeMessage(fh, node):
            ref = g["nodes"][count]
            if node.id != ref["id"] or node.name != ref["name"] or list(node.data_deps) != ref["data_deps"]:
                errors.append(f"rank {r} node {ref['id']}: official decoder disagrees")
                break
            count += 1
            node = Node()
        fh.close()
        if count != len(g["nodes"]):
            errors.append(f"rank {r}: official decoder read {count} of {len(g['nodes'])} nodes")
    return errors, f"decoded {len(graphs)} files with chakra et_def_pb2 (schema metadata version {meta.version!r})"


def check_graph(graphs, groups):
    errors = []
    sends, recvs = {}, {}
    colls = defaultdict(list)
    for r, g in graphs.items():
        for i, n in enumerate(g["nodes"]):
            if n["id"] != i:
                errors.append(f"rank {r}: node ids are not 0..n-1")
                break
            if any(dep >= i for dep in n["data_deps"] + n["ctrl_deps"]):
                errors.append(f"rank {r} node {i}: dependency on a later node")
            c = n.get("comm")
            if not c:
                continue
            if c["kind"] == "send":
                sends[(c["src"], c["dst"], c["tag"])] = c["bytes"]
            elif c["kind"] == "recv":
                recvs[(c["src"], c["dst"], c["tag"])] = c["bytes"]
            else:
                if r not in groups[str(c["pg"])]:
                    errors.append(f"rank {r} node {i}: not a member of group {c['pg']}")
                colls[(c["pg"], r)].append(c["bytes"])
    if sends != recvs:
        missing = set(sends) ^ set(recvs)
        errors.append(f"{len(missing)} sends without a matching receive (or the reverse)")
    for pg, ranks in groups.items():
        seqs = {tuple(colls.get((int(pg), r), [])) for r in ranks if r in graphs}
        if len(seqs) > 1:
            errors.append(f"group {pg}: ranks issue different collective sequences")
    n_coll = sum(len(v) for v in colls.values())
    return errors, f"{len(sends)} send/recv pairs matched, {n_coll} collective calls consistent across {len(groups)} groups"


def main():
    d = Path(sys.argv[1])
    graphs, groups = load(d)
    print(f"{d.name}: {len(graphs)} ranks")
    ok = True
    e = check_roundtrip(d, graphs)
    print(f"  round trip .et <-> JSON: {'PASS' if not e else 'FAIL'}")
    for x in e[:5]:
        print("    " + x)
    ok &= not e
    e, msg = check_official(d, graphs)
    print(f"  official chakra decoder: {'SKIP' if e is None else ('PASS' if not e else 'FAIL')} ({msg})")
    if e:
        ok = False
        for x in e[:5]:
            print("    " + x)
    e, msg = check_graph(graphs, groups)
    print(f"  graph validity: {'PASS' if not e else 'FAIL'} ({msg})")
    for x in e[:5]:
        print("    " + x)
    ok &= not e
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
