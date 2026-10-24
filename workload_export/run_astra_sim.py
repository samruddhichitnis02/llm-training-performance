"""Run an exported workload in ASTRA-sim (analytical network backend) and compare with
the scaling model in perf_model/scaling.py.

ASTRA-sim is an open source distributed training simulator (github.com/astra-sim/astra-sim)
that reads Chakra execution traces, one per NPU. This script writes the system, network
and memory configs for an exported layout, runs the simulator, and reads back the
simulated step time of every rank.

Two compute modes:
  replay     each compute node takes its duration_micros, the per op time predicted by
             the validated performance model (rounded to whole microseconds, a Chakra limit)
  roofline   ASTRA-sim's own roofline: num_ops / min(peak, bandwidth x num_ops / tensor_size),
             with the calibrated bf16 peak and memory bandwidth of this GPU

The network uses the same constants as perf_model/links.json (NVLink inside an 8 GPU
node, InfiniBand between nodes), so the comparison tests the schedule and the overlap of
communication with compute, not the link numbers. Those are still not measured here.

Needs the ASTRA-sim binary: build it with build/astra_analytical/build.sh in a clone of
the repo and pass --astra-sim, or set ASTRA_SIM_BIN.

  python workload_export/run_astra_sim.py profiles/workload/medium_b2_s1024_dp2_tp2_pp2_m4
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perf_model.roofline import RESULTS, SIZES, Machine, load_shape_times  # noqa: E402

DEFAULT_BIN = Path.home() / "sim/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
FINISHED = re.compile(r"sys\[(\d+)\] finished, (\d+) cycles, exposed communication (\d+) cycles")


def network_yml(n, gpn, links):
    intra, inter = links["links"]["nvlink"], links["links"]["infiniband"]
    if n <= gpn:
        return (f"topology: [ Switch ]\nnpus_count: [ {n} ]\nbandwidth: [ {intra['bw_gbs']} ]\n"
                f"latency: [ {intra['latency_us'] * 1000} ]\n")
    return (f"topology: [ Switch, Switch ]\nnpus_count: [ {gpn}, {n // gpn} ]\n"
            f"bandwidth: [ {intra['bw_gbs']}, {inter['bw_gbs']} ]\n"
            f"latency: [ {intra['latency_us'] * 1000}, {inter['latency_us'] * 1000} ]\n")


def system_json(mode, machine):
    cfg = {"scheduling-policy": "FIFO", "endpoint-delay": 10, "active-chunks-per-dimension": 1,
           "preferred-dataset-splits": 4, "all-reduce-implementation": ["ring"],
           "all-gather-implementation": ["ring"], "reduce-scatter-implementation": ["ring"],
           "all-to-all-implementation": ["ring"], "collective-optimization": "localBWAware",
           "local-mem-bw": machine.bandwidth_gbs, "boost-mode": 0,
           "roofline-enabled": 1 if mode == "roofline" else 0, "peak-perf": machine.peak_tflops["bf16"]}
    return cfg


def run(binary, wl_dir, mode, machine, links, workdir):
    summary = json.loads((wl_dir / "summary.json").read_text())
    n = summary["ranks"]
    gpn = links["systems"]["nvlink_ib"]["gpus_per_node"]
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"system_{mode}.json").write_text(json.dumps(system_json(mode, machine), indent=1))
    (workdir / "network.yml").write_text(network_yml(n, gpn, links))
    (workdir / "memory.json").write_text(json.dumps({"memory-type": "NO_MEMORY_EXPANSION"}))
    cmd = [str(binary), f"--workload-configuration={wl_dir / 'workload'}",
           f"--system-configuration={workdir / f'system_{mode}.json'}",
           f"--remote-memory-configuration={workdir / 'memory.json'}",
           f"--network-configuration={workdir / 'network.yml'}",
           f"--comm-group-configuration={wl_dir / 'comm_group.json'}"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0
    out = p.stdout + p.stderr
    (workdir / f"astra_sim_{mode}.log").write_text(out)
    ranks = {int(r): (int(c), int(e)) for r, c, e in FINISHED.findall(out)}
    # This ASTRA-sim build can crash while printing statistics at shutdown, after the
    # simulation itself is over, so some "finished" lines are missing. Ranks of one
    # pipeline stage are locked together by their collectives and finish at the same tick.
    # The result is accepted if every stage has finished ranks and they agree exactly.
    tp, pp = summary["config"]["tp"], summary["config"]["pp"]
    stage = lambda r: (r % (tp * pp)) // tp
    by_stage = {}
    for r, (c, _) in ranks.items():
        by_stage.setdefault(stage(r), set()).add(c)
    complete = len(ranks) == n and p.returncode == 0
    usable = len(by_stage) == pp and all(len(v) == 1 for v in by_stage.values())
    if not (complete or usable):
        tail = "\n".join(out.strip().splitlines()[-8:])
        raise RuntimeError(f"ASTRA-sim {mode} run failed (exit {p.returncode}, {len(ranks)} of {n} ranks finished):\n{tail}")
    step_ns = max(c for c, _ in ranks.values())
    exposed = max(e for _, e in ranks.values())
    return {"mode": mode, "ranks": n, "ranks_reported": len(ranks), "exit_code": p.returncode,
            "crashed_at_shutdown": not complete, "step_ms": step_ns / 1e6, "max_exposed_comm_ms": exposed / 1e6,
            "per_stage_ms": {s_: next(iter(v)) / 1e6 for s_, v in sorted(by_stage.items())},
            "sim_wall_s": round(wall, 1)}


def scaling_prediction(cfg, machine, links):
    """The analytical scaling model for exactly this layout."""
    from perf_model.scaling import Link, StageCompute, evaluate
    s = SIZES[cfg["size"]]
    L = {k: Link(v["bw_gbs"], v["latency_us"]) for k, v in links["links"].items()}
    comp = StageCompute(s, cfg["seq"], machine, load_shape_times(RESULTS / "shape_calibration_final.json"))
    n = cfg["dp"] * cfg["tp"] * cfg["pp"]
    G = cfg["dp"] * cfg["batch"] * cfg["microbatches"]
    big = dict(links, gpu_memory_gb=1e9)      # memory limit off: the export was chosen by hand
    r = evaluate(s, cfg["seq"], G, n, cfg["tp"], cfg["pp"], cfg["batch"], comp, L, links["systems"]["nvlink_ib"], big)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workload_dirs", nargs="+")
    p.add_argument("--astra-sim", default=os.environ.get("ASTRA_SIM_BIN", str(DEFAULT_BIN)))
    p.add_argument("--modes", default="replay,roofline")
    p.add_argument("--machine-label", default="final")
    args = p.parse_args()

    binary = Path(args.astra_sim)
    if not binary.exists():
        sys.exit(f"ASTRA-sim binary not found at {binary}")
    links = json.loads((ROOT / "perf_model" / "links.json").read_text())
    shape_cache = RESULTS / f"shape_calibration_{args.machine_label}.json"
    machine = Machine.from_calibration(RESULTS / f"calibration_{args.machine_label}.json",
                                       json.loads(shape_cache.read_text()).get("extras"))
    commit = subprocess.run(["git", "-C", str(binary.parents[4]), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()

    results = []
    for d in args.workload_dirs:
        wl = Path(d)
        summary = json.loads((wl / "summary.json").read_text())
        cfg = summary["config"]
        export_compute_ms = summary["per_rank"][0]["predicted_compute_ms"]
        rounded_ms = sum(max(1, round(n_["predicted_us"])) for n_ in
                         json.loads((wl / "graph.rank0.json").read_text())["nodes"]
                         if n_["category"] != "comm") / 1e3
        row = {"workload": summary["tag"], "ranks": summary["ranks"], "export_compute_ms_rank0": export_compute_ms,
               "replay_compute_ms_rank0_after_rounding": rounded_ms}
        sc = scaling_prediction(cfg, machine, links)
        if sc:
            row["scaling_model_step_ms"] = sc["step_s"] * 1e3
            row["scaling_model_breakdown_ms"] = {k: v * 1e3 for k, v in sc["breakdown_s"].items()}
        for mode in args.modes.split(","):
            try:
                r = run(binary, wl, mode, machine, links, wl / "astra_sim")
            except Exception as e:
                r = {"mode": mode, "error": str(e)}
            row[mode] = r
            if "step_ms" in r:
                msg = f"step {r['step_ms']:.2f} ms, exposed comm {r['max_exposed_comm_ms']:.2f} ms"
                if r["crashed_at_shutdown"]:
                    msg += f" [{r['ranks_reported']} of {r['ranks']} ranks reported before a crash at shutdown]"
                if sc:
                    r["vs_scaling_model_pct"] = round(100 * (r["step_ms"] / row["scaling_model_step_ms"] - 1), 2)
                    msg += f" ({r['vs_scaling_model_pct']:+.1f}% vs the scaling model)"
                print(f"{summary['tag']} [{mode}]: {msg}, simulated in {r['sim_wall_s']} s")
            else:
                print(f"{summary['tag']} [{mode}]: {r['error']}")
        if sc:
            print(f"{summary['tag']}: scaling model {row['scaling_model_step_ms']:.2f} ms "
                  f"({', '.join(f'{k} {v:.1f}' for k, v in row['scaling_model_breakdown_ms'].items())})")
        print(f"  rank 0 compute: {export_compute_ms:.2f} ms predicted, {rounded_ms:.2f} ms after rounding to whole us")
        results.append(row)

    out = {"astra_sim_commit": commit, "binary": binary.name, "arch": machine.arch,
           "machine_label": args.machine_label, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network": "NVLink switch inside 8 GPU nodes, InfiniBand switch between nodes (perf_model/links.json)",
           "results": results}
    path = RESULTS / f"astra_sim_{args.machine_label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
