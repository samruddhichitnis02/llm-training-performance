# Simulator workload export

NVIDIA's own processor and system simulators are proprietary and not available outside NVIDIA. This folder produces an open format description of the same kind of workload: one training step of the Llama style model as an operator graph, written as plain JSON and as an MLCommons Chakra execution trace, which the open source ASTRA-sim simulator reads.

| File | What it does |
|---|---|
| `export_trace.py` | Builds the graph of one training step, one per rank, for a chosen dp x tp x pp layout. Every node has its type, tensor shapes and dtypes, FLOPs, bytes, dependencies and a predicted duration. It also cross checks the totals against `perf_model/roofline.py`. |
| `chakra_writer.py` | Writes and reads the Chakra protobuf format with no dependencies (field numbers from `et_def.proto`). |
| `check_export.py` | Round trips the `.et` files, decodes them with the official `chakra` package if it is installed, and checks that the graph is acyclic, that sends and receives match, and that every group issues the same collectives. |
| `observe_pytorch.py` | Records a real step with PyTorch's `ExecutionTraceObserver` and compares its GEMM shapes, their order and the attention calls with the export. |
| `run_astra_sim.py` | Runs an export in ASTRA-sim's analytical backend and compares the simulated step time with `perf_model/scaling.py`. |

## Run it

```bash
python workload_export/export_trace.py --size medium --batch 2 --seq 1024 --dp 2 --tp 2 --pp 2 --microbatches 4
python workload_export/check_export.py profiles/workload/medium_b2_s1024_dp2_tp2_pp2_m4
python workload_export/observe_pytorch.py --size medium --batch 4 --seq 1024
```

Large exports go to `profiles/workload/` (not committed). A small complete example, `tiny` on 8 ranks with its ASTRA-sim configs and logs, is committed in `results/workload_example/tiny_dp2_tp2_pp2/`.

## ASTRA-sim

Build once (needs `cmake`, `libprotobuf-dev`, `protobuf-compiler`). The results in this repo used commit `4ce9ecfe746a3ee5fe47559a11d91644d641fa7f`.

```bash
git clone https://github.com/astra-sim/astra-sim.git ~/sim/astra-sim
cd ~/sim/astra-sim && git checkout 4ce9ecfe746a3ee5fe47559a11d91644d641fa7f
git submodule update --init --recursive
bash build/astra_analytical/build.sh
```

Then:

```bash
python workload_export/run_astra_sim.py profiles/workload/medium_b2_s1024_dp2_tp2_pp2_m4
```

This writes the system, network and memory configs next to the workload and runs ASTRA-sim in two modes. In replay mode, compute nodes take their predicted durations. In roofline mode, ASTRA-sim computes them from `num_ops` and `tensor_size` with this GPU's calibrated peak and bandwidth.

Known issue with this ASTRA-sim build: it sometimes crashes while printing statistics at shutdown, after the simulation has finished. `run_astra_sim.py` accepts a run only if every pipeline stage has reported and all reporting ranks of a stage agree to the tick. The JSON output records how many ranks reported.
