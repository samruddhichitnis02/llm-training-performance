# Results log

Numbers recorded at each milestone, in order. Every entry names its architecture tag and machine label. The raw JSON for each entry is in this folder.

## Milestone 1: benchmark harness and calibration

Machine: `sm_120`, label `final`, 1 GPU, WSL2 Ubuntu 24.04, PyTorch 2.11.0+cu128, CUDA 12.8.

### Calibration

Source: `calibration_final.json`. Dense matmul on 8192 by 8192 matrices, memory bandwidth from a 1 GB device to device copy.

| Precision | Measured TFLOPs | vs bf16 |
|---|---|---|
| fp32 | 65.3 | 0.29x |
| tf32 | 110.1 | 0.48x |
| fp16 | 220.3 | 0.97x |
| bf16 | 227.5 | 1.00x |
| fp8 (e4m3, tensorwise scale) | 447.1 | 1.97x |
| fp4 (NVFP4: e2m1 values, e4m3 scale per 16) | 1291.7 | 5.68x |

Memory bandwidth: 1441.3 GB/s. Launch overhead: 6.5 us per kernel (tiny kernels queued back to back).

Low precision format check with `torch._scaled_mm` on PyTorch 2.11.0+cu128 (4096 by 4096 matmul, bf16 output):

| Format | Values | Scales | Result |
|---|---|---|---|
| FP8 | e4m3 | one fp32 scale per tensor | works |
| NVFP4 | e2m1, two per byte | one e4m3 scale per 16 values | works |
| MXFP4 | e2m1, two per byte | one e8m0 scale per 32 values | rejected: "Invalid scaling configuration" |

So FP4 work on this machine uses NVFP4.

### First optimization ladder (rows 1 to 5)

Source: `milestones/m1_sweep_final_sm_120_small_b8_s1024.json`. Model `small` (134M params), batch 8, seq 1024, 5 warmup steps, median of 20 timed steps. MFU is against the bf16 matmul peak reached in the same run, 227.3 TFLOPs.

| Row | ms/step | tokens/s | Speedup | TFLOPs | MFU | Peak GB |
|---|---|---|---|---|---|---|
| 1 fp32 baseline, naive attention | 223.72 | 36,618 | 1.00 | 28.21 | 12.4% | 15.94 |
| 2 bf16 autocast | 155.78 | 52,586 | 1.44 | 40.51 | 17.8% | 16.06 |
| 3 + fused attention (SDPA) | 73.59 | 111,324 | 3.04 | 85.77 | 37.7% | 9.10 |
| 4 + fused Adam | 68.51 | 119,581 | 3.27 | 92.13 | 40.5% | 9.10 |
| 5 + torch.compile | 48.45 | 169,082 | 4.62 | 130.27 | 57.3% | 5.49 |
| 6 fused Triton kernels | not built yet | | | | | |
| 7 CUDA C++ add+RMSNorm | not built yet | | | | | |

All losses finite. Loss falls to about 2.5 because the model sees the same random batch every step; that is expected and does not affect step time.

What stands out: fused attention gives the biggest single jump (2x on its own, and 7 GB less memory) because naive attention builds a 1024 by 1024 score matrix per head. bf16 alone gains only 1.44x even though bf16 matmuls are 3.5x faster than fp32, which means a large part of the step is not matmul. The profiler in milestone 2 shows what that part is.

Correction made during milestone 2: the first version of this table used PyTorch's per parameter Adam loop for rows 1 to 3, because this PyTorch build did not pick the multi tensor Adam on its own. The profiler showed it (87 separate small kernels per Adam op, one per parameter). Rows 1 to 3 now use the multi tensor Adam explicitly, and the table above is the rerun. Row 4 against row 3 is now the fair fused Adam gain.

## Milestone 2: profiler

Script: `profile_step.py`. Step timed first without the profiler, then 3 steps recorded. Profiler wrapper rows (`aten::`, `autograd::`, `Optimizer`) are excluded from GPU time. Every kernel that runs inside `Optimizer.step` counts as optimizer.

Model `medium` (375M params), batch 4, seq 1024, bf16 autocast, SDPA. Sources: `profile_final_sm_120_medium_b4_s1024_*.json`.

| Bucket | bf16 + SDPA | + fused Adam | + fused Adam + torch.compile |
|---|---|---|---|
| matmul | 52.1% | 59.0% | 71.9% |
| attention | 5.7% | 6.4% | 7.9% |
| optimizer | 19.4% | 7.8% | 9.6% |
| memory copies (casts, contiguous) | 7.9% | 9.5% | 8.3% |
| elementwise and other | 14.8% | 17.2% | 2.4% |
| kernel launches per step | 3045 | 2921 | 1066 |
| GPU idle | 8.6% | 9.9% | 4.1% |
| clean step time | 117.1 ms | 103.7 ms | 79.9 ms |

What the profile says:
1. In the bf16 plus SDPA baseline, matmuls are only half the GPU time. The other half is the optimizer (19%), many small elementwise kernels (15%, about 1600 launches per step) and dtype casts (8%, about 800 launches). Attention is under 6% once it is fused.
2. Fused Adam cuts the optimizer from 20.8 ms to 7.3 ms per step.
3. `torch.compile` removes about 1900 launches per step and takes elementwise from 16 ms to 1.8 ms. That is the work the hand written Triton kernels in milestone 3 target, without a compiler.
4. On this `sm_120` GPU cuBLAS picks `cutlass_80_tensorop_bf16_s16816gemm` kernels for bf16, which are Ampere generation tensor core kernels. This goes into the architecture notes.

Compared with the expected finding in the spec (matmul about 40%, elementwise about 30%, optimizer about 15%, attention under 10%), the matmul share here is higher and elementwise lower. The direction is the same: after bf16 and fused attention, the non matmul kernels are about half the step.

Nsight Systems is not installed yet. Ubuntu's own package (2022.4) is too old for `sm_120`; a current version comes from NVIDIA's CUDA apt repository. The capture commands are in the `profile_step.py` docstring.

## Milestone 3: fused Triton kernels

Code: `kernels/triton_ops.py` (four fused ops, forward and backward, each a `torch.autograd.Function`), `kernels/test_kernels.py`, `kernels/bench_kernels.py`, `kernels/check_loss_parity.py`. Console output of every run below is saved in `milestones/m3_*.txt`.

### Correctness

Source: `milestones/m3_test_kernels.txt`. Every output and every gradient compared against a float32 PyTorch reference, relative norm error.

**76 of 76 checks passed.** Worst errors per group:

| Kernel | fp32 worst | bf16 worst | Tolerance fp32 / bf16 |
|---|---|---|---|
| rmsnorm, add_rmsnorm (D 768 and 2048, plus mixed bf16 a with fp32 residual) | 2.0e-7 | 2.5e-3 | 1e-4 / 3e-2 |
| swiglu (including a ragged shape) | 6.3e-8 | 2.4e-3 | 1e-4 / 3e-2 |
| rope_qkv (12 heads of 64, 16 heads of 128) | 3.9e-8 | 2.1e-3 | 1e-4 / 3e-2 |
| linear + cross entropy (N = 4133, V = 32000) | 8.4e-7 | 3.5e-3 | 1e-4 / 5e-2 |

The bf16 errors are about 2e-3, which is the rounding of bf16 itself (8 bit mantissa), so the kernels add no error of their own.

Training parity (`loss_parity_final_sm_120_small_torch_triton.json`): `small`, batch 8, seq 1024, 20 steps from the same initial weights. Step 0 loss is identical (10.5238 both). The curves stay within 1.8% of each other over 20 steps, which is the normal drift of bf16 rounding in training.

### Microbenchmarks (forward plus backward)

Source: `bench_kernels_final_sm_120_small_b8_s1024.json`. Real training shapes for `small` at batch 8, seq 1024 (8192 tokens). Bandwidth is minimum bytes moved divided by time, against the calibrated 1441 GB/s.

| Op | eager | torch.compile | Triton | Triton vs eager | Triton vs compile |
|---|---|---|---|---|---|
| rmsnorm (fp32 in, bf16 out) | 0.348 ms, 20% BW | 0.238 ms, 29% BW | 0.183 ms, 38% BW | 1.90x | 1.30x |
| add_rmsnorm (bf16 + fp32) | 0.408 ms, 30% BW | 0.295 ms, 42% BW | 0.237 ms, 52% BW | 1.72x | 1.24x |
| swiglu | 0.203 ms, 92% BW | 0.272 ms, 68% BW | 0.191 ms, 97% BW | 1.06x | 1.42x |
| rope_qkv | 0.595 ms, 18% BW | 0.443 ms, 24% BW | 0.204 ms, 51% BW | 2.92x | 2.17x |
| linear + cross entropy | 12.89 ms, 94 TFLOPs, 3.08 GB | 7.24 ms, 167 TFLOPs, 0.52 GB | 6.91 ms, 175 TFLOPs, 0.31 GB | 1.87x | 1.05x |

What it shows:
1. Every Triton kernel beats both eager and `torch.compile`. RoPE gains the most (2.9x) because the eager version does the split, the permute, the rotation and the cast as separate passes.
2. Eager swiglu already runs near the copy bandwidth. Its intermediates (33 MB each) fit in this GPU's large L2 cache, so the extra passes are cheap. Fusion still wins, but only by 6%.
3. The chunked cross entropy uses 10 times less extra memory than eager (0.31 GB against 3.08 GB) because the full 8192 by 32000 logits never exist.
4. The norm kernels reach only about 40 to 50% of bandwidth. One program per 768 wide row is a small amount of work per program. This is the obvious place to tune next.

### Ladder with row 6

Sources: `milestones/m3_sweep_final_sm_120_small_b8_s1024.json`, `milestones/m3_sweep_final_sm_120_medium_b4_s1024.json`.

| Row | small b8 s1024 ms | MFU | Peak GB | medium b4 s1024 ms | MFU | Peak GB |
|---|---|---|---|---|---|---|
| 1 fp32 baseline, naive attention | 223.31 | 12.6% | 15.94 | 320.37 | 13.2% | 21.16 |
| 2 bf16 autocast | 156.24 | 17.9% | 16.06 | 226.29 | 18.7% | 21.62 |
| 3 + fused attention (SDPA) | 73.86 | 38.0% | 9.10 | 117.51 | 36.0% | 12.36 |
| 4 + fused Adam | 68.76 | 40.8% | 9.10 | 103.60 | 40.9% | 12.36 |
| 5 + torch.compile | 48.34 | 58.0% | 5.49 | 80.17 | 52.8% | 9.74 |
| 6 fused Triton kernels (no compile) | 48.45 | 57.9% | 5.15 | 81.34 | 52.1% | 9.62 |
| 7 CUDA C++ add+RMSNorm | milestone 4 | | | | | |

Speedup at row 6: 4.61x for small, 3.94x for medium. All losses finite.

Four hand written kernels match what `torch.compile` gets from fusing the whole graph, with slightly less memory. Row 6 does not use the compiler at all.

### Profile after fusion

Source: `profile_final_sm_120_medium_b4_s1024_bf16_sdpa_fusedadam_triton.json`. Compared with the same config without the kernels (milestone 2):

| Bucket | fused Adam, torch ops | fused Adam, Triton kernels |
|---|---|---|
| matmul | 59.0% | 72.1% |
| attention | 6.4% | 7.5% |
| optimizer | 7.8% | 9.5% |
| memory copies | 9.5% | 3.1% |
| elementwise and other | 17.2% | 7.8% |
| launches per step | 2921 | 1177 |
| elementwise plus copies, ms per step | 25.0 | 8.4 |
| GPU idle | 9.9% | 5.7% |
| clean step time | 103.7 ms | 81.4 ms |

Launches per step dropped by 60%, and elementwise plus copy time dropped from 25 ms to 8.4 ms. Matmuls are now 72% of GPU time. The biggest remaining non matmul items are the optimizer (7.3 ms) and the fp32 to bf16 weight casts that autocast does every step (146 copy launches).

## Milestone 4: CUDA C++ add plus RMSNorm

Code: `kernels/cuda_ext/add_rmsnorm.cu` (forward and backward kernels), `kernels/cuda_ext/add_rmsnorm_ext.py` (built with `torch.utils.cpp_extension.load`, wrapped in an autograd Function with the same signature as the Triton version). Built with nvcc 12.8.93 for `sm_120` only. Console output of every run is in `milestones/m4_*.txt`.

Design:
- Forward: one block per row. Each thread loads 4 values at a time with one vectorized load (16 bytes for fp32, 8 for bf16) and keeps `s` in registers. The sum of squares is reduced with warp shuffles, then across warps through a 32 entry shared memory array.
- Backward: a persistent grid of 4 x 170 SMs = 680 blocks, each looping over rows. Every thread owns the same columns in every row, so the weight gradient accumulates in registers and each block writes one row of a (680, D) buffer.
- Templated on the dtype of `a`, `r` and `y` separately, so the bf16 sublayer output and the fp32 residual stream are read directly with no cast kernel.

One bug found by the tests: with D = 8192 the first version used 1024 threads per block, and the backward failed with `cudaErrorLaunchOutOfResources` (1024 threads leave only 64 registers per thread). Blocks are now capped at 512 threads with `__launch_bounds__(512)`, and wider rows give each thread more groups of 4.

### Correctness

Source: `milestones/m4_test_kernels.txt`. **142 of 142 checks passed.** The CUDA kernel is tested on exactly the same cases as the Triton one (D = 768, 2048 and 8192; fp32, bf16, and mixed bf16 plus fp32). Its errors match the Triton kernel's to two digits (fp32 about 5e-8, bf16 about 2e-3).

Training parity (`loss_parity_final_sm_120_small_torch_triton_cuda.json`): torch, Triton and CUDA models trained 20 steps from the same weights. Largest relative loss difference against torch: 1.4%. PASS.

### Microbenchmarks, remeasured

The milestone 3 microbenchmark timed back to back calls. For ops that take 0.1 to 0.4 ms, that mostly measured Python and autograd launch time, not the GPU, and repeated runs differed by up to 60% (Triton add_rmsnorm read 0.237 ms in one run and 0.373 ms in the next). When the CUDA kernel showed 120% of DRAM bandwidth, it was clear the inputs were also staying in the 96 MB L2 cache between calls.

`bench_kernels.py` now uses `triton.testing.do_bench`. It clears L2 before every call, times each call on the GPU with CUDA events, and reports the median. This replaces the milestone 3 table; `bench_kernels_final_sm_120_small_b8_s1024.json` holds the new numbers, and the old console output stays in `milestones/m3_bench_kernels_small.txt`.

Source: `bench_kernels_final_sm_120_small_b8_s1024.json` (`small`, 8192 tokens). BW = minimum bytes moved, against 1441 GB/s.

| Op (fwd + bwd) | eager | torch.compile | Triton | CUDA C++ | Best vs eager | Best vs compile |
|---|---|---|---|---|---|---|
| rmsnorm (fp32 in, bf16 out) | 0.328 ms, 21% | 0.179 ms, 39% | 0.072 ms, 97% | | 4.52x | 2.48x |
| add_rmsnorm (bf16 + fp32) | 0.383 ms, 32% | 0.175 ms, 70% | 0.123 ms, 100% | 0.122 ms, 100% | 3.13x | 1.43x |
| swiglu | 0.241 ms, 77% | 0.176 ms, 106% | 0.176 ms, 106% | | 1.37x | 1.00x |
| rope_qkv | 0.570 ms, 18% | 0.248 ms, 42% | 0.109 ms, 97% | | 5.25x | 2.28x |
| linear + cross entropy | 12.85 ms, 94 TFLOPs | 7.20 ms, 168 TFLOPs | 6.89 ms, 175 TFLOPs | | 1.87x | 1.05x |

What it shows:
1. Measured on the GPU alone, the Triton norm and RoPE kernels run at 97 to 100% of the copy bandwidth. They cannot go faster without moving fewer bytes. The 40 to 50% in milestone 3 was CPU launch time.
2. CUDA C++ and Triton add_rmsnorm are identical (0.1224 against 0.1225 ms). Both hit the bandwidth ceiling, so hand written CUDA has nothing left to win on this op.
3. Swiglu shows 106%: inside one forward plus backward call, the backward rereads `a` and `b` while they are still in L2, so fewer bytes come from DRAM than the minimum count assumes. `torch.compile` generates a swiglu as good as the hand written one.

### Ladder, all 7 rows

Sources: `milestones/m4_sweep_final_sm_120_small_b8_s1024.json`, `milestones/m4_sweep_final_sm_120_medium_b4_s1024.json`. MFU reference: 226.8 TFLOPs bf16 measured in each run.

| Row | small ms | tok/s | Speedup | MFU | GB | medium ms | tok/s | Speedup | MFU | GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 fp32 baseline, naive attention | 223.02 | 36,732 | 1.00 | 12.5% | 15.94 | 320.64 | 12,774 | 1.00 | 13.2% | 21.16 |
| 2 bf16 autocast | 155.91 | 52,542 | 1.43 | 17.8% | 16.06 | 226.01 | 18,123 | 1.42 | 18.8% | 21.62 |
| 3 + fused attention (SDPA) | 73.44 | 111,544 | 3.04 | 37.9% | 9.10 | 117.21 | 34,947 | 2.74 | 36.2% | 12.36 |
| 4 + fused Adam | 68.70 | 119,250 | 3.25 | 40.5% | 9.10 | 103.85 | 39,443 | 3.09 | 40.8% | 12.36 |
| 5 + torch.compile | 48.20 | 169,951 | 4.63 | 57.7% | 5.49 | 80.10 | 51,133 | 4.00 | 52.9% | 9.74 |
| 6 fused Triton kernels | 48.55 | 168,722 | 4.59 | 57.3% | 5.15 | 81.31 | 50,373 | 3.94 | 52.2% | 9.62 |
| 7 CUDA C++ add+RMSNorm | 48.36 | 169,390 | 4.61 | 57.5% | 5.15 | 81.20 | 50,441 | 3.95 | 52.2% | 9.62 |

Rows 6 and 7 are within 0.4% of each other, which is run to run noise. That matches the microbenchmark: both add_rmsnorm kernels sit at the bandwidth limit, and the op is about 1.6% of the step.

### Profile with the CUDA kernel

Source: `profile_final_sm_120_medium_b4_s1024_bf16_sdpa_fusedadam_cuda.json`. Bucket shares match the Triton profile to within 0.2 points (matmul 72.3%, elementwise and other 7.6%, 1177 launches per step, GPU idle 4.5%).

Inside the real step, the 24 add_rmsnorm calls per step take 1.29 ms with CUDA (0.64 forward, 0.65 backward) and about 1.03 ms with Triton (from the Triton profile, after subtracting the 25 plain rmsnorm calls). So the Triton version is slightly faster in place, even though the two tie in isolation. The step time difference this makes (0.26 ms of 81 ms) is below the noise.

### Nsight Systems

Nsight Systems 2024.6.2 came with the CUDA 12.8 toolkit. Trace of `medium`, batch 4, seq 1024, fused Adam, Triton kernels, with the `--nvtx` ranges:

```
nsys profile -t cuda,nvtx -o profiles/nsys_medium_triton --force-overwrite true \
  python bench_train.py --size medium --batch 4 --seq 1024 --fused-adam --kernels triton --nvtx --warmup 3 --steps 5
```

The `.nsys-rep` file is in `profiles/` (not committed). The summaries are committed as `milestones/m4_nsys_nvtx_sum.csv` and `milestones/m4_nsys_cuda_gpu_kern_sum.csv`.

From the NVTX summary: each timed step range lasts about 31.5 ms on the CPU, while the step takes 81 ms on the GPU. The CPU finishes queueing all the work for a step in well under half the GPU time, so this config is GPU bound and launch overhead is hidden. The first warmup step takes 454 ms (Triton compiling and autotuning the kernels, and cuBLAS choosing its kernels), which is why warmup steps are excluded from timing.

## Milestone 5: optimization notes, FP8 and NVFP4 training

Code: `kernels/lowp.py` (FP8 and NVFP4 linear layers, NVFP4 Triton quantizer), `kernels/test_lowp.py`, `kernels/ablate_kernels.py`, `--linear bf16|fp8|fp4` and `--fused-ops` in `bench_train.py`, `--linear` in `profile_step.py`, a GPU memory cap in all benchmark scripts. One note per optimization in `optimizations/`. Console output of every run is in `milestones/m5_*.txt`.

### Tests

- `milestones/m5_test_kernels.txt`: 142 of 142 (unchanged kernels, rerun).
- `milestones/m5_test_lowp.txt`: 22 of 22. NVFP4 GEMM matches a float32 matmul of the dequantized operands to 3e-7; the Triton quantizer gives bit identical output to the PyTorch reference; stochastic rounding is unbiased.

### Ladder, all 9 rows

Sources: `milestones/m5_sweep_final_sm_120_small_b8_s1024.json`, `milestones/m5_sweep_final_sm_120_medium_b4_s1024.json`. Every row in this run is about 4% slower than in milestone 4, and the bf16 reference matmul was also lower (219.2 and 215.7 against 226.8 TFLOPs), so the GPU ran at lower clocks during this session. Gains between rows match milestone 4.

| Row | small ms | Speedup | MFU | GB | medium ms | Speedup | MFU | GB |
|---|---|---|---|---|---|---|---|---|
| 1 fp32 baseline, naive attention | 233.85 | 1.00 | 12.3% | 15.94 | 333.97 | 1.00 | 13.4% | 21.16 |
| 2 bf16 autocast | 164.34 | 1.42 | 17.5% | 16.06 | 236.37 | 1.41 | 18.9% | 21.62 |
| 3 + fused attention (SDPA) | 76.85 | 3.04 | 37.5% | 9.10 | 121.81 | 2.74 | 36.6% | 12.36 |
| 4 + fused Adam | 71.61 | 3.27 | 40.2% | 9.10 | 107.50 | 3.11 | 41.5% | 12.36 |
| 5 + torch.compile | 50.23 | 4.66 | 57.3% | 5.49 | 83.51 | 4.00 | 53.4% | 9.74 |
| 6 fused Triton kernels | 50.55 | 4.63 | 57.0% | 5.15 | 84.70 | 3.94 | 52.7% | 9.62 |
| 7 CUDA C++ add+RMSNorm | 50.52 | 4.63 | 57.0% | 5.15 | 84.58 | 3.95 | 52.7% | 9.62 |
| 8 + FP8 linears (e4m3) | 44.67 | 5.24 | 64.5% | 4.73 | 71.40 | 4.68 | 62.5% | 8.85 |
| 9 + NVFP4 linears (e2m1) | 50.76 | 4.61 | 56.7% | 4.41 | 84.66 | 3.94 | 52.7% | 8.26 |

Best row: 8 (FP8), 5.24x faster than the fp32 baseline on `small`, 4.68x on `medium`. All losses finite. MFU uses the bf16 peak for every row.

### Each Triton kernel alone

Source: `ablation_kernels_final_sm_120_medium_b4_s1024.json`. `medium`, batch 4, seq 1024, fused Adam. Every row uses the fused model structure; "none" runs every op in PyTorch.

| Config | ms/step | Saved | Peak GB saved |
|---|---|---|---|
| none | 102.68 | | |
| only rmsnorm | 97.81 | 4.87 ms (4.7%) | 0.42 |
| only add_rmsnorm | 97.46 | 5.22 ms (5.1%) | 0.40 |
| only swiglu | 101.71 | 0.97 ms (0.9%) | 0.56 |
| only rope_qkv | 94.77 | 7.91 ms (7.7%) | 0.20 |
| only linear + cross entropy | 100.20 | 2.48 ms (2.4%) | 1.17 |
| all five | 80.51 | 22.17 ms (21.6%) | 2.74 |

The single kernel savings sum to 21.45 ms, close to the 22.17 ms of all five together.

### Activation checkpointing

Source: `milestones/m5_ckpt_medium_*.json`, `milestones/m5_ckpt_medium.txt`. `medium`, seq 1024, fused Adam.

| Batch | No ckpt ms | GB | Ckpt ms | GB |
|---|---|---|---|---|
| 4 | 98.71 | 12.37 | 127.11 | 7.13 |
| 8 | 187.84 | 19.42 | 229.09 | 8.83 |
| 16 | oom | | 479.56 | 12.73 |
| 32 | oom | | 958.77 | 20.79 |

Checkpointing saves 42 to 55% of memory and costs 22 to 29% time. It lets batch 32 fit, but does not raise throughput here: 34,177 tokens/s at best with checkpointing, against 43,612 without.

Spill found during these runs: the first attempt at batch 16 without checkpointing did not error. The Windows GPU driver spilled into system RAM and the step took 4,328 ms at 33.5 GB. All scripts now cap PyTorch at 97% of the card with `torch.cuda.set_per_process_memory_fraction`, so a row that does not fit records `oom`.

### FP8 and NVFP4 training

Stability (`loss_parity_final_sm_120_small_triton_triton-fp8_triton-fp4_200steps.json`, `small`, batch 8, 200 steps, same initial weights):

| | bf16 | FP8 | NVFP4 |
|---|---|---|---|
| steps to loss below 1.0 | 32 | 33 | 36 |
| final loss (step 199) | 0.0005 | 0.0005 | 0.0006 |
| largest loss gap / initial loss | | 0.7% | 6.4% |

The parity metric changed in this milestone. The old one (largest relative difference per step) blows up as the loss nears zero: 0.0006 against 0.0005 counts as 20%. It is now the largest absolute gap divided by the starting loss, plus the steps each run needs to pass loss 1.0. The kernel parity run was repeated with the new metric (`loss_parity_final_sm_120_small_torch_triton_cuda_20steps.json`): 1.4%, PASS.

Profile (`profile_final_sm_120_medium_b4_s1024_bf16_sdpa_fusedadam_triton_fp8.json` and `..._fp4.json`):

| | bf16 | FP8 | NVFP4 |
|---|---|---|---|
| block GEMMs, GPU ms | about 51 | about 21 (2.4x) | about 9 (5.6x) |
| quantization, GPU ms | | about 5 | 18.3 |
| launches per step | 1177 | 1993 | 4033 |
| GPU busy of clean step | 77 of 81 ms | 50 of 73 ms | 56 of 89 ms |
| GPU idle | 5.7% | 31% | 37% |

The NVFP4 block GEMMs are 5.6x faster than bf16 inside the real step, matching the 5.7x from calibration. The step does not get faster because quantization costs twice the FP4 GEMM time and the 4033 launches per step leave the GPU idle 37% of the time. Details and the path to a real FP4 speedup: `optimizations/09_fp8_fp4_linears.md`.

NVFP4 quantizer, alone (`milestones/m5_nvfp4_quantizer_timing.txt`, a (4096, 2816) bf16 tensor): 38 to 57 us per call, 510 to 760 GB/s. The first `torch.compile` version made the `small` step 120.7 ms, 2.4x slower than bf16; the Triton kernel brought it to parity.

## Milestone 6: performance model

Code: `perf_model/roofline.py` (operation inventory, prediction, what if tool), `perf_model/validate.py` (measured against predicted, MFU gap). Console output: `milestones/m6_*.txt`.

### How the model works

`inventory()` lists every GPU op in one training step as `bench_train.py` runs it: the 5 block GEMMs and their 10 backward GEMMs, flash attention forward and backward (or the naive attention ops), norms, RoPE, SwiGLU, residual adds, autocast weight casts, gradient casts, the embedding, the lm head and loss, and the optimizer. Each op has FLOPs, bytes moved, kernel launches and calls per step. There are two inventories: the PyTorch model (ladder row 4) and the fused model (row 6). Grouped query attention and a tensor parallel degree are supported, so it also takes Llama 3 shapes and the sharded shapes the scaling model needs.

Time per op is `max(FLOPs / peak, memory time)`, summed over ops, with two refinements found by the first validation (below):
1. **Launches.** The calibrated 6.5 us per launch is the CPU cost of issuing a kernel from PyTorch eager. The CPU runs ahead of the GPU, so that cost only shows up when the CPU falls behind. On the GPU itself, back to back kernels cost 0.94 us each (measured by replaying 1000 tiny kernels from a CUDA graph). The model now takes `max(GPU time, CPU issue time)`, with GPU time = ops + launches x 0.94 us.
2. **L2.** Multi kernel eager ops (the eager RMSNorm is 7 kernels forward and about 12 backward) pass intermediates from kernel to kernel. If those fit in half of the 96 MB L2, they move at the measured L2 bandwidth of 3428 GB/s, not DRAM's 1441 GB/s. Only the op's real inputs and outputs go to DRAM.

Two prediction levels:
- **Roofline:** calibration peaks only. This is what the hardware would allow with perfect GEMMs.
- **Calibrated:** each distinct GEMM and attention shape timed once on this GPU (`shape_calibration_final.json`, L2 flushed, median). Elementwise ops stay roofline. Shapes too big to time (Llama 3) use the peak times the efficiency measured on the largest timed GEMMs, 89%.

### Validation

Sources: `perf_model_validation_final.json`, `perf_model_validation_final.png`. Steps measured fresh in this session.

| Config | Measured ms | Roofline ms | Error | Calibrated ms | Error | Measured MFU |
|---|---|---|---|---|---|---|
| small b8 s1024, unfused | 74.43 | 57.57 | -22.6% | 69.21 | **-7.0%** | 37.3% |
| small b8 s1024, fused | 49.04 | 38.76 | -21.0% | 50.43 | **+2.8%** | 56.6% |
| medium b4 s1024, unfused | 109.75 | 81.46 | -25.8% | 103.64 | **-5.6%** | 38.5% |
| medium b4 s1024, fused | 81.64 | 62.09 | -23.9% | 84.58 | **+3.6%** | 51.8% |
| large b2 s1024, unfused | 118.98 | 95.43 | -19.8% | 113.90 | **-4.3%** | 42.8% |
| large b2 s1024, fused | 109.71 | 83.03 | -24.3% | 101.50 | **-7.5%** | 46.4% |

**Mean absolute error of the calibrated model: 5.6% unfused, 4.6% fused.** The roofline alone underpredicts by about 22% for every size; that 22% is GEMM inefficiency, which it cannot see.

Held out check (`perf_model_validation_final_holdout.json`): five configurations the model was never compared against while it was being built (tiny b16 s1024, small b4 s2048, medium b8 s512, large b4 s512, medium b2 s2048). Calibrated mean absolute error **6.1% unfused, 2.7% fused**; worst case -10.2% (tiny, unfused).

How good is that? The same config measured in two sessions of this milestone came out 102.94 ms and 109.75 ms (medium, unfused), a 6.6% difference from the GPU's clock state alone. The model's error is about the size of that noise.

What the first version got wrong (`milestones/m6_validate_v1.txt`): it charged the full 6.5 us CPU launch cost to every kernel and sent all eager traffic to DRAM. The calibrated prediction was 28.5% too high for the unfused model. Both fixes above come from that run. The roofline column moved down at the same time, since it shares those fixes.

### Where the time goes (MFU gap)

Source: `mfu_gap_final.png`. Calibrated model, ms per step. "Ideal" is the model FLOPs at the bf16 peak.

| Category | medium unfused | medium fused | large fused |
|---|---|---|---|
| ideal matmul time | 42.3 | 42.3 | 50.9 |
| matmuls below peak | 18.2 | 18.5 | 15.7 |
| attention | 7.2 | 7.2 | 4.9 |
| memory bound elementwise, casts, loss | 31.6 | 13.8 | 14.7 |
| optimizer | 7.3 | 7.3 | 18.3 |
| launch gaps | 2.6 | 1.0 | 0.7 |
| measured total | 109.75 | 81.64 | 109.71 |
| MFU | 38.5% | 51.8% | 46.4% |

1. **Matmuls run at 67 to 75% of the bf16 peak** at these shapes (`matmul_efficiency` in the JSON). cuBLAS picks `cutlass_80` kernels on this GPU (milestone 2), and these GEMMs are small next to an 8192 cube. That costs 16 to 19 ms per step and is the largest loss left in the fused model.
2. **Memory bound ops** are the largest loss in the unfused model (31.6 ms). The Triton kernels cut that to 13.8 ms. What remains is mostly autocast casts and the residual adds.
3. **The optimizer grows with parameters, not with tokens.** For `large` (940M params) fused Adam moves 26 GB per step and takes 18.3 ms, as much as the GEMM inefficiency. Larger batches spread it over more tokens.
4. **Launch overhead is small** once the GPU is busy: the CPU issues 1099 kernels in about 7 ms while the GPU works for 85 ms. It only becomes the bottleneck when GPU work per kernel gets small, as in the NVFP4 row of milestone 5 (37% idle).

### What if

Source: `milestones/m6_what_if_medium.txt`, `perf_model_medium_b4_s1024.json`. `medium`, batch 4, seq 1024.

| Scenario | unfused | fused |
|---|---|---|
| as measured on this GPU (predicted) | 103.64 ms | 84.58 ms |
| memory bandwidth doubled | 84.22 ms (-18.7%) | 74.06 ms (-12.4%) |
| memory bandwidth halved | 142.49 ms (+37.5%) | 105.62 ms (+24.9%) |
| matmul peak doubled | 72.54 ms (-30.0%) | 53.33 ms (-37.0%) |
| no launch cost at all | 101.05 ms (-2.5%) | 83.55 ms (-1.2%) |
| launch cost 3x | 108.84 ms (+5.0%) | 86.64 ms (+2.4%) |

Fusion moves the step from bandwidth sensitive to compute sensitive. Doubling bandwidth helps the unfused model 18.7% but the fused one only 12.4%; doubling the matmul peak helps the fused model more (37.0% against 30.0%).

### Llama 3 8B on one GPU (prediction only)

Source: `milestones/m6_llama3_8b_single_gpu.txt`. Llama 3 8B shape (d 4096, 32 layers, 8 KV heads, ffn 14336, vocab 128256), batch 1, seq 4096, fused model, on this GPU's calibration: **1268 ms per step, 73% MFU**. Matmuls are 72% of the time, the optimizer 12%. This is a model of the compute only. It cannot actually run here: fp32 weights, gradients and Adam state for 8B parameters need about 128 GB. The scaling model in milestone 7 shards it.
