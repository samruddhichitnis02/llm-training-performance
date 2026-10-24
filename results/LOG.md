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
