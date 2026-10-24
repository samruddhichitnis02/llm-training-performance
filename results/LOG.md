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
