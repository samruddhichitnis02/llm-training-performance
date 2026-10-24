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

### First optimization ladder (rows 1 to 5)

Source: `milestones/m1_sweep_final_sm_120_small_b8_s1024.json`. Model `small` (134M params), batch 8, seq 1024, 5 warmup steps, median of 20 timed steps. MFU is against the bf16 matmul peak reached in the same run, 228.9 TFLOPs.

| Row | ms/step | tokens/s | Speedup | TFLOPs | MFU | Peak GB |
|---|---|---|---|---|---|---|
| 1 fp32 baseline, naive attention | 223.34 | 36,679 | 1.00 | 28.26 | 12.3% | 15.94 |
| 2 bf16 autocast | 156.33 | 52,401 | 1.43 | 40.37 | 17.6% | 16.06 |
| 3 + fused attention (SDPA) | 77.87 | 105,207 | 2.87 | 81.05 | 35.4% | 9.10 |
| 4 + fused Adam | 68.82 | 119,028 | 3.25 | 91.70 | 40.1% | 9.10 |
| 5 + torch.compile | 49.97 | 163,950 | 4.47 | 126.31 | 55.2% | 5.48 |
| 6 fused Triton kernels | not built yet | | | | | |
| 7 CUDA C++ add+RMSNorm | not built yet | | | | | |

All losses finite. Loss falls to about 2.5 because the model sees the same random batch every step; that is expected and does not affect step time.

What stands out: fused attention gives the biggest single jump (2x on its own, and 7 GB less memory) because naive attention builds a 1024 by 1024 score matrix per head. bf16 alone gains only 1.43x even though bf16 matmuls are 3.5x faster than fp32, which means a large part of the step is not matmul. The profiler in milestone 2 will show exactly what that part is.
