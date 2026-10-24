# CUDA C++ add plus RMSNorm

**Change:** the same fused add plus RMSNorm as the Triton kernel, written in CUDA C++ (`kernels/cuda_ext/add_rmsnorm.cu`) and loaded with `torch.utils.cpp_extension`. One block per row, 4 wide vectorized loads, warp shuffle reductions, shared memory for the cross warp step, and a persistent backward grid that keeps the weight gradient in registers.

**Measured:**

| | PyTorch eager | torch.compile | Triton | CUDA C++ |
|---|---|---|---|---|
| op alone, fwd + bwd (`small`, 8192 x 768) | 0.383 ms | 0.175 ms | 0.1225 ms | 0.1224 ms |
| share of bandwidth | 32% | 70% | 100% | 100% |
| ladder, medium, ms/step (rows 6 and 7) | | | 84.70 | 84.58 |
| inside the step, 24 calls, from the profile | | | about 1.03 ms | 1.29 ms |

**Result:** no gain over Triton. Both reach the memory bandwidth limit, so a faster kernel would have to move fewer bytes, and both already move the minimum. In isolation they tie to four digits; inside the real step the Triton one is slightly faster (about 0.26 ms per step, within run to run noise of the step time).

**What it took:** the first version used up to 1024 threads per block and failed to launch for D = 8192 (`cudaErrorLaunchOutOfResources`, since 1024 threads leave only 64 registers each). The test suite caught it. Blocks are now capped at 512 threads with `__launch_bounds__(512)`.

**Takeaway:** for a memory bound op, Triton reaches the same bandwidth as hand tuned CUDA with less code. CUDA C++ is worth it when the kernel needs control Triton doesn't expose, which this op doesn't.

Sources: `results/bench_kernels_final_sm_120_small_b8_s1024.json`, `results/milestones/m5_sweep_final_sm_120_medium_b4_s1024.json`, `results/profile_final_sm_120_medium_b4_s1024_bf16_sdpa_fusedadam_cuda.json`.
