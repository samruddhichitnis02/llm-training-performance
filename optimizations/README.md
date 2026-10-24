# Optimizations

One note per optimization, each with the gain measured on `sm_120` (label `final`). Unless a note says otherwise, the model is `medium` (375M params) at batch 4, seq 1024. Every number comes from a file in `results/`, named in the note.

| # | Optimization | Note |
|---|---|---|
| 1 | bf16 autocast | [01_bf16_autocast.md](01_bf16_autocast.md) |
| 2 | fused attention (SDPA) | [02_fused_attention.md](02_fused_attention.md) |
| 3 | fused Adam | [03_fused_adam.md](03_fused_adam.md) |
| 4 | torch.compile | [04_torch_compile.md](04_torch_compile.md) |
| 5 | activation checkpointing | [05_activation_checkpointing.md](05_activation_checkpointing.md) |
| 6 | chunked linear plus cross entropy | [06_chunked_cross_entropy.md](06_chunked_cross_entropy.md) |
| 7 | Triton kernels, one by one | [07_triton_kernels.md](07_triton_kernels.md) |
| 8 | CUDA C++ add plus RMSNorm | [08_cuda_add_rmsnorm.md](08_cuda_add_rmsnorm.md) |
| 9 | FP8 and NVFP4 linear layers | [09_fp8_fp4_linears.md](09_fp8_fp4_linears.md) |

Ladder source for the step times: `results/milestones/m5_sweep_final_sm_120_medium_b4_s1024.json` (all 9 rows in one run, so they compare directly). This run came out about 4% slower in every row than the milestone 4 run, and its bf16 matmul reference was lower too (215.7 against 226.8 TFLOPs), which points to the GPU clocking lower during this session. The gains between rows are the same in both runs.
