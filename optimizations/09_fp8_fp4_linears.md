# FP8 and NVFP4 linear layers

**Change:** ladder rows 8 and 9 take row 6 (Triton kernels, fused Adam, bf16 autocast) and run all three GEMMs of every block linear (qkv, wo, w1, w2, w3: forward, input gradient, weight gradient) in a low precision format. Code: `kernels/lowp.py`, tests in `kernels/test_lowp.py`. The lm head stays bf16. Weights and Adam stay fp32.

- **FP8:** e4m3 for activations, weights and gradients, with one scale per tensor computed every step from its absolute max. Quantization is a `torch.compile` kernel. Built on `torch._scaled_mm`.
- **NVFP4:** e2m1 values (0, 0.5, 1, 1.5, 2, 3, 4, 6) packed two per byte, one e4m3 scale per 16 values along the contraction dimension, plus one fp32 scale per tensor. Each GEMM gets operands blocked along its own contraction axis, so `x`, `w` and `dy` are each quantized twice (normal and transposed). Gradients use stochastic rounding. Quantization is a hand written Triton kernel that reads through strides (no transpose copy), rounds, packs, and writes the block scales directly in the tiled layout the Blackwell block scaled GEMM reads. Not included from NVIDIA's NVFP4 recipe: random Hadamard transforms and 2D weight blocks.

Transformer Engine was not used. It needs a source build against this exact PyTorch, and whether its NVFP4 path supports `sm_120` was not checked. Writing the two paths directly also exposed the costs below.

## Correctness

`kernels/test_lowp.py`: 22 of 22 checks pass.
- The NVFP4 GEMM (`cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1`) matches a float32 matmul of the dequantized operands to 3e-7, which confirms packing, nibble order and the scale layout.
- The Triton quantizer produces the same bits as the PyTorch reference, for fp32 and bf16, normal and transposed.
- Stochastic rounding is unbiased: the mean of 400 draws is within 1.2% of the input.
- Linear layer error against bf16 is the format's own: 3.7% for FP8 and 13 to 17% for NVFP4.

The tests caught two bugs while building this: a quantizer that paired values with `reshape` plus `split` gave wrong bits for contiguous fp32 input, and an atomic float max ran as a slow compare and swap loop.

## Training stability

`small`, batch 8, seq 1024, 200 steps from the same weights (`results/loss_parity_final_sm_120_small_triton_triton-fp8_triton-fp4_200steps.json`):

| | bf16 | FP8 | NVFP4 |
|---|---|---|---|
| loss at step 24 | 2.52 | 2.59 | 3.21 |
| steps to loss below 1.0 | 32 | 33 | 36 |
| loss at step 199 | 0.0005 | 0.0005 | 0.0006 |
| largest loss gap / initial loss | | 0.7% | 6.4% |

All three train stably and reach the same final loss. FP8 tracks bf16 almost exactly. NVFP4 learns about 12% slower early on (36 steps instead of 32 to pass loss 1.0), which is the known cost of 4 bit gradients. This is a memorization test on one fixed batch, not a full pretraining run; it shows the step is numerically sound, not that NVFP4 matches bf16 over a real training run.

## Speed

Ladder (`results/milestones/m5_sweep_final_sm_120_*.json`):

| | row 6 bf16 | row 8 FP8 | row 9 NVFP4 |
|---|---|---|---|
| medium, ms/step | 84.70 | 71.40 (1.19x) | 84.66 (1.00x) |
| small, ms/step | 50.55 | 44.67 (1.13x) | 50.76 (1.00x) |
| medium peak memory | 9.62 GB | 8.85 GB | 8.26 GB |
| medium MFU (against bf16 peak) | 52.7% | 62.5% | 52.7% |

MFU here uses the bf16 peak for every row, as the spec defines it, so a faster GEMM format can push it up.

Profile of one step on `medium` (`results/profile_final_sm_120_medium_b4_s1024_bf16_sdpa_fusedadam_triton_*.json`):

| | bf16 (row 6) | FP8 | NVFP4 |
|---|---|---|---|
| all matmuls, GPU ms | 55.3 | 25.4 | 13.1 |
| block GEMMs only (matmuls minus the bf16 lm head, about 4 ms) | about 51 | about 21 (2.4x) | about 9 (5.6x) |
| quantization, GPU ms | | about 5 | 18.3 (16.2 quantize + 2.1 amax) |
| kernel launches per step | 1177 | 1993 | 4033 |
| GPU busy / clean step time | 77 / 81 ms | 50 / 73 ms | 56 / 89 ms |
| GPU idle | 5.7% | 31% | 37% |

**What this shows:**
1. **The NVFP4 tensor cores deliver.** The block GEMMs run 5.6x faster than bf16, the same ratio the calibration measured on a large matmul (1291.7 against 227.5 TFLOPs, 5.7x). FP8 gives 2.4x where calibration says 2.0x; the gain beyond 2x comes from smaller operands moving less memory.
2. **The GPU work for NVFP4 is smaller than for bf16** (56 against 77 ms busy), but the step is not faster. Quantizing six tensors per linear costs 18 ms, twice the FP4 GEMM time itself, and at 4033 launches per step the CPU can no longer queue work fast enough: the GPU sits idle 37% of the step.
3. **FP8 is the practical win at this model size** (19% faster step, 0.8 GB less memory). Its quantization is cheaper (one scale per tensor, and the quantized tensors can be transposed and reused).

**What would make NVFP4 pay off:** fuse quantization into the producing kernels (the norm and SwiGLU kernels could write NVFP4 directly), quantize each weight once per step instead of once per GEMM, and capture the step in a CUDA graph to remove the launch bound. Larger models help on their own: GEMM time grows with d squared while quantization grows with d, so at Llama 8B widths the 5.6x GEMM gain dominates.

## The first version

The first NVFP4 version used `torch.compile` for quantization and ran at 120.7 ms per step on `small`, 2.4x slower than bf16. The profile showed 78 ms of the 121 ms in quantization and transpose copies, with the FP4 GEMMs under 5 ms. The Triton quantizer brought it to parity.

Sources: files named above, plus `results/milestones/m5_nvfp4_quantizer_timing.txt`, `results/milestones/m5_profile_medium_fp4.txt`, `results/milestones/m5_profile_medium_fp8.txt`.
