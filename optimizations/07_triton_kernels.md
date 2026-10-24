# Triton kernels, one by one

**Change:** five fused ops in `kernels/triton_ops.py`, each with forward and backward: `rmsnorm`, `add_rmsnorm` (residual add plus norm), `swiglu`, `rope_qkv` (RoPE on the packed qkv output, writing q, k, v in attention layout), and the chunked linear plus cross entropy (note 06).

**Measured in the training step** (`medium`, batch 4, seq 1024, bf16, SDPA, fused Adam, no compile). "none" runs every op in PyTorch inside the same model structure; each row switches on exactly one kernel:

| Config | ms/step | Saved | Peak GB | GB saved |
|---|---|---|---|---|
| none (all ops in PyTorch) | 102.68 | | 12.37 | |
| only rmsnorm | 97.81 | 4.87 ms (4.7%) | 11.95 | 0.42 |
| only add_rmsnorm | 97.46 | 5.22 ms (5.1%) | 11.97 | 0.40 |
| only swiglu | 101.71 | 0.97 ms (0.9%) | 11.81 | 0.56 |
| only rope_qkv | 94.77 | 7.91 ms (7.7%) | 12.17 | 0.20 |
| only linear + cross entropy | 100.20 | 2.48 ms (2.4%) | 11.20 | 1.17 |
| all five | 80.51 | 22.17 ms (21.6%) | 9.63 | 2.74 |

The single kernel savings add up to 21.45 ms, and all five together save 22.17 ms, so the gains are independent of each other.

**Measured as ops on their own** (`small`, 8192 tokens, forward plus backward, L2 cleared before each call, bandwidth against the measured 1441 GB/s):

| Op | eager | torch.compile | Triton | Triton vs eager | Triton vs compile |
|---|---|---|---|---|---|
| rmsnorm | 0.328 ms | 0.179 ms | 0.072 ms (97% BW) | 4.52x | 2.48x |
| add_rmsnorm | 0.383 ms | 0.175 ms | 0.123 ms (100% BW) | 3.13x | 1.43x |
| swiglu | 0.241 ms | 0.176 ms | 0.176 ms | 1.37x | 1.00x |
| rope_qkv | 0.570 ms | 0.248 ms | 0.109 ms (97% BW) | 5.25x | 2.28x |

**Why each one gains what it does:**
- **RoPE gains most.** Eager splits the packed qkv with a view and permute, rotates q and k each with a `cat` of negated halves, multiplies in fp32, and casts back. That is about ten passes over the q and k data. The kernel reads each value once and writes q, k and v straight into the `(B, H, T, HD)` layout SDPA wants.
- **The two norms** replace a float cast, a square, a mean, an `rsqrt`, two multiplies and a cast (plus the residual add) with one read and one write per value. The backward reduces the weight gradient in registers across a persistent grid instead of materializing `dy * s * rstd`.
- **SwiGLU gains least.** It is only three tensors in and one out, and eager's intermediates (33 MB) stay in this GPU's 96 MB L2 cache, so the extra eager passes are mostly L2 hits.

Sources: `results/ablation_kernels_final_sm_120_medium_b4_s1024.json`, `results/bench_kernels_final_sm_120_small_b8_s1024.json`.
