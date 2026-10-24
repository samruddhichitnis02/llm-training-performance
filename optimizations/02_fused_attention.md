# Fused attention (SDPA)

**Change:** replace the naive attention (`q @ k^T`, mask, softmax, `@ v`, with the full T by T score matrix in memory) with `F.scaled_dot_product_attention(is_causal=True)`, which runs a flash attention kernel.

**Measured (ladder row 2 to row 3):**

| | naive | SDPA | Gain |
|---|---|---|---|
| medium, ms/step | 236.37 | 121.81 | 1.94x |
| small, ms/step | 164.34 | 76.85 | 2.14x |
| peak memory, medium | 21.62 GB | 12.36 GB | 9.3 GB less |
| peak memory, small | 16.06 GB | 9.10 GB | 7.0 GB less |

This is the largest single gain in the ladder.

**Why:** at seq 1024 with 16 heads and batch 4, the score matrix is 4 x 16 x 1024 x 1024 = 67M values per layer, kept for the backward pass in several forms (scores, the fp32 softmax input, probabilities). Flash attention computes the softmax in tiles held on chip and never writes the matrix to memory, then recomputes what it needs in the backward pass. The attention FLOPs are the same; the memory traffic for those matrices disappears.

After this change, attention is only 5.7% of GPU time in the profile (`results/profile_final_sm_120_medium_b4_s1024_bf16_sdpa_torch.json`).

Sources: `results/milestones/m5_sweep_final_sm_120_*.json`.
