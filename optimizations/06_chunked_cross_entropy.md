# Chunked linear plus cross entropy

**Change:** compute the lm head matmul and the cross entropy loss 2048 rows at a time. Each chunk's logits are turned into their gradient in place by one Triton kernel (row max, log sum exp, loss, then `(softmax - onehot) / N`), and `dh` and `dW` are accumulated right away. The full `(tokens, 32000)` logits never exist. Code: `fused_linear_cross_entropy` in `kernels/triton_ops.py`.

**Measured, op on its own (`small`, 8192 tokens, forward plus backward):**

| | eager | torch.compile | chunked Triton |
|---|---|---|---|
| time | 12.85 ms | 7.20 ms | 6.89 ms |
| extra memory during the op | 3.08 GB | 0.52 GB | 0.31 GB |
| achieved TFLOPs (lm head GEMMs) | 94 | 168 | 175 |

Against eager: 1.87x faster, **10x less memory**.

**Measured in the training step (`medium`, ablation with only this kernel switched on):** 2.48 ms saved per step (2.4%) and **1.17 GB lower peak memory**, the largest memory saving of the five kernels.

**Why the memory drops:** eager PyTorch materializes the bf16 logits (8192 x 32000 x 2 bytes = 0.52 GB), their fp32 copy for the loss (1.05 GB), the log softmax output, and the gradient of the logits, all at full size. The chunked version only ever holds one 2048 row chunk of logits (131 MB in bf16), and reuses it for the gradient. The saving grows with tokens per step and with vocabulary size; at Llama 3's 128K vocabulary it would be four times larger.

**Why it is also faster:** the loss and softmax gradient become one pass over each chunk, and the chunk is still in L2 when the two GEMMs that follow it read it.

Sources: `results/bench_kernels_final_sm_120_small_b8_s1024.json`, `results/ablation_kernels_final_sm_120_medium_b4_s1024.json`.
