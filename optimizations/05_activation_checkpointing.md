# Activation checkpointing

**Change:** `--ckpt` wraps each transformer block in `torch.utils.checkpoint` (non reentrant). Only the block inputs are kept from the forward pass; each block's forward is recomputed during the backward pass. Recompute FLOPs are not counted in MFU, so MFU shows the real cost.

**Measured** (`medium`, seq 1024, bf16, SDPA, fused Adam, PyTorch ops; memory capped at 97% of the card so a row that does not fit is recorded as oom):

| Batch | No checkpointing | | With checkpointing | | Memory saved | Time cost |
|---|---|---|---|---|---|---|
| | ms/step | peak GB | ms/step | peak GB | | |
| 4 | 98.71 | 12.37 | 127.11 | 7.13 | 42% | +29% |
| 8 | 187.84 | 19.42 | 229.09 | 8.83 | 55% | +22% |
| 16 | oom | | 479.56 | 12.73 | fits only with ckpt | |
| 32 | oom | | 958.77 | 20.79 | fits only with ckpt | |

Throughput: the best run without checkpointing is batch 8 at 43,612 tokens/s. With checkpointing it levels off at about 34,170 tokens/s from batch 16 up.

**What it buys and what it costs:**
- Memory: each extra sequence in the batch costs about 1.76 GB without checkpointing (12.37 GB at batch 4, 19.42 GB at batch 8) and about 0.49 GB with it (8.83, 12.73, 20.79 GB at batch 8, 16, 32). Batch 16 without checkpointing works out to about 33.5 GB, which is what the spilled run showed, so it cannot fit on a 32 GB card.
- Time: checkpointing costs 22 to 29% more per step. The forward pass is about a third of the model FLOPs, and the optimizer step and the lm head are not recomputed, so the extra work is less than a full third.
- On this model and card it does **not** raise throughput. The GPU is already saturated at batch 8, so the larger batches that checkpointing allows still pay the recompute and stay about 22% below the best run. It is the right tool when a model or sequence length does not fit at all, not a speedup here.

**Earlier mistake this fixed:** before the memory cap, batch 16 without checkpointing did not raise an out of memory error. The Windows GPU driver spilled 1.5 GB into system RAM and the step took 4,328 ms instead of failing. `bench_train.py` now calls `torch.cuda.set_per_process_memory_fraction(0.97)` so these rows record `oom`.

Sources: `results/milestones/m5_ckpt_medium_*.json`, `results/milestones/m5_ckpt_medium.txt`.
