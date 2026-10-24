# Fused Adam

**Change:** `torch.optim.AdamW(fused=True)` in place of the multi tensor (`foreach=True`) version.

**Measured (ladder row 3 to row 4):**

| | multi tensor Adam | fused Adam | Gain |
|---|---|---|---|
| medium, ms/step | 121.81 | 107.50 | 1.13x (14.3 ms saved) |
| small, ms/step | 76.85 | 71.61 | 1.07x |
| optimizer GPU time, medium (profile) | 20.8 ms | 7.3 ms | 2.8x |
| optimizer kernel launches per step | 144 | 20 | |

**Why:** the multi tensor version still runs each Adam step (`lerp` for the first moment, multiply and add for the second, `sqrt`, divide, update) as a separate pass over all parameters, so the optimizer state is read and written several times. The fused kernel reads each parameter, gradient and both moments once, does all the math in registers, and writes them back once. For 375M fp32 parameters that is 16 bytes per parameter read and 12 written, about 10.5 GB per step at minimum, and 7.3 ms is close to that at 1441 GB/s.

**A bug the profiler found:** passing only `fused=False` made this PyTorch build run the slow per parameter loop (one kernel per parameter per op), not the multi tensor version. The benchmark now sets `foreach=True` explicitly for the non fused rows, so row 4 against row 3 is a fair comparison.

Sources: `results/milestones/m5_sweep_final_sm_120_*.json`, `results/profile_final_sm_120_medium_b4_s1024_bf16_sdpa_torch.json`, `results/profile_final_sm_120_medium_b4_s1024_bf16_sdpa_fusedadam_torch.json`.
