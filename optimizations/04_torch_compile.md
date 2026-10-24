# torch.compile

**Change:** wrap the model in `torch.compile` (default mode). The optimizer step is not compiled.

**Measured (ladder row 4 to row 5):**

| | eager | torch.compile | Gain |
|---|---|---|---|
| medium, ms/step | 107.50 | 83.51 | 1.29x |
| small, ms/step | 71.61 | 50.23 | 1.43x |
| peak memory, medium | 12.36 GB | 9.74 GB | 2.6 GB less |
| kernel launches per step, medium (profile) | 2921 | 1066 | 64% fewer |
| elementwise GPU time, medium (profile) | 16.1 ms | 1.8 ms | |

**Why:** Inductor fuses chains of elementwise ops (the norms, the residual adds, SwiGLU, RoPE and the dtype casts) into generated Triton kernels. Each chain reads its inputs once and writes once, instead of one round trip to memory per op. Fewer intermediates also have to be saved for the backward pass, which is where the 2.6 GB comes from.

**Compared with the hand written kernels:** row 6 (four fused Triton ops, no compiler) reaches the same step time within 1.5%: 84.70 against 83.51 ms. So the hand written kernels recover what the compiler finds, on the ops the profile said matter.

**Cost:** the first step compiles the graph, which takes tens of seconds. The benchmark's warmup steps absorb it.

Sources: `results/milestones/m5_sweep_final_sm_120_*.json`, `results/profile_final_sm_120_medium_b4_s1024_bf16_sdpa_fusedadam_compile_torch.json`.
