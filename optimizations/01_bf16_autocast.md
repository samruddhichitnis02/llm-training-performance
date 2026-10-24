# bf16 autocast

**Change:** run the matmuls in bf16 under `torch.autocast`. Weights and the optimizer stay fp32. The baseline is true fp32 with TF32 switched off.

**Measured (ladder row 1 to row 2):**

| | fp32 | bf16 autocast | Gain |
|---|---|---|---|
| medium, ms/step | 333.97 | 236.37 | 1.41x |
| small, ms/step | 233.85 | 164.34 | 1.42x |
| peak memory, medium | 21.16 GB | 21.62 GB | +0.46 GB |

**Why the gain is only 1.4x:** calibration shows bf16 matmuls run 3.5x faster than fp32 on this GPU (227.5 against 65.3 TFLOPs). The step only gets 1.4x because, in row 2, most of the time is not in the linear layers. Naive attention builds a T by T score matrix in fp32 for the softmax, and many small elementwise kernels and the optimizer don't get faster with bf16 at all.

**Why memory goes up slightly:** autocast keeps a bf16 copy of every weight for the step on top of the fp32 master weights. The activations it saves are smaller, but at this size the extra weight copies and the fp32 softmax inputs cancel that out.

Sources: `results/milestones/m5_sweep_final_sm_120_*.json`, `results/calibration_final.json`.
