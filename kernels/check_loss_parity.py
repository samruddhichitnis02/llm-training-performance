"""Train the same model from the same initial weights with the PyTorch ops and with the
fused kernels, and compare the loss step by step. The fused step must follow the same
training trajectory, not only pass unit tests.

  python kernels/check_loss_parity.py --size small --batch 8 --seq 1024 --steps 20
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_train import RESULTS, SIZES, build, cap_gpu_memory, cleanup, env_info  # noqa: E402


def losses(cfg, run, steps):
    cap_gpu_memory()
    model, opt, step = build(cfg, run)
    out = [float(step().item()) for _ in range(steps)]
    del model, opt, step
    cleanup()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small", choices=list(SIZES))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--modes", default="torch,triton",
                   help="comma list of kernels[+linear], for example torch,triton,triton+fp8,triton+fp4")
    p.add_argument("--tol", type=float, default=0.02, help="max |loss gap| divided by the initial loss")
    p.add_argument("--machine-label", default="dev")
    args = p.parse_args()

    cfg = SIZES[args.size]
    base = dict(size=args.size, batch=args.batch, seq=args.seq, precision="bf16", attn="sdpa",
                fused_adam=True, compile=False, ckpt=False, nvtx=False)
    modes = args.modes.split(",")

    def run_cfg(mode):
        kernels, _, linear = mode.partition("+")
        return dict(base, kernels=kernels, linear=linear or "bf16")

    curves = {m: losses(cfg, run_cfg(m), args.steps) for m in modes}

    ref = curves[modes[0]]
    worst = 0.0
    w = max(12, max(len(m) for m in modes))
    print("step  " + "  ".join(f"{m:>{w}}" for m in modes))
    every = max(1, args.steps // 25)
    for i in range(args.steps):
        if i % every == 0 or i == args.steps - 1:
            print(f"{i:>4}  " + "  ".join(f"{curves[m][i]:{w}.4f}" for m in modes))
    # Metric: largest absolute gap between the curves, divided by the starting loss. A plain
    # relative difference blows up once the loss nears zero (0.0006 against 0.0005 reads as 20%),
    # which says nothing about training quality. Also report the steps each mode needs to get
    # below a loss threshold, which shows a slowdown in learning directly.
    def steps_below(curve, thr):
        return next((i for i, v in enumerate(curve) if v < thr), None)

    per_mode = {}
    for m in modes[1:]:
        d = max(abs(a - b) for a, b in zip(curves[m], ref)) / abs(ref[0])
        per_mode[m] = {"max_abs_diff_over_initial": d, "final_loss": curves[m][-1],
                       "steps_to_loss_below_1": steps_below(curves[m], 1.0)}
        worst = max(worst, d)
        print(f"{m:>{w}}: max |loss gap| / initial loss {d:.2e}, steps to loss < 1.0: "
              f"{steps_below(curves[m], 1.0)} (vs {steps_below(ref, 1.0)}), final loss {curves[m][-1]:.4f} "
              f"against {ref[-1]:.4f}")
    ok = worst <= args.tol
    print(f"\nworst: {worst:.2e} (tol {args.tol}) -> {'PASS' if ok else 'FAIL'}")

    info = env_info(args.machine_label)
    out = {**info, "size": args.size, "batch": args.batch, "seq": args.seq, "steps": args.steps, "curves": curves,
           "reference_steps_to_loss_below_1": steps_below(ref, 1.0), "per_mode": per_mode,
           "metric": "max |loss gap| / initial loss", "worst": worst, "tol": args.tol, "pass": ok}
    RESULTS.mkdir(exist_ok=True)
    tag = "_".join(m.replace("+", "-") for m in modes)
    path = RESULTS / f"loss_parity_{args.machine_label}_{info['arch']}_{args.size}_{tag}_{args.steps}steps.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path.name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
