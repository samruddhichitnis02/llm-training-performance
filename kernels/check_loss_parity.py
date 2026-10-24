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
from bench_train import RESULTS, SIZES, build, cleanup, env_info  # noqa: E402


def losses(cfg, run, steps):
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
    p.add_argument("--modes", default="torch,triton")
    p.add_argument("--tol", type=float, default=0.02, help="max relative loss difference")
    p.add_argument("--machine-label", default="dev")
    args = p.parse_args()

    cfg = SIZES[args.size]
    base = dict(size=args.size, batch=args.batch, seq=args.seq, precision="bf16", attn="sdpa",
                fused_adam=True, compile=False, ckpt=False, nvtx=False)
    modes = args.modes.split(",")
    curves = {m: losses(cfg, dict(base, kernels=m), args.steps) for m in modes}

    ref = curves[modes[0]]
    worst = 0.0
    print("step  " + "  ".join(f"{m:>10}" for m in modes))
    for i in range(args.steps):
        print(f"{i:>4}  " + "  ".join(f"{curves[m][i]:10.4f}" for m in modes))
    for m in modes[1:]:
        worst = max(worst, max(abs(a - b) / abs(b) for a, b in zip(curves[m], ref)))
    ok = worst <= args.tol
    print(f"\nmax relative loss difference vs {modes[0]}: {worst:.2e} (tol {args.tol}) -> {'PASS' if ok else 'FAIL'}")

    info = env_info(args.machine_label)
    out = {**info, "size": args.size, "batch": args.batch, "seq": args.seq, "curves": curves,
           "max_rel_diff": worst, "pass": ok}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"loss_parity_{args.machine_label}_{info['arch']}_{args.size}_{'_'.join(modes)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path.name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
