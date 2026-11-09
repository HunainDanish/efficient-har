"""Compare the computational cost of model variants: params, GFLOPs, CPU latency.

Needs no data or trained weights: cost depends only on the architecture and input size.

    python benchmark.py                               # default comparison table
    python benchmark.py --frames 8 16 --temporal transformer avg
"""
import argparse
import csv
import itertools
from pathlib import Path

import torch

from har.metrics import count_gflops, count_params, cpu_latency_ms
from har.model import BACKBONES, TEMPORAL, HybridHAR


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbones", nargs="+", choices=sorted(BACKBONES),
                   default=["mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--temporal", nargs="+", choices=TEMPORAL, default=["avg", "lstm", "transformer"])
    p.add_argument("--frames", nargs="+", type=int, default=[8, 16])
    p.add_argument("--img-size", type=int, default=112)
    p.add_argument("--num-classes", type=int, default=101)
    p.add_argument("--runs", type=int, default=10, help="timed CPU runs per model (median reported)")
    p.add_argument("--threads", type=int, help="CPU threads (default: PyTorch default)")
    p.add_argument("--out", default="runs/benchmark.csv")
    args = p.parse_args(argv)

    rows = []
    header = f"{'backbone':20s} {'temporal':12s} {'frames':>6s} {'params(M)':>10s} {'GFLOPs':>8s} {'CPU ms':>8s} {'clips/s':>8s}"
    print(header)
    print("-" * len(header))
    for bb, tm, t in itertools.product(args.backbones, args.temporal, args.frames):
        torch.manual_seed(0)
        model = HybridHAR(args.num_classes, backbone=bb, temporal=tm, pretrained=False).eval()
        shape = (t, 3, args.img_size, args.img_size)
        row = {
            "backbone": bb, "temporal": tm, "frames": t, "img_size": args.img_size,
            "params_m": round(count_params(model) / 1e6, 3),
            "gflops": round(count_gflops(model, shape), 3),
            "cpu_ms": round(cpu_latency_ms(model, shape, runs=args.runs, threads=args.threads), 1),
        }
        row["clips_per_s"] = round(1000 / row["cpu_ms"], 2)
        rows.append(row)
        print(f"{bb:20s} {tm:12s} {t:6d} {row['params_m']:10.2f} {row['gflops']:8.2f} "
              f"{row['cpu_ms']:8.1f} {row['clips_per_s']:8.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}  (GFLOPs = multiply-accumulates per clip, the convention in video papers)")


if __name__ == "__main__":
    main()
