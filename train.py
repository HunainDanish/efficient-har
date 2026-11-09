"""Train a hybrid CNN + temporal model for action recognition.

Examples
--------
Smoke test (no download, runs on CPU in a few minutes):
    python train.py --dataset synthetic --frames 8 --img-size 64 --epochs 3 --no-pretrained

UCF101, 10-class subset, official split 1 (Kaggle T4):
    python train.py --dataset ucf101 --data-root /path/UCF-101 \
        --split-dir /path/ucfTrainTestlist --max-classes 10 --epochs 15
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from har.data import build_datasets
from har.metrics import classification_report, count_params, predict
from har.model import BACKBONES, TEMPORAL, build_model


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--dataset", choices=["ucf101", "folder", "synthetic"], default="synthetic")
    d.add_argument("--data-root", help="folder with the videos (class sub-folders)")
    d.add_argument("--split-dir", help="UCF101: folder with classInd.txt / trainlist01.txt / testlist01.txt")
    d.add_argument("--split", type=int, default=1, help="UCF101 official split (1-3)")
    d.add_argument("--max-classes", type=int, help="use only the first N classes (quick experiments)")
    d.add_argument("--val-ratio", type=float, default=0.2, help="folder datasets: validation share")
    d.add_argument("--frames", type=int, default=16, help="frames sampled per clip")
    d.add_argument("--img-size", type=int, default=112)
    m = p.add_argument_group("model")
    m.add_argument("--backbone", choices=sorted(BACKBONES), default="mobilenet_v3_small")
    m.add_argument("--temporal", choices=TEMPORAL, default="transformer")
    m.add_argument("--dim", type=int, default=256)
    m.add_argument("--layers", type=int, default=2)
    m.add_argument("--heads", type=int, default=4)
    m.add_argument("--dropout", type=float, default=0.1)
    m.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   help="do not load ImageNet weights (offline use)")
    o = p.add_argument_group("optimisation")
    o.add_argument("--epochs", type=int, default=15)
    o.add_argument("--batch-size", type=int, default=16)
    o.add_argument("--lr", type=float, default=3e-4)
    o.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="learning-rate multiplier for the pretrained CNN")
    o.add_argument("--weight-decay", type=float, default=0.05)
    o.add_argument("--label-smoothing", type=float, default=0.1)
    o.add_argument("--workers", type=int, default=2)
    o.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/exp", help="output folder for checkpoint and logs")
    return p.parse_args(argv)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None):
    args = get_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_set, val_set, classes = build_datasets(
        args.dataset, args.data_root, args.split_dir, args.split, args.max_classes,
        args.frames, args.img_size, args.val_ratio, args.seed)
    print(f"{len(classes)} classes | {len(train_set)} train / {len(val_set)} val clips | device {device}")
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)

    cfg = {k: getattr(args, k) for k in
           ("backbone", "temporal", "dim", "layers", "heads", "dropout", "pretrained", "frames", "img_size")}
    model = build_model(cfg, len(classes)).to(device)
    print(f"model: {args.backbone} + {args.temporal} | {count_params(model) / 1e6:.2f}M params")

    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb_lr = args.lr * (args.backbone_lr_mult if args.pretrained else 1.0)
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": bb_lr},
                             {"params": other_params, "lr": args.lr}], weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    warmup = min(len(train_loader), total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(clips), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * labels.shape[0]
            seen += labels.shape[0]

        logits, labels = predict(model, val_loader, device)
        rep = classification_report(logits, labels, classes)
        row = {"epoch": epoch, "train_loss": run_loss / max(seen, 1), "val_top1": rep["top1"],
               "val_top5": rep["top5"], "val_macro_f1": rep["macro_f1"], "seconds": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} | top1 {row['val_top1']:.4f} | "
              f"top5 {row['val_top5']:.4f} | F1 {row['val_macro_f1']:.4f} | {row['seconds']:.0f}s")
        if rep["top1"] > best:
            best = rep["top1"]
            torch.save({"model": model.state_dict(), "config": cfg, "classes": classes,
                        "epoch": epoch, "val_top1": best}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"best val top-1: {best:.4f} -> {out / 'best.pt'}")
    return best


if __name__ == "__main__":
    main()
