"""Evaluate a trained checkpoint: top-1/top-5, macro F1, per-class recall, confusion matrix.

    python evaluate.py --ckpt runs/exp/best.pt --dataset ucf101 \
        --data-root /path/UCF-101 --split-dir /path/ucfTrainTestlist --max-classes 10
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from har.data import build_datasets
from har.metrics import classification_report, predict
from har.model import build_model


def save_confusion_png(cm: list[list[int]], classes: list[str], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_t = torch.tensor(cm, dtype=torch.float)
    norm = cm_t / cm_t.sum(1, keepdim=True).clamp(min=1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.45), max(4, n * 0.4)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", choices=["ucf101", "folder", "synthetic"], default="synthetic")
    p.add_argument("--data-root")
    p.add_argument("--split-dir")
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--max-classes", type=int)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, classes = ckpt["config"], ckpt["classes"]
    _, val_set, data_classes = build_datasets(
        args.dataset, args.data_root, args.split_dir, args.split, args.max_classes,
        cfg["frames"], cfg["img_size"], args.val_ratio)
    if list(data_classes) != list(classes):
        raise SystemExit("class list of the data does not match the checkpoint (check --max-classes)")

    model = build_model(cfg, len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    logits, labels = predict(model, loader, device)
    rep = classification_report(logits, labels, classes)

    out = Path(args.ckpt).parent
    (out / "metrics.json").write_text(json.dumps(rep, indent=2))
    try:
        save_confusion_png(rep["confusion_matrix"], classes, out / "confusion_matrix.png")
    except ImportError:
        print("matplotlib not installed: skipped confusion_matrix.png")
    print(f"top1 {rep['top1']:.4f} | top5 {rep['top5']:.4f} | macro F1 {rep['macro_f1']:.4f}")
    print(f"saved {out / 'metrics.json'}")
    return rep


if __name__ == "__main__":
    main()
