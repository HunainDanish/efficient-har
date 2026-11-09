"""Evaluation metrics and efficiency profiling."""
import statistics
import time

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader -> (logits, labels) on CPU."""
    model.eval()
    logits, labels = [], []
    for clips, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(clips.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    k = min(k, logits.shape[1])
    hits = logits.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    idx = labels * num_classes + preds
    return torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)


def classification_report(logits: torch.Tensor, labels: torch.Tensor, classes: list[str]) -> dict:
    preds = logits.argmax(dim=1)
    cm = confusion_matrix(preds, labels, len(classes)).double()
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    present = cm.sum(1) > 0  # macro-average over classes that appear in the labels
    return {
        "top1": topk_accuracy(logits, labels, 1),
        "top5": topk_accuracy(logits, labels, 5),
        "macro_f1": f1[present].mean().item(),
        "per_class_recall": {c: recall[i].item() for i, c in enumerate(classes) if present[i]},
        "confusion_matrix": cm.long().tolist(),
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_gflops(model: nn.Module, clip_shape: tuple[int, ...]) -> float:
    """GFLOPs for ONE clip, reported as multiply-accumulates (MACs).

    Most video papers (TSM, X3D, MoViNet) quote MACs under the name "FLOPs",
    so we divide PyTorch's FLOP count by 2 to stay comparable.
    """
    model.eval()
    counter = FlopCounterMode(display=False)
    with counter:
        model(torch.zeros(1, *clip_shape))
    return counter.get_total_flops() / 2 / 1e9


@torch.no_grad()
def cpu_latency_ms(model: nn.Module, clip_shape: tuple[int, ...], warmup: int = 3,
                   runs: int = 10, threads: int | None = None) -> float:
    """Median single-clip latency on CPU in milliseconds."""
    if threads:
        torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = torch.randn(1, *clip_shape)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)
