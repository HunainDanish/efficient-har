"""Fast checks that run on CPU without downloads: python -m pytest -q"""
import random

import numpy as np
import torch

from har.data import SyntheticMotionDataset, sample_indices, to_clip_tensor
from har.metrics import classification_report, count_gflops
from har.model import BACKBONES, TEMPORAL, HybridHAR


def test_sample_indices_cover_the_video():
    idx = sample_indices(100, 8, train=False, rng=random.Random(0))
    assert len(idx) == 8 and idx == sorted(idx) and idx[0] < 13 and idx[-1] > 87
    assert sample_indices(3, 8, train=True, rng=random.Random(0))[-1] <= 2  # short videos repeat frames


def test_clip_tensor_shape():
    frames = np.random.randint(0, 255, (8, 120, 160, 3), dtype=np.uint8)
    x = to_clip_tensor(frames, 112, train=True, rng=random.Random(0))
    assert x.shape == (8, 3, 112, 112)


def test_all_model_variants_forward_and_backward():
    clip = torch.randn(2, 4, 3, 64, 64)
    for bb in BACKBONES:
        for tm in TEMPORAL:
            model = HybridHAR(5, backbone=bb, temporal=tm, dim=64, heads=4, pretrained=False)
            out = model(clip)
            assert out.shape == (2, 5)
            out.sum().backward()


def test_flops_grow_with_frames():
    model = HybridHAR(10, pretrained=False)
    assert count_gflops(model, (16, 3, 112, 112)) > 1.9 * count_gflops(model, (8, 3, 112, 112))


def test_synthetic_dataset_and_report():
    ds = SyntheticMotionDataset(8, num_frames=6, img_size=32)
    x, y = ds[1]
    assert x.shape == (6, 3, 32, 32) and y == 1
    logits = torch.eye(4)[[0, 1, 2, 3, 0]] * 5
    rep = classification_report(logits, torch.tensor([0, 1, 2, 3, 1]), ds.classes)
    assert abs(rep["top1"] - 0.8) < 1e-6 and len(rep["confusion_matrix"]) == 4
