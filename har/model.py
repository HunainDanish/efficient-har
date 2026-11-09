"""Lightweight hybrid CNN + temporal models for video action recognition.

A 2D CNN (ImageNet-pretrained) encodes every sampled frame; a temporal module
(Transformer, LSTM or plain averaging) aggregates the frame features over time.
"""
import math

import torch
from torch import nn
from torchvision import models

BACKBONES = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}
TEMPORAL = ("transformer", "lstm", "avg")


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return a CNN that maps (N, 3, H, W) -> (N, D) and its feature size D."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    fn, weights = BACKBONES[name]
    net = fn(weights=weights if pretrained else None)
    if hasattr(net, "fc"):
        net.fc = nn.Identity()
    else:
        net.classifier = nn.Identity()
    # torchvision's MobileNetV3/EfficientNet use BN momentum 0.01, tuned for long ImageNet
    # runs. In short fine-tuning runs the running statistics then lag far behind the
    # data and eval-mode accuracy collapses, so use PyTorch's default of 0.1.
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.1
    net.eval()
    with torch.no_grad():
        dim = net(torch.zeros(1, 3, 64, 64)).shape[-1]
    net.train()
    return net, dim


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        return x + self.pe[: x.shape[1]]


class TemporalTransformer(nn.Module):
    """Small pre-norm Transformer encoder with a learnable [CLS] token."""

    def __init__(self, dim: int, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos = SinusoidalPositionalEncoding(dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(x)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        return self.norm(self.encoder(x)[:, 0])


class TemporalLSTM(nn.Module):
    def __init__(self, dim: int, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return h[-1]


class TemporalAverage(nn.Module):
    """Order-invariant average over frames (TSN-style consensus)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class HybridHAR(nn.Module):
    """Frame-level CNN + temporal module + linear classifier.

    Input: clips of shape (B, T, 3, H, W), ImageNet-normalised.
    Output: logits of shape (B, num_classes).
    """

    def __init__(self, num_classes: int, backbone: str = "mobilenet_v3_small",
                 temporal: str = "transformer", dim: int = 256, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone, pretrained)
        self.proj = nn.Sequential(nn.Linear(feat_dim, dim), nn.LayerNorm(dim), nn.Dropout(dropout))
        if temporal == "transformer":
            self.temporal = TemporalTransformer(dim, layers, heads, dropout)
        elif temporal == "lstm":
            self.temporal = TemporalLSTM(dim, layers, dropout)
        elif temporal == "avg":
            self.temporal = TemporalAverage()
        else:
            raise ValueError(f"unknown temporal module {temporal!r}; choose from {TEMPORAL}")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        b, t = clips.shape[:2]
        feats = self.backbone(clips.flatten(0, 1)).view(b, t, -1)
        return self.head(self.temporal(self.proj(feats)))


def build_model(cfg: dict, num_classes: int, pretrained: bool | None = None) -> HybridHAR:
    """Build a model from a config dict (as saved in checkpoints)."""
    return HybridHAR(
        num_classes=num_classes,
        backbone=cfg["backbone"],
        temporal=cfg["temporal"],
        dim=cfg["dim"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
        pretrained=cfg["pretrained"] if pretrained is None else pretrained,
    )
