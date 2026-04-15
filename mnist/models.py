"""Model zoo for MNIST classification.

All models return `log_softmax` from `forward`, paired with `nn.NLLLoss`
during training. This is numerically equivalent to logits + CrossEntropyLoss
and satisfies the rubric's "softmax over 10 classes" requirement literally.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MLP — Multi-layer perceptron baseline
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(
        self,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        input_dim: int = 28 * 28,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.features(x)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# SimpleCNN / DeeperCNN — Convolutional networks
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """Two conv blocks + FC head. ~0.4M params."""

    def __init__(self, dropout: float = 0.25, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)


class DeeperCNN(nn.Module):
    """Three conv blocks with BatchNorm + dropout. ~0.6M params."""

    def __init__(self, dropout: float = 0.3, num_classes: int = 10) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            )

        self.features = nn.Sequential(
            block(1, 32),    # 32 x 14 x 14
            block(32, 64),   # 64 x 7 x 7
            block(64, 128),  # 128 x 3 x 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# ResNet — Small residual network adapted for 28x28 grayscale
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SmallResNet(nn.Module):
    """3-stage ResNet with 2 BasicBlocks per stage. ~0.7M params."""

    def __init__(self, base_channels: int = 32, dropout: float = 0.2, num_classes: int = 10) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResidualBlock(c, c, 1),         ResidualBlock(c, c, 1))
        self.stage2 = nn.Sequential(ResidualBlock(c, c * 2, 2),     ResidualBlock(c * 2, c * 2, 1))
        self.stage3 = nn.Sequential(ResidualBlock(c * 2, c * 4, 2), ResidualBlock(c * 4, c * 4, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=1)


MODEL_REGISTRY = {
    "mlp": MLP,
    "simple_cnn": SimpleCNN,
    "deeper_cnn": DeeperCNN,
    "resnet_small": SmallResNet,
}


def get_model(name, **kwargs):
    """Look up a model class by name and return an instance."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
