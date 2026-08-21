"""Custom 2D Residual Convolutional Baseline."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

  def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(
        out_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_channels, out_channels, kernel_size=1, stride=stride, bias=False
          ),
          nn.BatchNorm2d(out_channels),
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = self.shortcut(x)
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += residual
    return self.relu(out)


class BaselineCNN(nn.Module):

  def __init__(self, in_channels: int = 2, num_classes: int = 4):
    super().__init__()
    self.stem = nn.Sequential(
        nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
        ),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
    )
    self.layer1 = ResidualBlock(32, 64, stride=2)  # [B, 64, 32, 32]
    self.layer2 = ResidualBlock(64, 128, stride=2)  # [B, 128, 16, 16]
    self.layer3 = ResidualBlock(128, 256, stride=2)  # [B, 256, 8, 8]
    self.gap = nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 1, 1]
    self.head = nn.Linear(256, num_classes)

  def forward(
      self, x: torch.Tensor, condition: torch.Tensor = None
  ) -> torch.Tensor:
    feat = self.stem(x)
    feat = self.layer1(feat)
    feat = self.layer2(feat)
    feat = self.layer3(feat)
    feat = self.gap(feat).flatten(1)
    return self.head(feat)