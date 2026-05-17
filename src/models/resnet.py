"""
resnet.py

ResNet34-SE 기반 화자 인식 모델.
Mel Spectrogram을 (1, 80, T) 이미지처럼 취급하여 2D CNN으로 처리.

구조:
    입력 (batch, 1, 80, T)
    → ResNet34 블록들 (2D Conv)
    → Statistics Pooling (T 축 압축)
    → FC → embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock2D(nn.Module):
    """2D Squeeze-and-Excitation 블록"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class BasicBlock(nn.Module):
    """ResNet BasicBlock + SE (2D)"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.se    = SEBlock2D(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample:
            x = self.downsample(x)
        return F.relu(out + x)


class ResNet34SE(nn.Module):
    """
    Args:
        embedding_dim: 임베딩 차원 (기본 256)
        num_speakers : 화자 수
    """

    def __init__(self, embedding_dim=128, num_speakers=None):
        super().__init__()

        # 초기 Conv: (batch, 1, 80, T) → (batch, 16, 80, T)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # ResNet 레이어 (주파수 축만 압축, 시간 축은 유지)
        # stride=(2,1): 주파수 축 절반, 시간 축 유지
        self.layer1 = self._make_layer(16,  16,  blocks=3, stride=1)
        self.layer2 = self._make_layer(16,  32,  blocks=4, stride=(2, 1))
        self.layer3 = self._make_layer(32,  64,  blocks=6, stride=(2, 1))
        self.layer4 = self._make_layer(64,  128, blocks=3, stride=(2, 1))

        # 주파수 축 완전 압축: (batch, 128, freq, T) → (batch, 128, T)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Statistics Pooling: 시간 축 T 압축
        # (batch, 128, T) → (batch, 256)
        self.embedding = nn.Sequential(
            nn.Linear(128 * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def get_embedding(self, x):
        """임베딩만 반환. x: (batch, 1, 80, T)"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.freq_pool(x)             # (batch, 128, 1, T)
        x = x.squeeze(2)                  # (batch, 128, T)

        # Statistics Pooling
        mean = x.mean(dim=2)              # (batch, 128)
        std  = x.std(dim=2) + 1e-8        # (batch, 128)
        x    = torch.cat([mean, std], dim=1)  # (batch, 256)

        emb = self.embedding(x)           # (batch, embedding_dim)
        return emb

    def forward(self, x):
        emb   = self.get_embedding(x)
        logit = self.classifier(emb)
        return logit, emb
