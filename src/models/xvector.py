"""
xvector.py

x-vector: TDNN 기반 화자 임베딩 모델 (2018년 Snyder et al.)

구조:
    입력 (1, 80, T)
    → Frame-level TDNN layers   : 각 프레임의 맥락(앞뒤 시간) 학습
    → Statistics Pooling        : 가변 길이 T를 고정 크기로 압축
    → Segment-level FC layers   : 화자 임베딩 생성
    → 출력: embedding (512차원)

핵심: Statistics Pooling
    - 시간 축 T에 대해 평균(mean)과 표준편차(std)를 계산
    - 길이가 달라도 항상 같은 크기 벡터 생성 → 가변 길이 음성 처리 가능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNLayer(nn.Module):
    """
    Time Delay Neural Network 레이어.

    context: 몇 프레임 앞뒤를 함께 보는지
        context=[-2,-1,0,1,2] → 현재 프레임 기준 앞뒤 2프레임씩 총 5프레임 참조
        구현상 1D Conv의 dilation으로 표현
    """

    def __init__(self, in_channels, out_channels, context_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=context_size,
            dilation=dilation,
            padding=(context_size - 1) * dilation // 2,  # 길이 유지
        )
        self.bn   = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (batch, channels, T)
        return F.relu(self.bn(self.conv(x)))


class StatisticsPooling(nn.Module):
    """
    가변 길이 T → 고정 크기 벡터.
    평균과 표준편차를 이어붙임 → 차원 2배.
    """

    def forward(self, x):
        # x: (batch, channels, T)
        mean = x.mean(dim=2)              # (batch, channels)
        std  = x.std(dim=2) + 1e-8        # (batch, channels)
        return torch.cat([mean, std], dim=1)   # (batch, channels*2)


class XVector(nn.Module):
    """
    Args:
        in_channels : Mel 필터 수 (기본 80)
        embedding_dim: 임베딩 차원 (기본 512)
        num_speakers : 화자 수 (Softmax 분류 헤드용)
    """

    def __init__(self, in_channels=80, embedding_dim=128, num_speakers=None):
        super().__init__()

        # Frame-level: 각 프레임 학습
        self.frame_layers = nn.Sequential(
            TDNNLayer(in_channels, 128, context_size=5, dilation=1),
            TDNNLayer(128, 128, context_size=3, dilation=2),
            TDNNLayer(128, 128, context_size=3, dilation=3),
            TDNNLayer(128, 128, context_size=1, dilation=1),
            TDNNLayer(128, 256, context_size=1, dilation=1),
        )

        # Statistics Pooling: (batch, 256, T) → (batch, 512)
        self.stats_pool = StatisticsPooling()

        # Segment-level: 임베딩 생성
        self.segment_layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # 분류 헤드 (학습 시 사용, 추론 시 embedding만 사용)
        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def get_embedding(self, x):
        """임베딩만 반환 (화자 비교, 코사인 유사도 계산용)"""
        # x: (batch, 1, 80, T) → (batch, 80, T)
        x = x.squeeze(1)
        x = self.frame_layers(x)
        x = self.stats_pool(x)
        x = self.segment_layers(x)
        return x   # (batch, embedding_dim)

    def forward(self, x):
        """임베딩 + 분류 로짓 반환 (학습용)"""
        emb  = self.get_embedding(x)
        logit = self.classifier(emb)
        return logit, emb
