"""
ecapa_tdnn.py

ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation TDNN
(Desplanques et al., Interspeech 2020)

x-vector 대비 개선점:
    1. SE-Res2Block   : 채널 어텐션 + 다중 스케일 특징 추출
    2. Multi-scale pooling: 여러 레이어의 출력을 합쳐서 풀링
    3. 더 작은 데이터에서도 좋은 성능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation: 채널별 중요도(가중치)를 학습.
    중요한 주파수 채널은 강조, 덜 중요한 건 억제.
    """

    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),      # (batch, C, T) → (batch, C, 1)
            nn.Flatten(),                  # (batch, C)
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, C, T)
        scale = self.se(x).unsqueeze(2)   # (batch, C, 1)
        return x * scale                   # 채널별 스케일링


class Res2Conv1d(nn.Module):
    """
    Res2Net 방식의 1D Conv: 채널을 scale개로 쪼개서 계층적으로 처리.
    다양한 시간 스케일의 특징을 동시에 학습.
    """

    def __init__(self, channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        assert channels % scale == 0
        self.scale    = scale
        self.width    = channels // scale
        self.convs    = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size,
                      dilation=dilation,
                      padding=(kernel_size - 1) * dilation // 2)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x):
        chunks = torch.chunk(x, self.scale, dim=1)
        out    = [chunks[0]]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            y = chunks[i + 1] if i == 0 else chunks[i + 1] + out[-1]
            out.append(F.relu(bn(conv(y))))
        return torch.cat(out, dim=1)


class SERes2Block(nn.Module):
    """
    ECAPA-TDNN의 핵심 블록:
        1x1 Conv → Res2Conv → 1x1 Conv → SE → residual 합산
    """

    def __init__(self, channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        self.conv1  = nn.Conv1d(channels, channels, 1)
        self.bn1    = nn.BatchNorm1d(channels)
        self.res2   = Res2Conv1d(channels, scale=scale, kernel_size=kernel_size, dilation=dilation)
        self.conv2  = nn.Conv1d(channels, channels, 1)
        self.bn2    = nn.BatchNorm1d(channels)
        self.se     = SEModule(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.res2(x))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.relu(x + residual)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling:
    단순 평균/분산 대신, 중요한 시간 프레임에 더 높은 가중치를 줌.
    """

    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, channels, 1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        # x: (batch, C, T)
        global_mean = x.mean(dim=2, keepdim=True).expand_as(x)
        global_std  = x.std(dim=2, keepdim=True).expand_as(x)
        context     = torch.cat([x, global_mean, global_std], dim=1)

        attn   = self.attention(context)         # (batch, C, T)
        mean   = (attn * x).sum(dim=2)           # (batch, C)
        std    = (attn * (x ** 2)).sum(dim=2) - mean ** 2
        std    = (std.clamp(min=1e-8)).sqrt()

        return torch.cat([mean, std], dim=1)     # (batch, C*2)


class ECAPA_TDNN(nn.Module):
    """
    Args:
        in_channels  : Mel 필터 수 (기본 80)
        channels     : 내부 채널 수 (기본 512 or 1024)
        embedding_dim: 임베딩 차원 (기본 192)
        num_speakers : 화자 수
    """

    def __init__(self, in_channels=80, channels=128, embedding_dim=128, num_speakers=None):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        self.layer1 = SERes2Block(channels, kernel_size=3, dilation=2)
        self.layer2 = SERes2Block(channels, kernel_size=3, dilation=3)
        self.layer3 = SERes2Block(channels, kernel_size=3, dilation=4)

        # 3개 레이어 출력을 이어붙여 멀티스케일 특징 활용
        self.cat_conv = nn.Conv1d(channels * 3, channels * 3, 1)
        self.cat_bn   = nn.BatchNorm1d(channels * 3)

        self.pool = AttentiveStatisticsPooling(channels * 3)

        self.embedding = nn.Sequential(
            nn.Linear(channels * 6, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def get_embedding(self, x):
        """임베딩만 반환"""
        x = x.squeeze(1)                          # (batch, 1, 80, T) → (batch, 80, T)
        x = self.input_conv(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x_cat = torch.cat([x1, x2, x3], dim=1)   # (batch, C*3, T)
        x_cat = F.relu(self.cat_bn(self.cat_conv(x_cat)))

        x_pool = self.pool(x_cat)                 # (batch, C*6)
        emb    = self.embedding(x_pool)            # (batch, embedding_dim)
        return emb

    def forward(self, x):
        emb   = self.get_embedding(x)
        logit = self.classifier(emb)
        return logit, emb
