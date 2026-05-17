"""
dataset.py

preprocess.py 가 저장한 .npy 파일을 PyTorch Dataset으로 읽습니다.

사용 예시:
    from src.dataset import MelDataset
    from torch.utils.data import DataLoader

    train_ds = MelDataset(split='train', duration='1.0')
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    for mel, label in train_dl:
        # mel   shape: (batch, 1, N_MELS, T)  ← CNN 입력용
        # label shape: (batch,)
        ...
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'processed'


class MelDataset(Dataset):
    """
    Args:
        split    : 'train' 또는 'test'
        duration : '0.5', '1.0', '1.5', '2.0', 'full' 중 하나
        normalize: True면 각 샘플을 평균0 표준편차1로 정규화
        fixed_len: None이면 가변 길이, 정수면 T축을 해당 길이로 자르거나 패딩
    """

    def __init__(self, split: str, duration: str, normalize: bool = True, fixed_len: int = None):
        assert split in ('train', 'test'), "split은 'train' 또는 'test'"

        self.data_dir  = PROCESSED_DIR / split / duration
        self.normalize = normalize
        self.fixed_len = fixed_len

        # 파일 목록과 레이블 로드
        label_path = PROCESSED_DIR / f'labels_{split}.npy'
        self.files  = sorted(self.data_dir.glob('*.npy'))
        self.labels = np.load(label_path)

        assert len(self.files) == len(self.labels), (
            f'파일 수({len(self.files)})와 레이블 수({len(self.labels)})가 다릅니다. '
            f'preprocess.py를 다시 실행하세요.'
        )

        # 화자 수 확인
        with open(PROCESSED_DIR / 'speaker_to_label.json', 'r', encoding='utf-8') as f:
            self.speaker_to_label = json.load(f)
        self.num_speakers = len(self.speaker_to_label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = np.load(self.files[idx])   # shape: (N_MELS, T)

        # 길이 고정 (CNN 배치 처리를 위해 필요)
        if self.fixed_len is not None:
            mel = self._fix_length(mel, self.fixed_len)

        # 정규화
        if self.normalize:
            mean = mel.mean()
            std  = mel.std() + 1e-8
            mel  = (mel - mean) / std

        # (N_MELS, T) → (1, N_MELS, T)  ← 채널 차원 추가 (CNN 입력)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)
        label      = torch.tensor(self.labels[idx], dtype=torch.long)

        return mel_tensor, label

    def _fix_length(self, mel: np.ndarray, length: int) -> np.ndarray:
        """T 축을 length로 자르거나 0 패딩"""
        T = mel.shape[1]
        if T >= length:
            return mel[:, :length]
        else:
            pad = np.zeros((mel.shape[0], length - T), dtype=mel.dtype)
            return np.concatenate([mel, pad], axis=1)


def get_dataloaders(duration: str, batch_size: int = 32, fixed_len: int = None, num_workers: int = 0):
    """
    train / test DataLoader를 한번에 반환하는 편의 함수.

    Args:
        duration  : '0.5', '1.0', '1.5', '2.0', 'full'
        batch_size: 배치 크기
        fixed_len : T축 고정 길이 (None이면 가변)
        num_workers: DataLoader worker 수 (Windows에서는 0 권장)

    Returns:
        train_loader, test_loader, num_speakers
    """
    train_ds = MelDataset('train', duration, fixed_len=fixed_len)
    test_ds  = MelDataset('test',  duration, fixed_len=fixed_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, train_ds.num_speakers
