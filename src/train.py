"""
train.py

방법 C: 배치마다 랜덤 duration으로 학습
  → 모델이 0.5s / 1.0s / 1.5s / full 모두 경험
  → "정보량 자체의 한계"를 순수하게 측정 가능

실행:
    python src/train.py --model xvector
    python src/train.py --model ecapa_tdnn
    python src/train.py --model resnet34se

이어서 학습 (중단된 경우):
    python src/train.py --model xvector --resume
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from models import build_model

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'processed'
CKPT_DIR      = Path(__file__).resolve().parent.parent / 'outputs' / 'checkpoints'
LOG_DIR       = Path(__file__).resolve().parent.parent / 'outputs' / 'logs'

DURATIONS   = ['0.5', '1.0', '1.5', 'full']
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-3
FIXED_LEN   = 150   # T축 고정 프레임 수 (150 × 10ms = 1.5초)
              # → 모든 duration을 같은 길이로 맞춤
              #   0.5s(50프레임)는 패딩, full은 앞에서 자름
NUM_WORKERS = 0     # Windows에서는 0 권장


# ──────────────────────────────────────────────
# 방법 C Dataset
# ──────────────────────────────────────────────
class RandomDurationDataset(Dataset):
    """
    __getitem__ 호출마다 랜덤 duration 폴더에서 파일을 읽음.
    파일명이 duration 폴더 전체에서 동일하므로 폴더만 교체.
    """

    def __init__(self, split: str, normalize: bool = True):
        self.base_dir  = PROCESSED_DIR / split
        self.normalize = normalize

        # 파일 목록은 'full' 기준으로 구성 (어느 duration이든 동일)
        self.files  = sorted((self.base_dir / 'full').glob('*.npy'))
        labels_path = PROCESSED_DIR / f'labels_{split}.npy'
        self.labels = np.load(labels_path)

        assert len(self.files) == len(self.labels), (
            f'파일 수({len(self.files)})와 레이블 수({len(self.labels)})가 다릅니다.'
        )

        with open(PROCESSED_DIR / 'speaker_to_label.json', 'r', encoding='utf-8') as f:
            self.num_speakers = len(json.load(f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 랜덤 duration 선택
        duration = random.choice(DURATIONS)

        # 같은 파일명을 해당 duration 폴더에서 로드
        file_name = self.files[idx].name
        mel = np.load(self.base_dir / duration / file_name)  # (80, T)

        # T축 고정 (패딩 or 자르기)
        mel = self._fix_length(mel, FIXED_LEN)

        # 정규화
        if self.normalize:
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        mel_tensor = torch.from_numpy(mel).unsqueeze(0).float()  # (1, 80, T)
        label      = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_tensor, label

    def _fix_length(self, mel: np.ndarray, length: int) -> np.ndarray:
        T = mel.shape[1]
        if T >= length:
            return mel[:, :length]
        pad = np.zeros((mel.shape[0], length - T), dtype=mel.dtype)
        return np.concatenate([mel, pad], axis=1)


# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)

        optimizer.zero_grad()
        logit, _ = model(mel)
        loss = criterion(logit, label)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * len(label)
        total_correct += (logit.argmax(1) == label).sum().item()
        total         += len(label)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)
        logit, _ = model(mel)
        loss = criterion(logit, label)

        total_loss    += loss.item() * len(label)
        total_correct += (logit.argmax(1) == label).sum().item()
        total         += len(label)

    return total_loss / total, total_correct / total


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main(model_name: str, resume: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'모델  : {model_name}')
    print(f'디바이스: {device}')
    print(f'FIXED_LEN: {FIXED_LEN} 프레임 ({FIXED_LEN * 10}ms)')
    print()

    # 데이터셋
    train_ds = RandomDurationDataset('train')
    test_ds  = RandomDurationDataset('test')
    num_speakers = train_ds.num_speakers
    print(f'화자 수     : {num_speakers}')
    print(f'학습 샘플 수: {len(train_ds)}')
    print(f'테스트 샘플 수: {len(test_ds)}')
    print()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 모델
    model = build_model(model_name, num_speakers=num_speakers).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'파라미터 수 : {total_params:,}')
    print()

    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 체크포인트 저장 경로
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    best_path   = CKPT_DIR / f'{model_name}_best.pt'
    latest_path = CKPT_DIR / f'{model_name}_latest.pt'
    log_path    = LOG_DIR  / f'{model_name}_train.log'

    best_acc   = 0.0
    history    = []
    start_epoch = 1

    # Resume: 최신 체크포인트에서 이어서 학습
    if resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt.get('best_acc', 0.0)
        history     = ckpt.get('history', [])
        print(f'[Resume] 에폭 {ckpt["epoch"]}에서 이어서 학습합니다.')
        print()

    # 로그 파일 오픈 (append 모드)
    log_file = open(log_path, 'a', encoding='utf-8')
    if start_epoch == 1:
        log_file.write(f'=== {model_name} 학습 시작 ===\n')
        log_file.write(f'파라미터 수: {total_params:,}\n\n')

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        record = {
            'epoch'     : epoch,
            'train_loss': round(train_loss, 4),
            'train_acc' : round(train_acc,  4),
            'test_loss' : round(test_loss,  4),
            'test_acc'  : round(test_acc,   4),
        }
        history.append(record)

        line = (f'[{epoch:3d}/{EPOCHS}] '
                f'train loss: {train_loss:.4f}  acc: {train_acc:.4f} | '
                f'test  loss: {test_loss:.4f}  acc: {test_acc:.4f}')

        print(line, end='')
        log_file.write(line)

        # best 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch'       : epoch,
                'model_name'  : model_name,
                'model_state' : model.state_dict(),
                'num_speakers': num_speakers,
                'test_acc'    : test_acc,
            }, best_path)
            print('  ← best 저장', end='')
            log_file.write('  ← best 저장')

        print()
        log_file.write('\n')
        log_file.flush()

        # 매 에폭마다 latest 체크포인트 저장 (resume용)
        torch.save({
            'epoch'          : epoch,
            'model_name'     : model_name,
            'model_state'    : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'num_speakers'   : num_speakers,
            'best_acc'       : best_acc,
            'history'        : history,
        }, latest_path)

    log_file.close()

    # 학습 기록 저장
    history_path = CKPT_DIR / f'{model_name}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print(f'=== 학습 완료 ===')
    print(f'  best test accuracy : {best_acc:.4f}')
    print(f'  체크포인트         : {best_path}')
    print(f'  학습 기록          : {history_path}')
    print(f'  학습 로그          : {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='xvector',
        choices=['xvector', 'ecapa_tdnn', 'resnet34se'],
        help='학습할 모델 선택'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='최신 체크포인트에서 이어서 학습'
    )
    args = parser.parse_args()
    main(args.model, resume=args.resume)
