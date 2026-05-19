"""
evaluate.py

각 모델의 체크포인트를 로드하고, duration별(0.5s / 1.0s / 1.5s / full) 정확도를 측정.
결과는 outputs/results/evaluation_results.json 에 저장.

실행:
    python src/evaluate.py
    python src/evaluate.py --models xvector ecapa_tdnn   # 일부 모델만
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'processed'
CKPT_DIR      = Path(__file__).resolve().parent.parent / 'outputs' / 'checkpoints'
RESULTS_DIR   = Path(__file__).resolve().parent.parent / 'outputs' / 'results'

DURATIONS   = ['0.5', '1.0', '1.5', 'full']
FIXED_LEN   = 300
BATCH_SIZE  = 64
NUM_WORKERS = 0


# ──────────────────────────────────────────────
# 고정 duration Dataset
# ──────────────────────────────────────────────
class FixedDurationDataset(Dataset):
    """특정 duration의 Mel 파일만 로드."""

    def __init__(self, split: str, duration: str):
        self.dir    = PROCESSED_DIR / split / duration
        self.files  = sorted(self.dir.glob('*.npy'))
        labels_path = PROCESSED_DIR / f'labels_{split}.npy'
        self.labels = np.load(labels_path)

        assert len(self.files) == len(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = np.load(self.files[idx])        # (80, T)
        mel = self._fix_length(mel, FIXED_LEN)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).float()
        label      = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_tensor, label

    def _fix_length(self, mel, length):
        T = mel.shape[1]
        if T >= length:
            return mel[:, :length]
        pad = np.zeros((mel.shape[0], length - T), dtype=mel.dtype)
        return np.concatenate([mel, pad], axis=1)


# ──────────────────────────────────────────────
# 평가 함수
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_duration(model, duration: str, device):
    """한 duration에 대한 test accuracy 반환."""
    ds     = FixedDurationDataset('test', duration)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    correct, total = 0, 0
    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)
        logit, _   = model(mel)
        correct += (logit.argmax(1) == label).sum().item()
        total   += len(label)

    return correct / total


def evaluate_model(model_name: str, device):
    """모델 체크포인트 로드 후 전 duration 평가."""
    best_path = CKPT_DIR / f'{model_name}_best.pt'
    if not best_path.exists():
        print(f'  [건너뜀] {model_name}: 체크포인트 없음 ({best_path})')
        return None

    ckpt         = torch.load(best_path, map_location=device)
    num_speakers = ckpt['num_speakers']
    model        = build_model(model_name, num_speakers=num_speakers).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    best_acc = ckpt.get('test_acc', None)
    print(f'\n[{model_name}]  (학습 best acc: {best_acc:.4f})')

    results = {}
    for dur in DURATIONS:
        acc = evaluate_duration(model, dur, device)
        results[dur] = round(acc, 4)
        label = f'{float(dur):.1f}s' if dur != 'full' else 'full'
        print(f'  {label:6s} → accuracy: {acc:.4f} ({acc*100:.1f}%)')

    return results


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main(model_names: list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'디바이스: {device}')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / 'evaluation_results.json'

    # 기존 결과 로드 (있으면 합산 — 여러 사람이 순차 실행 가능)
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for model_name in model_names:
        print(f'\n{"="*50}')
        result = evaluate_model(model_name, device)
        if result is not None:
            all_results[model_name] = result

    # 저장
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f'\n결과 저장: {results_path}')

    # 요약 출력
    print(f'\n{"="*50}')
    print('[ 요약: duration별 accuracy ]')
    print(f'{"모델":<15}', end='')
    for d in DURATIONS:
        label = f'{float(d):.1f}s' if d != 'full' else 'full'
        print(f'{label:>8}', end='')
    print()
    print('-' * (15 + 8 * len(DURATIONS)))
    for model_name, res in all_results.items():
        print(f'{model_name:<15}', end='')
        for d in DURATIONS:
            val = res.get(d, '-')
            if isinstance(val, float):
                print(f'{val*100:>7.1f}%', end='')
            else:
                print(f'{"-":>8}', end='')
        print()


if __name__ == '__main__':
    ALL_MODELS = ['xvector', 'ecapa_tdnn', 'resnet34se']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        choices=ALL_MODELS,
        help='평가할 모델 목록 (기본: 체크포인트 있는 것 전부)'
    )
    args = parser.parse_args()

    if args.models:
        target_models = args.models
    else:
        # 체크포인트 있는 모델만 자동 선택
        target_models = [m for m in ALL_MODELS if (CKPT_DIR / f'{m}_best.pt').exists()]

    print(f'평가 대상 모델: {target_models}')
    main(target_models)
