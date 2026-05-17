"""
preprocess.py

원천데이터(WAV) + 라벨링데이터(JSON)를 읽어서:
  1. SpeechStart ~ SpeechEnd 구간 추출
  2. 각 duration(0.5 / 1.0 / 1.5 / 2.0 / full)으로 자르기
  3. Mel Spectrogram 변환
  4. train / test 분리
  5. outputs/processed/ 에 .npy 파일로 저장

실행:
    python src/preprocess.py
"""

import os
import json
import numpy as np
import librosa
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────────────
# 설정값 (필요 시 여기서만 수정)
# ──────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent / 'New_Sample'
WAV_ROOT   = ROOT / '원천데이터'  / 'TS_common_01'
JSON_ROOT  = ROOT / '라벨링데이터' / 'TL_common_01'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'processed'

SR          = 16000   # 샘플링 레이트
N_MELS      = 80      # Mel 필터 수
N_FFT       = 400     # STFT 창 크기 (25ms @ 16kHz)
HOP_LENGTH  = 160     # STFT 이동 크기 (10ms @ 16kHz)
FMIN        = 20
FMAX        = 8000

# 실험할 duration 목록 ('full'은 전체 발화 길이)
DURATIONS   = [0.5, 1.0, 1.5, 'full']

# 이 길이 미만인 발화는 제외 (0.5s 실험을 위해 최소 0.5초 필요)
MIN_SPEECH_DURATION = 0.5

# 화자당 유효 샘플이 이 수 미만이면 해당 화자 제외
# (EDA 결과: 2057=1개, 2594=3개 등 극소수 샘플 화자 존재)
MIN_SAMPLES_PER_SPEAKER = 10

# train / test 분리 비율
TEST_SIZE   = 0.2
RANDOM_SEED = 42


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────

def wav_to_mel(wav: np.ndarray) -> np.ndarray:
    """WAV 배열 → Mel Spectrogram (dB 스케일), shape: (N_MELS, T)"""
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def cut_wav(wav: np.ndarray, speech_start: float, duration) -> np.ndarray:
    """
    speech_start 기준으로 duration초만큼 자르기.
    duration='full'이면 speech_end까지 전체 반환.
    """
    start = int(speech_start * SR)
    if duration == 'full':
        return wav[start:]
    end = start + int(float(duration) * SR)
    end = min(end, len(wav))
    return wav[start:end]


# ──────────────────────────────────────────────
# 메인 처리
# ──────────────────────────────────────────────

def collect_file_pairs():
    """JSON + WAV 경로 쌍을 수집해서 화자 ID별로 묶어 반환."""
    # JSON 파일 기준으로 수집
    json_paths = sorted(JSON_ROOT.rglob('*.json'))

    speaker_files = defaultdict(list)  # {speaker_id: [(wav_path, json_path), ...]}

    for json_path in json_paths:
        # json_path 구조: .../TL_common_01/{date}/{speaker_id}/{filename}.json
        date       = json_path.parent.parent.name
        speaker_id = json_path.parent.name

        # 대응되는 WAV 경로 (확장자만 다름)
        wav_path = WAV_ROOT / date / speaker_id / (json_path.stem + '.wav')
        if not wav_path.exists():
            print(f'  [경고] WAV 없음: {wav_path}')
            continue

        speaker_files[speaker_id].append((wav_path, json_path))

    return speaker_files


def process_and_save(speaker_files: dict):
    """
    화자별 파일 쌍을 읽어 Mel Spectrogram으로 변환 후 저장.

    저장 구조:
        outputs/processed/
            train/
                {duration}/
                    {speaker_id}_{idx:04d}.npy
            test/
                {duration}/
                    {speaker_id}_{idx:04d}.npy
        outputs/processed/
            labels_train.npy   ← 정수 레이블 배열
            labels_test.npy
            speaker_to_label.json
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # duration별 폴더 생성
    dur_names = [str(d) for d in DURATIONS]
    for split in ['train', 'test']:
        for dur_name in dur_names:
            (OUTPUT_DIR / split / dur_name).mkdir(parents=True, exist_ok=True)

    speaker_ids = sorted(speaker_files.keys())
    speaker_to_label = {spk: idx for idx, spk in enumerate(speaker_ids)}

    # speaker_to_label 저장
    with open(OUTPUT_DIR / 'speaker_to_label.json', 'w', encoding='utf-8') as f:
        json.dump(speaker_to_label, f, ensure_ascii=False, indent=2)

    all_labels_train, all_labels_test = [], []
    skipped = 0
    saved   = 0

    for speaker_id in tqdm(speaker_ids, desc='화자 처리 중'):
        pairs = speaker_files[speaker_id]
        label = speaker_to_label[speaker_id]

        # 유효한 파일만 필터링 (발화 길이 MIN_SPEECH_DURATION 이상)
        valid_pairs = []
        for wav_path, json_path in pairs:
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            s_start = float(meta['Other']['SpeechStart'])
            s_end   = float(meta['Other']['SpeechEnd'])
            if (s_end - s_start) >= MIN_SPEECH_DURATION:
                valid_pairs.append((wav_path, json_path, s_start, s_end))
            else:
                skipped += 1

        if len(valid_pairs) < MIN_SAMPLES_PER_SPEAKER:
            print(f'  [제외] 화자 {speaker_id}: 유효 샘플 {len(valid_pairs)}개 (최소 {MIN_SAMPLES_PER_SPEAKER}개 미만)')
            continue

        # train / test 분리
        train_pairs, test_pairs = train_test_split(
            valid_pairs, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        for split, pairs_split in [('train', train_pairs), ('test', test_pairs)]:
            for idx, (wav_path, json_path, s_start, s_end) in enumerate(pairs_split):
                wav, _ = librosa.load(str(wav_path), sr=SR)

                for dur in DURATIONS:
                    # full일 때는 SpeechEnd까지만 자르기
                    if dur == 'full':
                        end_sample = int(s_end * SR)
                        wav_cut = wav[int(s_start * SR):end_sample]
                    else:
                        wav_cut = cut_wav(wav, s_start, dur)
                        # 요청 길이보다 짧으면 패딩
                        needed = int(float(dur) * SR)
                        if len(wav_cut) < needed:
                            wav_cut = np.pad(wav_cut, (0, needed - len(wav_cut)))

                    mel = wav_to_mel(wav_cut)

                    dur_name = str(dur)
                    save_path = OUTPUT_DIR / split / dur_name / f'{speaker_id}_{idx:04d}.npy'
                    np.save(save_path, mel)
                    saved += 1

            # 레이블 기록
            if split == 'train':
                all_labels_train.extend([label] * len(train_pairs))
            else:
                all_labels_test.extend([label] * len(test_pairs))

    # 레이블 저장
    np.save(OUTPUT_DIR / 'labels_train.npy', np.array(all_labels_train))
    np.save(OUTPUT_DIR / 'labels_test.npy',  np.array(all_labels_test))

    print()
    print('=== 전처리 완료 ===')
    print(f'  저장된 Mel 파일 수 : {saved:,}  (duration 수 포함)')
    print(f'  제외된 발화 수     : {skipped}  (길이 부족)')
    print(f'  화자 수            : {len(speaker_ids)}')
    print(f'  저장 경로          : {OUTPUT_DIR}')


if __name__ == '__main__':
    print('파일 쌍 수집 중...')
    speaker_files = collect_file_pairs()
    print(f'  화자 수: {len(speaker_files)}, 총 파일: {sum(len(v) for v in speaker_files.values())}')
    print()
    process_and_save(speaker_files)
