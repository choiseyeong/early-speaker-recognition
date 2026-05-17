# 02. 전처리 (Preprocessing)

> **파일**: `src/preprocess.py`
>
> 1. 전처리란 원본 데이터(WAV + JSON)를 모델이 직접 소화할 수 있는 형태(Mel Spectrogram .npy 파일)로 변환하는 단계이다. 이 단계를 미리 실행해두면 학습 중에 매번 WAV를 읽고 변환하는 비용을 없앨 수 있다.
> 2. EDA와 전처리의 차이? <br/> EDA는 데이터를 이해하고 문제를 발견하는 과정이고, 전처리는 모델에 넣기 좋게 데이터를 고치는 과정이다.

---


## 1. 배경 지식: 전처리는 왜 필요한가?

### 문제 1: 모델은 숫자 배열만 이해한다

모델은 WAV 파일을 직접 읽을 수 없다.
파형은 1초에 16,000개 숫자가 들어있는 너무 날것의 정보이기 때문이다.
신경망에서는 입력으로는 정해진 크기의 숫자 행렬이 필요하다. 파형(1D 신호) → Mel Spectrogram(2D 행렬)으로의 변환이 바로 이 역할이다.

### 문제 2: 발화 길이가 제각각이다

```
화자 0047의 파일들:
  파일1: 2.17초 → 34,720 샘플
  파일2: 4.46초 → 71,360 샘플
  파일3: 3.41초 → 54,560 샘플
```

딥러닝 모델은 같은 배치 안의 샘플 크기가 같아야 효율적으로 처리된다. 이렇게 제각각의 길이를 다루려면 별도의 처리가 필요하다.

### 문제 3: 학습마다 변환하면 느리다

WAV → Mel Spectrogram 변환은 시간이 걸린다. 30 에폭 학습이라면 같은 변환을 30번 반복하게 됩니다. **미리 변환해서 .npy 파일로 저장**해두면 학습 시 파일 로드만 하면 된다.

### .npy 파일이란?

NumPy 배열을 그대로 저장한 바이너리 파일. `np.save()`로 저장하고 `np.load()`로 불러온다. CSV보다 훨씬 빠르고 파이썬 외부 도구 없이도 사용 가능하다.

---

## 2. 설정값(하이퍼파라미터) 설명

```python
SR          = 16000   # 샘플링 레이트
N_MELS      = 80      # Mel 필터 수
N_FFT       = 400     # STFT 창 크기 (25ms @ 16kHz)
HOP_LENGTH  = 160     # STFT 이동 크기 (10ms @ 16kHz)
FMIN        = 20      # 최소 주파수 (Hz)
FMAX        = 8000    # 최대 주파수 (Hz)

DURATIONS   = [0.5, 1.0, 1.5, 'full']
MIN_SPEECH_DURATION     = 0.5    # 이보다 짧은 발화는 제외
MIN_SAMPLES_PER_SPEAKER = 10     # 이보다 샘플 적은 화자 제외
TEST_SIZE   = 0.2                # 전체의 20%를 테스트셋으로
RANDOM_SEED = 42                 # 재현성을 위한 랜덤 시드

```

* 각 파라미터를 이렇게 설정한 이유는 EDA 문서에서 확인 가능하다.
* 랜덤 시드란, 난수 생성 알고리즘에 시작점을 주어 항상 같은 숫자가 나올 수 있게 하는 것이다. 이렇게 하면 다른 사람이 같은 코드로 돌려도 동일한 결과가 재현이 가능하다. 그리고 예를 들어 성능이 올랐을 때 그것이 모델 덕분인지, 또는 운 좋은 test셋 배정 덕분인지 알 수 있다.

---

## 3. 단계별 처리 과정

```
WAV 파일 + JSON 파일
        │
        ▼ Step 1. 파일 쌍 수집
{화자ID: [(wav_path, json_path), ...]}
        │
        ▼ Step 2. 유효 파일 필터링
발화 길이 >= 0.5초 이고 화자 샘플 >= 10개
        │
        ▼ Step 3. Train/Test 분리 (화자별 80:20)
train_pairs / test_pairs
        │
        ▼ Step 4. 각 파일에 대해:
   WAV 로드 → SpeechStart 기준으로 자르기 (4가지 duration)
        │
        ▼ Step 5. Mel Spectrogram 변환
   (N_MELS, T) 형태의 numpy 배열
        │
        ▼ Step 6. .npy 파일로 저장
outputs/processed/train(또는 test)/{duration}/{화자ID}_{번호}.npy
```

---

## 4. 핵심 함수 설명
※ 함수와 함수에 대한 설명은 전적으로 인공지능에 의해 생성되었습니다.

### `collect_file_pairs()`

```python
def collect_file_pairs():
    json_paths = sorted(JSON_ROOT.rglob('*.json'))  # JSON 파일 전체 탐색
    speaker_files = defaultdict(list)

    for json_path in json_paths:
        date       = json_path.parent.parent.name   # 폴더 이름으로 날짜 추출
        speaker_id = json_path.parent.name           # 폴더 이름으로 화자ID 추출
        wav_path   = WAV_ROOT / date / speaker_id / (json_path.stem + '.wav')

        if wav_path.exists():
            speaker_files[speaker_id].append((wav_path, json_path))

    return speaker_files  # {화자ID: [(wav경로, json경로), ...]}
```

**핵심**: JSON 파일을 기준으로 대응하는 WAV를 찾는다. 파일명이 같고 경로 구조만 다르기 때문에 `.stem`(확장자 제외 파일명)으로 매칭한다.

---

### `cut_wav(wav, speech_start, duration)`

```python
def cut_wav(wav: np.ndarray, speech_start: float, duration) -> np.ndarray:
    start = int(speech_start * SR)         # 초 → 샘플 번호

    if duration == 'full':
        return wav[start:]                 # 발화 시작부터 끝까지 전부

    end = start + int(float(duration) * SR)
    end = min(end, len(wav))               # 파일 길이 초과 방지
    return wav[start:end]
```

**예시**:
```
speech_start = 0.4초, duration = 1.0초, SR = 16000
  start = 0.4 × 16000 = 6,400번째 샘플
  end   = 6,400 + 1.0 × 16000 = 22,400번째 샘플
  → wav[6400:22400] (16,000개 샘플 = 1.0초)
```

**패딩(Padding)**:
요청한 길이보다 발화가 짧을 경우(예: 0.5초를 요청했는데 발화가 0.3초뿐인 경우) 뒤를 0으로 채운다.

```python
needed = int(float(dur) * SR)
if len(wav_cut) < needed:
    wav_cut = np.pad(wav_cut, (0, needed - len(wav_cut)))
    # (0, 부족한 샘플 수) → 앞에는 패딩 없음, 뒤에만 0으로 채움
```

---

### `wav_to_mel(wav)`

```python
def wav_to_mel(wav: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR,
        n_mels=N_MELS,          # 80
        n_fft=N_FFT,            # 400 (25ms 창)
        hop_length=HOP_LENGTH,  # 160 (10ms 이동)
        fmin=FMIN,              # 20 Hz
        fmax=FMAX,              # 8000 Hz
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)
```

변환 과정을 단계별로 설명하면:

```
1. 파형(1D)을 25ms 창으로 자름 (겹치는 부분 15ms)
2. 각 창에 FFT 적용 → 주파수별 에너지 (0~8000Hz, 201개 bin)
3. Mel 필터뱅크 적용 → 80개 Mel 빈으로 압축
4. 에너지 → dB 변환 (로그 스케일)
최종: (80, T) 형태의 float32 행렬
```

`float32`를 사용하는 이유: `float64`보다 메모리를 절반만 사용하고, GPU 연산도 더 빠르다. 정밀도 손실은 무시할 수 있는 수준이다.

---

## 5. Train/Test 분리

```python
train_pairs, test_pairs = train_test_split(
    valid_pairs,
    test_size=TEST_SIZE,       # 0.2 = 20%
    random_state=RANDOM_SEED   # 42 → 항상 같은 결과
)
```


---

## 6. 출력 파일 구조

```
outputs/processed/
├── train/
│   ├── 0.5/
│   │   ├── 0047_0000.npy   ← shape: (80, 50)  [80필터 × 50프레임]
│   │   ├── 0047_0001.npy
│   │   └── ...
│   ├── 1.0/
│   │   ├── 0047_0000.npy   ← shape: (80, 100)
│   │   └── ...
│   ├── 1.5/
│   │   └── ...             ← shape: (80, 150)
│   └── full/
│       └── ...             ← shape: (80, T) T는 발화마다 다름
│
├── test/
│   └── (동일 구조)
│
├── labels_train.npy        ← [0, 0, 0, ..., 1, 1, 1, ..., 47] 정수 배열
├── labels_test.npy
└── speaker_to_label.json   ← {"0047": 0, "0121": 1, ..., "2636": 47}
```

각 .npy 파일 하나에는 발화 하나를 Mel Spectogram으로 변환한 2D 행렬이 담겨있다. 각 행은 주파수 축, 각 열은 시간 축이다. 즉 .npy 파일은 mel spectrogram을 숫자로 표현한 행렬이라고 할 수 있다.

### labels_train.npy

각 mel spectrogram 파일에 대응하는 화자 번호를 담은 1D 배열이다. 이 매핑은 speaker_to_label.json에 저장되고, 변환된 정수 레이블들을 순서대로 쌓은 게 labels_train.npy이다.

### speaker_to_label.json

```json
{
  "0047": 0,
  "0121": 1,
  "0134": 2,
  ...
  "2636": 47
}
```


### 파일명 규칙

```
{speaker_id}_{idx:04d}.npy
  ↓
0047_0000.npy  ← 화자 0047의 train 0번째 파일
0047_0001.npy  ← 화자 0047의 train 1번째 파일
```

`{idx:04d}`: 4자리 0-패딩 정수. `sorted()`로 정렬했을 때 순서가 일관되도록 한다.


---

## 7. 실행 방법

```bash
# 프로젝트 루트에서 실행
python src/preprocess.py
```

**예상 출력**:
```
파일 쌍 수집 중...
  화자 수: 50, 총 파일: 4,835

화자 처리 중: 100%|████████████| 50/50
  [제외] 화자 2057: 유효 샘플 1개 (최소 10개 미만)
  [제외] 화자 2594: 유효 샘플 3개 (최소 10개 미만)

=== 전처리 완료 ===
  저장된 Mel 파일 수 : 약 76,000개  (4835파일 × 4duration × train+test)
  제외된 발화 수     : 10   (0.5초 미만)
  화자 수            : 48   (10개 미만 화자 2명 제외)
  저장 경로          : outputs/processed/
```

> **주의**: 전처리는 한 번만 실행하면 되고, 이미 `outputs/processed/`가 있다면 다시 실행할 필요 없다.

다음 단계: → [모델 학습 (README_03_TRAIN.md)](README_03_TRAIN.md)
