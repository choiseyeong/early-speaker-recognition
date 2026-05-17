# 03. 모델 학습 (Model Training)

> **파일**: `src/train.py`, `src/models/`
>
> 전처리된 Mel Spectrogram을 입력으로 받아 화자를 분류하는 딥러닝 모델을 학습하는 단계이다. 이 프로젝트에서는 세 가지 모델(X-Vector, ECAPA-TDNN, ResNet34-SE)을 동일한 조건에서 비교한다.

---

## 1. 배경 지식: 딥러닝으로 화자를 어떻게 구분하는가?

### 화자 임베딩 (Speaker Embedding)

딥러닝 기반 화자 인식의 핵심 아이디어는 **임베딩(embedding)** 이다. 임베딩이란 고차원 데이터(예: 수천 차원의 Mel Spectrogram)를 저차원의 의미 있는 벡터로 압축하는 과정이다.


```
Mel Spectrogram (80 × 150) → 모델 → 임베딩 벡터 (128차원)
     12,000개 숫자                       128개 숫자

* 128차원은 위치를 지정하는 데 128개의 독립적인 좌표나 데이터가 필요한 공간
```

잘 학습된 모델은 같은 화자의 발화들은 임베딩 공간에서 가깝게, 다른 화자의 발화들은 멀리 위치시킨다.

참고: https://s-core.co.kr/insight/view/%EB%B2%A1%ED%84%B0-%EC%9E%84%EB%B2%A0%EB%94%A9-%EB%AA%A8%EB%8D%B8%EC%9D%98-%EC%9D%B4%ED%95%B4%EC%99%80-%ED%99%9C%EC%9A%A9-%EC%B5%9C%EC%A0%81%EC%9D%98-%EC%9E%84%EB%B2%A0%EB%94%A9-%EB%AA%A8%EB%8D%B8/

### 학습 방법: 분류 문제로 치환

직접 "두 발화가 같은 화자인가?"를 학습하는 대신, **N명 중 누구인지 맞히는 분류 문제**로 학습한다.

```
입력: Mel Spectrogram
출력: [화자0일 확률, 화자1일 확률, ..., 화자47일 확률]
정답: 실제 화자 번호 (정수 레이블)
```

48명 화자를 정확하게 구분하려면 모델이 각 화자의 음성 특성을 학습해야 한다. 학습 후 분류 헤드를 제거하고 임베딩만 사용하면 새로운 화자도 비교할 수 있다.

### (참고) 딥러닝의 주요 문제 유형

| 유형 | 출력 형태 | 예시 |
|------|---------|------|
| **분류 (Classification)** | 클래스 확률 | 화자 구분, 이미지 분류, 스팸 필터 |
| **회귀 (Regression)** | 연속 숫자 | 집값 예측, 발화 길이 예측 |
| **생성 (Generation)** | 새로운 데이터 | 음성 합성(TTS), 이미지 생성, ChatGPT |
| **임베딩 학습 (Metric Learning)** | 거리 관계 | 얼굴 인식, 유사 문서 검색 |
| **시퀀스-to-시퀀스 (Seq2Seq)** | 가변 길이 시퀀스 | 음성 인식(STT), 번역 |
| **강화학습 (RL)** | 행동 선택 | 게임 AI, 로봇 제어 |



### 이 프로젝트의 목표

"발화 시작 후 **얼마나 짧은 구간**만으로도 화자를 구분할 수 있는가?"

→ 0.5초 / 1.0초 / 1.5초 / 전체 길이를 섞어 학습하고, 각 duration에서의 정확도를 측정한다.

---

## 2. 핵심 설계: 배치마다 랜덤 Duration

### 왜 랜덤 Duration인가?

---

**방법 A** — 한 모델을 duration별로 나눠서 순서대로 학습

에폭을 4등분해서, 처음엔 0.5초 데이터, 그 다음엔 1.0초 데이터... 순서대로 학습한다.

```
에폭  1~ 7: 0.5초 데이터만 학습
에폭  8~15: 1.0초 데이터만 학습
에폭 16~22: 1.5초 데이터만 학습
에폭 23~30: full 데이터만 학습
```

- 문제 1: 나중에 학습한 duration(full)에 편향됨 — 모델이 마지막에 본 것을 더 잘 기억하는 경향(catastrophic forgetting)
- 문제 2: 학습 후반에 0.5초 입력을 주면 성능이 떨어질 수 있음

**방법 B (채택)** — 매 샘플마다 4가지 duration 중 하나를 랜덤으로 선택

```python
duration = random.choice(['0.5', '1.0', '1.5', 'full'])
mel = np.load(base_dir / duration / file_name)
```

**장점**:
- duration별로 모델을 따로 만들면, 0.5초 모델이 성능이 낮을 때 "음성 정보가 부족해서인지" vs "이 모델이 짧은 음성에 덜 학습됐기 때문인지"를 구분할 수 없다. 하나의 모델이 모든 duration을 경험하면 모델 역량은 동일하게 유지되므로, 성능 차이가 순수하게 "0.5초에 담긴 정보 vs 1.5초에 담긴 정보의 차이"가 된다.
- 같은 발화를 0.5초, 1.0초, 1.5초, full 버전으로 번갈아 보여주는 것 자체가 다양한 입력을 경험시키는 것이기 때문에 데이터 증강(Data Augmentation) 효과가 있다. 모델 입장에서는 매 배치마다 조금씩 다른 형태의 데이터를 보게 되므로 특정 길이에 과적합되는 것을 막아준다.
- 실제 배포 환경에서 음성 길이는 제각각이다. 하나의 모델이 짧은 것도 긴 것도 다 경험했기 때문에, 학습 때 보지 못한 길이가 들어와도 어느 정도 대응할 수 있다.

---

## 3. 데이터 로더 (RandomDurationDataset)

```python
class RandomDurationDataset(Dataset):
    def __init__(self, split: str, normalize: bool = True):
        self.base_dir = PROCESSED_DIR / split      # 'train' 또는 'test'
        self.files    = sorted((self.base_dir / 'full').glob('*.npy'))  # full 기준으로 파일 목록
        self.labels   = np.load(PROCESSED_DIR / f'labels_{split}.npy')

    def __getitem__(self, idx):
        duration  = random.choice(DURATIONS)       # 랜덤 duration 선택
        file_name = self.files[idx].name           # ex) '0047_0000.npy'
        mel = np.load(self.base_dir / duration / file_name)

        mel = self._fix_length(mel, FIXED_LEN)     # 150 프레임으로 정규화

        if self.normalize:
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        mel_tensor = torch.from_numpy(mel).unsqueeze(0).float()  # (1, 80, 150)
        label      = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_tensor, label
```

### `_fix_length(mel, length=150)`

모든 duration의 Mel을 150 프레임으로 통일. (신경망에 넣으려면 모든 입력의 크기가 같아야 하기 때문)

```
0.5초 (50프레임):  [▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░]  → 뒤에 0 패딩
1.0초 (100프레임): [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░]  → 뒤에 0 패딩
1.5초 (150프레임): [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓]  → 그대로
full (200프레임+): [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓]  → 앞 150프레임만 사용
```

150인 이유: 1.5초 × (1프레임/10ms) = 150 프레임. "1.5초 이상이면 온전히 포함"되는 기준값.

### 정규화 (Normalize)

```python
mel = (mel - mel.mean()) / (mel.std() + 1e-8)
```

각 샘플을 평균 0, 표준편차 1로 변환한다(Z-score 정규화).

- 서로 다른 녹음 환경에서 오는 음량 차이를 제거
- 모델이 절대적인 에너지 크기 대신 상대적인 패턴에 집중할 수 있도록
- `+ 1e-8`: 분모가 0이 되는 것을 방지 (완전히 무음인 구간이 있을 경우 대비)

### `unsqueeze(0)`: 채널 차원 추가

```
(80, 150) → (1, 80, 150)
```

CNN은 `(채널, 높이, 너비)` 형식을 기대한다. Mel Spectrogram은 채널이 1개(흑백 이미지와 동일)이므로 채널 차원을 추가한다. 채널이 추가되어도 실제 데이터는 그대로이고 형태만 바뀐다.

---

## 4. 모델 1: X-Vector

※ 이하 3가지 모델들의 설명에 관해서는 전적으로 AI의 도움을 받음.

> **출처**: Snyder et al., "X-Vectors: Robust DNN Embeddings for Speaker Recognition", ICASSP 2018

### 전체 구조

```
입력: (batch, 1, 80, 150)
  ↓ squeeze(1)
(batch, 80, 150)          ← 2D → 1D: 주파수 축을 채널로 취급
  ↓ TDNN 레이어 5개
(batch, 256, 150)         ← 프레임 수(시간 축)는 유지
  ↓ Statistics Pooling
(batch, 512)              ← [평균; 표준편차] 이어붙임 → 차원 2배
  ↓ FC 레이어 2개
(batch, 128)              ← 화자 임베딩
  ↓ 분류 헤드
(batch, 48)               ← 화자 수만큼 출력
```

### 핵심 개념: TDNN (Time Delay Neural Network)

일반 1D Conv와 같지만, **dilation(팽창)** 을 사용해 더 넓은 시간 범위를 효율적으로 참조한다.

```
dilation=1: [프레임1, 프레임2, 프레임3] → 연속된 3프레임
dilation=2: [프레임1, 프레임3, 프레임5] → 1프레임 건너뛰며 5프레임 범위 커버
dilation=3: [프레임1, 프레임4, 프레임7] → 2프레임 건너뛰며 7프레임 범위 커버
```

이 프로젝트의 TDNN 레이어 구성:

| 레이어 | 입력 채널 | 출력 채널 | Context | Dilation | 참조 범위 |
|--------|---------|---------|---------|---------|---------|
| TDNN1  | 80      | 128     | 5       | 1       | 5프레임 = 50ms |
| TDNN2  | 128     | 128     | 3       | 2       | 5프레임 = 50ms (건너뜀) |
| TDNN3  | 128     | 128     | 3       | 3       | 7프레임 = 70ms (건너뜀) |
| TDNN4  | 128     | 128     | 1       | 1       | 1프레임 (포인트와이즈) |
| TDNN5  | 128     | 256     | 1       | 1       | 1프레임 (차원 확장) |

### 핵심 개념: Statistics Pooling

가변 길이 음성을 고정 크기 벡터로 변환하는 핵심 모듈이다.

```python
def forward(self, x):
    # x: (batch, 256, T)  ← T는 시간 프레임 수
    mean = x.mean(dim=2)              # (batch, 256) ← 시간 평균
    std  = x.std(dim=2) + 1e-8       # (batch, 256) ← 시간 표준편차
    return torch.cat([mean, std], dim=1)  # (batch, 512)
```

**왜 평균만 쓰지 않고 표준편차도 쓰는가?**
- 평균: "이 화자는 평균적으로 어떤 주파수를 사용하는가"
- 표준편차: "이 화자의 음성은 시간에 따라 얼마나 변화하는가"
- 두 정보를 합치면 화자 구분력이 더 높아진다

---

## 5. 모델 2: ECAPA-TDNN

> **출처**: Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification", Interspeech 2020

X-Vector의 세 가지 한계를 개선한 모델이다.

### 전체 구조

```
입력: (batch, 1, 80, 150)
  ↓ squeeze(1) + Conv1d(80→128, kernel=5)
(batch, 128, 150)
  ↓ SERes2Block × 3 (dilation: 2, 3, 4)
  x1:(batch,128,150), x2:(batch,128,150), x3:(batch,128,150)
  ↓ Concat [x1, x2, x3]
(batch, 384, 150)         ← 3개 레이어 출력 합산 (멀티스케일)
  ↓ Attentive Statistics Pooling
(batch, 768)              ← [어텐션 가중 평균; 어텐션 가중 표준편차]
  ↓ FC
(batch, 128)              ← 화자 임베딩
  ↓ 분류 헤드
(batch, 48)
```

### 개선점 1: SE Module (Squeeze-and-Excitation)

채널별 중요도를 학습한다. "이 주파수 채널은 화자 구분에 중요, 저 채널은 덜 중요"를 스스로 파악한다.

```python
def forward(self, x):
    # x: (batch, C, T)
    scale = self.se(x)          # (batch, C, 1) ← 채널별 가중치 (0~1)
    return x * scale            # 중요한 채널은 강조, 덜 중요한 채널은 억제
```

내부적으로:
1. 전체 시간을 평균내 채널별 요약 생성 → `(batch, C, 1)`
2. FC → ReLU → FC → Sigmoid 로 0~1 가중치 학습
3. 원본에 곱함

### 개선점 2: Res2Conv1d (다중 스케일 특징)

채널을 8개 그룹으로 나눠 서로 다른 시간 스케일을 동시에 학습한다.

```
입력 채널 128개 → 8개 그룹(16채널씩)
그룹 0: 그대로 통과
그룹 1: Conv 적용
그룹 2: Conv 적용 (그룹1 결과 + 그룹2 입력)  ← 앞 그룹 정보가 누적
그룹 3: Conv 적용 (그룹2 결과 + 그룹3 입력)
...
```

이를 통해 단일 레이어에서 여러 시간 스케일(음소 수준 ~ 단어 수준)을 학습한다.

### 개선점 3: Attentive Statistics Pooling

단순 평균/표준편차 대신, 중요한 시간 프레임에 더 높은 가중치를 준다.

```python
context = torch.cat([x, global_mean, global_std], dim=1)  # 전체 맥락 포함
attn    = self.attention(context)  # (batch, C, T) ← 프레임별 가중치
mean    = (attn * x).sum(dim=2)   # 가중 평균
```

예: 발화 중간의 중요한 음절에 더 집중하고, 무음 구간이나 노이즈는 무시할 수 있음

### 개선점 4: Multi-scale Aggregation

3개 SERes2Block의 출력을 모두 합쳐서 풀링한다.

```python
x_cat = torch.cat([x1, x2, x3], dim=1)  # 각 레이어의 출력을 채널 방향으로 합침
```

얕은 레이어(저수준 특징)와 깊은 레이어(고수준 특징)를 모두 활용한다.

---

## 6. 모델 3: ResNet34-SE

ResNet을 1D 시퀀스 대신 **2D 이미지**처럼 처리하는 접근법이다. Mel Spectrogram을 (주파수 × 시간) 이미지로 보고 2D Conv를 적용한다.

### 전체 구조

```
입력: (batch, 1, 80, 150)   ← (배치, 채널=1, 주파수=80, 시간=150)
  ↓ Stem: Conv2d(1→16, 3×3)
(batch, 16, 80, 150)
  ↓ Layer1: BasicBlock×3, stride=1
(batch, 16, 80, 150)        ← 크기 유지
  ↓ Layer2: BasicBlock×4, stride=(2,1)
(batch, 32, 40, 150)        ← 주파수만 절반으로 (80→40)
  ↓ Layer3: BasicBlock×6, stride=(2,1)
(batch, 64, 20, 150)        ← 주파수만 절반으로 (40→20)
  ↓ Layer4: BasicBlock×3, stride=(2,1)
(batch, 128, 10, 150)       ← 주파수만 절반으로 (20→10)
  ↓ AdaptiveAvgPool2d((1, None))
(batch, 128, 1, 150)        ← 주파수 축 완전 압축
  ↓ squeeze(2)
(batch, 128, 150)           ← 이제 1D 시퀀스처럼
  ↓ Statistics Pooling [mean; std]
(batch, 256)
  ↓ FC
(batch, 128)                ← 화자 임베딩
  ↓ 분류 헤드
(batch, 48)
```

### 핵심 설계: stride=(2, 1)

```python
self.layer2 = self._make_layer(16, 32, blocks=4, stride=(2, 1))
#                                                        ↑  ↑
#                                                   주파수  시간
```

- 주파수 축(axis 2)은 절반으로 압축: 높은 주파수 간 유사한 패턴을 합침
- 시간 축(axis 3)은 그대로 유지: 시간 해상도를 잃지 않음

이는 음성 처리에서 검증된 방식이다. 시간 정보를 최대한 유지해야 Speech 특성을 잘 학습할 수 있다.

### ResNet BasicBlock + SE

```python
def forward(self, x):
    out = relu(bn1(conv1(x)))     # 3×3 Conv
    out = bn2(conv2(out))         # 3×3 Conv
    out = self.se(out)            # SE: 채널별 가중치
    if self.downsample:
        x = self.downsample(x)   # stride로 크기가 달라진 경우 residual 맞춤
    return relu(out + x)          # Residual 연결
```

**Residual 연결(잔차 연결)**: `out + x`. 입력을 그대로 더해줌으로써 그래디언트가 깊은 층까지 잘 전달되게 한다. 이 아이디어 덕분에 수십~수백 층도 학습 가능해졌다.

---

## 7. 학습 설정 (Loss, Optimizer, Scheduler)

### 하이퍼파라미터

```python
EPOCHS     = 30      # 전체 데이터를 30번 반복 학습
BATCH_SIZE = 32      # 한 번에 32개 샘플을 처리
LR         = 1e-3    # 학습률 (초기값 0.001)
FIXED_LEN  = 150     # Mel 시간 프레임 고정 길이
```

### Loss: CrossEntropyLoss

```python
criterion = nn.CrossEntropyLoss()
```

분류 문제의 표준 손실 함수. 모델 출력(logit)과 정답 레이블의 차이를 측정한다.

```
CrossEntropy = -log(정답 클래스의 예측 확률)
  → 정답을 1.0으로 예측하면 loss = 0
  → 정답을 0.0으로 예측하면 loss = 무한대
```

### Optimizer: AdamW

> * Optimizer(옵티마이저)란?<br/>
딥러닝과 머신러닝 학습 과정에서 모델의 예측값과 실제 정답 의 오차를 줄이기 위해, 모델 내부의 파라미터(가중치 \(w\), 편향 \(b\))를 최적화하여 갱신(Update)하는 알고리즘.간
> * Adam이란?<br/>
경사하강법의 발전형. 기본 경사하강법은 모든 파라미터에 같은 학습률을 적용하는데, Adam은 파라미터마다 학습률을 개별적으로 조정한다. 단순히 한 방향으로 조금 이동하는 게 아니라 이전 gradient의 흐름(Momentum)과 크기(RMSProp 방식)까지 고려해서 더 효율적으로 이동


```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
```

* AdamW란?
Adam 옵티마이저에 **Weight Decay(과적합 방지를 위해 가중치를 조금씩 감쇠시키는 것)** 를 올바르게 적용한 버전이다. Adam은 학습률을 파라미터마다 다르게 조정하기 때문에, 여기에 Weight Decay를 얹으면 조정된 학습률에 비례해서 감쇠량도 달라진다. 즉 파라미터마다 Weight Decay가 제멋대로 적용된다.<br/>
AdamW는 이걸 분리해서, 학습률 조정과 무관하게 모든 파라미터에 동일한 비율로 Weight Decay를 적용한다.


### Scheduler: CosineAnnealingLR

> * Scheduler란?<br>
학습 중에 learning rate(학습률)을 자동으로 조절해주는 도구이다. 모델 학습에서는 보통 처음에는 크게 움직이고, 나중에는 조금씩 정교하게 맞추는 방식이 권장된다.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

학습률을 코사인 곡선을 따라 점진적으로 감소시킴.

```
학습률
0.001 |●
      |  ↘
0.0005|     ↘
      |       ↘
  0   |_________●
      에폭 1    에폭 30
```

---

## 8. 학습 루프

### 학습 단계 (train_one_epoch)

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()               # Dropout, BatchNorm을 학습 모드로 설정

    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)

        optimizer.zero_grad()   # 이전 배치의 그래디언트 초기화
        logit, _ = model(mel)   # 순전파 (forward pass)
        loss = criterion(logit, label)
        loss.backward()         # 역전파 (backward pass) → 그래디언트 계산
        optimizer.step()        # 파라미터 업데이트
```

**`model.train()` vs `model.eval()`**:
- `train()`: Dropout이 활성화되어 랜덤하게 뉴런을 끔. BatchNorm이 배치 통계 사용
- `eval()`: Dropout 비활성화. BatchNorm이 학습 중 누적된 이동 평균 사용
- 평가 시에는 반드시 `model.eval()`을 호출해야 일관된 결과가 나온다.

### 평가 단계 (evaluate)
현재 모델이 얼마나 잘 맞히는지 측정

```python
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    ...
```

`@torch.no_grad()`: 그래디언트를 계산하지 않아 메모리와 계산량을 절약. 평가에서는 파라미터를 업데이트하지 않으므로 그래디언트가 필요 없다.

### 에폭당 출력 예시

```
[  1/30] train loss: 3.8521  acc: 0.0612 | test  loss: 3.7894  acc: 0.0741
[  2/30] train loss: 3.5203  acc: 0.1234 | test  loss: 3.4812  acc: 0.1389
...
[ 15/30] train loss: 1.2341  acc: 0.6823 | test  loss: 1.4521  acc: 0.6312  ← best 저장
...
[ 30/30] train loss: 0.8123  acc: 0.8234 | test  loss: 1.1234  acc: 0.7891
```

- `train acc`: 학습 데이터에서의 정확도 (높을수록 학습이 잘 됨)
- `test acc`: 테스트 데이터에서의 정확도 (이게 실제 성능. train acc보다 낮은 게 정상)
- `← best 저장`: test acc가 이전 최고를 넘으면 best 체크포인트 저장

---

## 9. 체크포인트와 Resume(재시작)

### 저장되는 파일

```
outputs/checkpoints/
├── {model}_best.pt      ← test accuracy가 가장 높았던 시점의 모델
├── {model}_latest.pt    ← 마지막 에폭 (학습 재시작용)
└── {model}_history.json ← 에폭별 loss/accuracy 기록

outputs/logs/
└── {model}_train.log    ← 텍스트 형태의 학습 로그 (무시 가능)
```

### best vs latest

| | `_best.pt` | `_latest.pt` |
|--|-----------|------------|
| **내용** | 모델 가중치만 | 모델 + optimizer + scheduler |
| **용도** | 실제 추론, 성능 보고 | 학습 재시작(resume) |
| **optimizer 포함?** | X | O |

optimizer와 scheduler 상태를 저장하는 이유: 컴퓨터까 꺼지거나 실수로 터미널을 닫는 등 학습 중단 후 재시작할 때 이전과 같은 조건에서 이어받기 위함.

---

## 10. 실행 방법 및 결과 파일

### 학습 실행

```bash
# X-Vector 학습
python src/train.py --model xvector

# ECAPA-TDNN 학습
python src/train.py --model ecapa_tdnn

# ResNet34-SE 학습
python src/train.py --model resnet34se

# 중단된 경우 이어서 학습
python src/train.py --model xvector --resume
```

### 모델 비교 요약

| 모델 | 처리 방식 | 특징 | 파라미터 수 |
|------|---------|------|-----------|
| **X-Vector** | 1D TDNN | 가장 단순, 빠름 | 가장 적음 |
| **ECAPA-TDNN** | 1D TDNN + SE + Res2 | 어텐션, 멀티스케일 | 중간 |
| **ResNet34-SE** | 2D CNN | Mel을 이미지로 처리 | 가장 많음 |


### 학습 완료 후 확인 사항

```bash
# 학습 기록 확인
cat outputs/logs/xvector_train.log

# 최고 성능 확인
python -c "
import json
with open('outputs/checkpoints/xvector_history.json') as f:
    h = json.load(f)
best = max(h, key=lambda x: x['test_acc'])
print(f'Best epoch: {best[\"epoch\"]}, test_acc: {best[\"test_acc\"]:.4f}')
"
```

