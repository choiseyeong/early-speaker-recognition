# 04. 평가 및 시각화 (Evaluation & Visualization)

> **파일**: `src/evaluate.py`, `src/visualize.py`
>
> 학습이 끝난 모델의 체크포인트를 불러와 테스트셋으로 정확도를 측정하고, 결과를 그래프로 시각화하는 단계이다. 이 단계는 "짧은 발화로도 화자 구분이 가능한가?"라는 프로젝트의 핵심 질문에 직접 답한다.

---

## 1. 배경 지식: 왜 평가를 별도로 분리하는가?

학습 루프 안에서도 매 에폭마다 테스트 정확도를 기록한다. 그렇다면 왜 별도의 평가 스크립트가 필요한가?

학습 중 기록되는 정확도는 **학습에 사용된 duration 조합**(RandomDurationDataset이 무작위로 선택한 조합)에서 측정된 값이다. 이 프로젝트의 핵심 질문은 "0.5초짜리 입력만 주었을 때", "1.0초짜리 입력만 주었을 때"처럼 **각 duration을 독립적으로** 평가하는 것이다. 이를 위해 별도의 평가 단계에서 duration별로 고정된 테스트셋을 구성해 모델을 평가한다.

---

## 2. 실행 순서

```
python src/evaluate.py      # 1단계: 체크포인트 로드 → duration별 정확도 측정 → JSON 저장
python src/visualize.py     # 2단계: JSON 읽어 그래프 생성 → PNG 저장
```

두 스크립트는 독립적으로 실행할 수 있다. 단, `visualize.py`는 `evaluate.py`가 생성한 `evaluation_results.json`이 있어야 실행된다.

---

## 3. evaluate.py 동작 원리

### 전체 흐름

```
체크포인트 로드 ({model}_best.pt)
        │
        ▼ duration마다 반복 (0.5 / 1.0 / 1.5 / full)
FixedDurationDataset 구성
        │
        ▼
테스트셋 전체 추론 (분류 헤드 포함)
        │
        ▼
accuracy = 맞은 수 / 전체 수
        │
        ▼ 모든 duration 완료 후
outputs/results/evaluation_results.json 저장
```

### FixedDurationDataset

학습 때 사용한 `RandomDurationDataset`과 달리, 평가 때는 한 번에 하나의 duration만 고정해서 로드한다.

```python
class FixedDurationDataset(Dataset):
    def __init__(self, split: str, duration: str):
        self.dir   = PROCESSED_DIR / split / duration  # 예: outputs/processed/test/0.5
        self.files = sorted(self.dir.glob('*.npy'))
        self.labels = np.load(PROCESSED_DIR / f'labels_{split}.npy')
```

각 샘플은 다음 세 단계를 거쳐 모델에 입력된다.

```
1. _fix_length(mel, 150)
   - T >= 150이면 앞 150프레임만 사용 (잘라냄)
   - T < 150이면 뒤를 0으로 패딩

2. Z-score 정규화
   mel = (mel - mel.mean()) / (mel.std() + 1e-8)

3. 채널 차원 추가
   mel_tensor = mel.unsqueeze(0)   → shape: (1, 80, 150)
```

이 전처리는 학습 DataLoader와 동일해야 모델이 올바르게 작동한다.

### 정확도 계산

```python
@torch.no_grad()
def evaluate_duration(model, duration, device):
    correct, total = 0, 0
    for mel, label in loader:
        logit, _ = model(mel)                      # 분류 헤드 출력
        correct += (logit.argmax(1) == label).sum()
        total   += len(label)
    return correct / total
```

`@torch.no_grad()`는 평가 중 기울기를 계산하지 않도록 한다. 메모리를 절약하고 속도를 높이기 위함이다.

### 기존 결과와 병합

`evaluate.py`는 실행할 때마다 기존 `evaluation_results.json`을 읽어 결과를 합산한 뒤 저장한다. 따라서 모델 하나씩 순차적으로 실행해도 결과가 누적된다.

```bash
python src/evaluate.py --models xvector        # xvector 결과만 저장
python src/evaluate.py --models ecapa_tdnn     # 기존 xvector 결과에 추가
python src/evaluate.py --models resnet34se     # 세 모델 결과 모두 저장
```

---

## 4. visualize.py 동작 원리

`evaluation_results.json`을 읽어 "입력 길이 vs 정확도" 꺾은선 그래프를 생성한다.

```python
DURATION_X = {'0.5': 0.5, '1.0': 1.0, '1.5': 1.5, 'full': 2.0}  # x축 위치

MODEL_STYLE = {
    'xvector'   : {'color': '#4C72B0', 'marker': 'o', 'label': 'X-Vector'},
    'ecapa_tdnn': {'color': '#DD8452', 'marker': 's', 'label': 'ECAPA-TDNN'},
    'resnet34se': {'color': '#55A868', 'marker': '^', 'label': 'ResNet34-SE'},
}
```

`full` duration은 x축에서 2.0 위치에 배치한다. 실제 평균 발화 길이는 2.51초이지만, 그래프 비례를 맞추기 위해 2.0으로 고정한다.

각 점 위에는 수치(예: `88.7%`)가 자동으로 표기된다. 그래프는 `outputs/results/accuracy_by_duration.png`에 저장된다.

---

## 5. 출력 파일 구조

```
outputs/results/
├── evaluation_results.json   ← 모델별·duration별 정확도 (소수점 4자리)
└── accuracy_by_duration.png  ← 꺾은선 그래프
```

### evaluation_results.json 구조

```json
{
  "xvector": {
    "0.5": 0.8078,
    "1.0": 0.8967,
    "1.5": 0.9202,
    "full": 0.9213
  },
  "ecapa_tdnn": {
    "0.5": 0.8865,
    "1.0": 0.9417,
    "1.5": 0.9642,
    "full": 0.9581
  },
  "resnet34se": {
    "0.5": 0.8425,
    "1.0": 0.9100,
    "1.5": 0.9427,
    "full": 0.9458
  }
}
```

---

## 6. 결과 해석

### duration별 정확도 요약

| duration | X-Vector | ECAPA-TDNN | ResNet34-SE |
|----------|----------|------------|-------------|
| 0.5초    | 80.8%    | 88.7%      | 84.3%       |
| 1.0초    | 89.7%    | 94.2%      | 91.0%       |
| 1.5초    | 92.0%    | 96.4%      | 94.3%       |
| full     | 92.1%    | 95.8%      | 94.6%       |

### 관찰 1: 발화가 짧아질수록 정확도가 내려간다

0.5초에서 1.5초로 늘어날 때 정확도가 크게 상승한다(X-Vector: +11.2%p, ECAPA-TDNN: +7.7%p). 이는 발화 초반부만으로는 화자 특성을 충분히 담기 어렵기 때문이다. 그러나 0.5초에서도 80% 이상의 정확도를 달성하며, 이는 짧은 음성으로도 화자 구분이 어느 정도 가능함을 보여준다.

### 관찰 2: 1.5초와 full의 차이가 작다

1.5초 이후로는 정확도 향상이 거의 없다(대부분 1%p 이내). 이는 화자 구분에 필요한 핵심 정보가 발화 초반 1.5초 안에 집중되어 있음을 의미한다.

### 관찰 3: ECAPA-TDNN이 전 구간에서 가장 높다

ECAPA-TDNN은 세 모델 중 0.5초·1.0초·1.5초 모두에서 가장 높은 정확도를 기록한다. Attentive Statistics Pooling과 Res2Conv 구조가 짧은 발화에서도 화자 특성을 효과적으로 추출한다는 점을 확인할 수 있다. 단, full에서는 1.5초보다 0.6%p 낮아지는데, 이는 모델이 고정 길이(150프레임 = 1.5초)로 학습되었기 때문에 full에서는 1.5초 이후 정보가 잘려 오히려 불리해지기 때문으로 해석된다.

### 관찰 4: ResNet34-SE는 X-Vector보다 높고 ECAPA-TDNN보다 낮다

2D CNN 기반 구조(ResNet34-SE)가 1D TDNN 기반 구조(X-Vector)보다 전반적으로 높은 성능을 보인다. Mel Spectrogram을 이미지처럼 다루는 접근이 유효함을 보여준다.

---

## 7. ResNet34-SE 결과가 누락된 경우

`evaluation_results.json`에 `resnet34se` 항목이 없다면 `evaluate.py`가 해당 모델에 대해 실행되지 않은 것이다. 다음 명령으로 해결한다.

```bash
python src/evaluate.py --models resnet34se
```

`resnet34se_best.pt` 체크포인트가 `outputs/checkpoints/`에 존재하면 자동으로 로드된다. 결과는 기존 JSON에 추가 저장되므로 다른 모델의 결과는 덮어쓰지 않는다.

---

## 8. 실행 방법

```bash
# 프로젝트 루트에서 실행
python src/evaluate.py              # 체크포인트 있는 모델 전체 평가
python src/visualize.py             # 그래프 생성
```

**예상 출력 (evaluate.py)**:
```
평가 대상 모델: ['xvector', 'ecapa_tdnn', 'resnet34se']
디바이스: cuda

==================================================
[xvector]  (학습 best acc: 0.9213)
  0.5s   → accuracy: 0.8078 (80.8%)
  1.0s   → accuracy: 0.8967 (89.7%)
  1.5s   → accuracy: 0.9202 (92.0%)
  full   → accuracy: 0.9213 (92.1%)
...

결과 저장: outputs/results/evaluation_results.json
```

**예상 출력 (visualize.py)**:
```
그래프 저장: outputs/results/accuracy_by_duration.png
```

> **주의**: `evaluate.py`는 학습이 완료된 모델 체크포인트(`_best.pt`)가 `outputs/checkpoints/`에 있어야 실행된다. 학습을 먼저 완료한 뒤 이 단계를 실행한다.

이전 단계: ← [모델 학습 (README_03_TRAIN.md)](README_03_TRAIN.md)
