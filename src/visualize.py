"""
visualize.py

evaluation_results.json 을 읽어 "duration vs accuracy" 그래프 생성.
그래프는 outputs/results/accuracy_by_duration.png 에 저장.

실행:
    python src/visualize.py
"""

import json
import matplotlib
matplotlib.use('Agg')   # 화면 없이 파일로 저장
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'results'
RESULTS_PATH = RESULTS_DIR / 'evaluation_results.json'
OUT_PATH     = RESULTS_DIR / 'accuracy_by_duration.png'

DURATION_LABELS = {'0.5': '0.5s', '1.0': '1.0s', '1.5': '1.5s', 'full': 'full'}
DURATION_X      = {'0.5': 0.5, '1.0': 1.0, '1.5': 1.5, 'full': 3.0}  # x축 위치

MODEL_STYLE = {
    'xvector'   : {'color': '#4C72B0', 'marker': 'o', 'label': 'X-Vector'},
    'ecapa_tdnn': {'color': '#DD8452', 'marker': 's', 'label': 'ECAPA-TDNN'},
    'resnet34se': {'color': '#55A868', 'marker': '^', 'label': 'ResNet34-SE'},
}


def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f'결과 파일 없음: {RESULTS_PATH}\n먼저 python src/evaluate.py 를 실행하세요.')
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot(results: dict):
    durations = ['0.5', '1.0', '1.5', 'full']
    x_vals    = [DURATION_X[d] for d in durations]
    x_labels  = [DURATION_LABELS[d] for d in durations]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, res in results.items():
        style = MODEL_STYLE.get(model_name, {'color': 'gray', 'marker': 'x', 'label': model_name})
        y_vals = [res.get(d, np.nan) * 100 for d in durations]

        ax.plot(x_vals, y_vals,
                color=style['color'],
                marker=style['marker'],
                label=style['label'],
                linewidth=2,
                markersize=7)

        # 각 점에 수치 표기
        for x, y in zip(x_vals, y_vals):
            if not np.isnan(y):
                ax.annotate(f'{y:.1f}%',
                            xy=(x, y),
                            xytext=(0, 8),
                            textcoords='offset points',
                            ha='center',
                            fontsize=8,
                            color=style['color'])

    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Input Duration', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Early Speaker Recognition\nAccuracy by Input Duration', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f'그래프 저장: {OUT_PATH}')
    plt.close(fig)


if __name__ == '__main__':
    results = load_results()
    plot(results)
