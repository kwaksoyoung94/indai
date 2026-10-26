"""시스템 구조도(논문/발표자료용) 생성. 실제 구현(pipeline_manager.py, system_eval.py)의 단계와 수치를 그대로 반영한다.

    python make_diagram.py   ->  evaluation_results/figures/pipeline_overview_wide.png, pipeline_overview_compact.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import Config

OUT = Config.RESULTS_DIR / 'figures'
plt.rcParams['font.family'] = 'DejaVu Sans' if sys.platform.startswith('linux') else 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BLUE, ORANGE, RED, GREEN, GRAY = '#dbeaf6', '#fcd5a5', '#f8c9c9', '#cfe8c6', '#e6e6e6'


def box(ax, x, y, w, h, text, color, fs):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=color, ec='#444444', lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold', linespacing=1.25)


def arrow(ax, p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=0.9, color='#333333',
                                 connectionstyle=f'arc3,rad={rad}'))


def wide():
    fig, ax = plt.subplots(figsize=(13, 3.3))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.3)
    ax.axis('off')
    steps = [
        ('CCTV 영상\n10 FPS 샘플링', BLUE, '1. 입력'),
        ('탐지 단계\nYOLOv8n\n후보 탐지 (conf 0.4)', ORANGE, '2. 후보 탐지'),
        ('IoU 추적\n동적 크롭 (ROI)', ORANGE, '3. ROI 추적과 크롭'),
        ('지속성 확인\n동일 트랙\n30프레임 누적', GRAY, '4. 시퀀스 구성'),
        ('검증 단계\nCNN-LSTM\n외형 검증', GREEN, '5. 후보 검증'),
        ('화재 확률\n≥ 0.75 이면\n화재 경보', RED, '6. 최종 판정'),
    ]
    w, h, gap = 1.85, 1.45, 0.38
    x0 = (13 - (6 * w + 5 * gap)) / 2
    for i, (t, c, cap) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, 0.85, w, h, t, c, 10)
        ax.text(x + w / 2, 2.62, cap, ha='center', fontsize=10, color='#c00000', fontweight='bold')
        if i:
            arrow(ax, (x - gap + 0.04, 1.575), (x - 0.04, 1.575))
    ax.text(6.5, 0.38, '탐지 단계가 통과시킨 후보 중 3초 분량인 30프레임 이상 지속되는 트랙만 검증 단계로 전달하여, '
                       '점멸과 일시적 광원을 걸러내고 연산량을 줄인다.', ha='center', fontsize=9.5, color='#333333')
    fig.savefig(OUT / 'pipeline_overview_wide.png', dpi=250, bbox_inches='tight')
    plt.close(fig)


def compact():
    """2단 논문의 한 단 너비(약 8.5cm)에 맞춘 2행 구성."""
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 2.3)
    ax.axis('off')
    row1 = [('CCTV 영상\n10 FPS', BLUE), ('탐지 단계\nYOLOv8n\n후보 탐지', ORANGE), ('IoU 추적\n동적 크롭', ORANGE)]
    row2 = [('30프레임\n시퀀스 구성', GRAY), ('검증 단계\nCNN-LSTM\n외형 검증', GREEN), ('화재 확률\n≥ 0.75\n경보', RED)]
    w, h, gap = 1.12, 0.78, 0.32
    for r, (row, y) in enumerate(((row1, 1.38), (row2, 0.2))):
        for i, (t, c) in enumerate(row):
            x = 0.12 + i * (w + gap)
            box(ax, x, y, w, h, t, c, 6.6)
            if i:
                arrow(ax, (x - gap + 0.03, y + h / 2), (x - 0.03, y + h / 2))
    # 1행 끝 -> 2행 처음: 박스 사이 빈 공간(y=1.18)으로 꺾어 이동
    xe, xs = 0.12 + 2 * (w + gap) + w / 2, 0.12 + w / 2
    ax.plot([xe, xe, xs], [1.38, 1.18, 1.18], color='#333333', lw=0.9, solid_capstyle='butt')
    arrow(ax, (xs, 1.18), (xs, 0.98))
    fig.savefig(OUT / 'pipeline_overview_compact.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    wide()
    compact()
    print('저장:', OUT)
