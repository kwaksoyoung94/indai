"""논문 그림 생성. 모든 그림은 실제 ROI 클립과 실측 결과(JSON/CSV)에서만 만든다. (합성/상수 데이터 없음)

    python generate_report_images.py
입력: evaluation_results/ 의 system_eval_*.json/csv, benchmark_*.json, cv_*.json, dataset/roi_index.csv
출력: evaluation_results/figures/*.png
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from models.features import ssim_series

R = Config.RESULTS_DIR
METHODS = [('yolo_only', 'Stage 1 단독'), ('persistence', '+ 지속성 필터'), ('stage2_c1', '+ Stage 2 (제안)')]
COLORS = ['#9aa5b1', '#6fa8dc', '#d9534f']


def setup_font():
    plt.rcParams['font.family'] = 'DejaVu Sans' if sys.platform.startswith('linux') else 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def ssim_chart(out, n_per_class=60):
    """실제 클립의 프레임 간 SSIM 시계열(클래스별 평균 ± 표준편차)."""
    idx = pd.read_csv(Config.ROI_INDEX)
    rng = np.random.default_rng(Config.SEED)
    fig, ax = plt.subplots(figsize=(8, 5))
    for y, name, color in ((0, '비화재(외란 등)', 'tab:blue'), (1, '화재', 'tab:red')):
        sub = idx[idx.y == y]
        pick = sub.iloc[rng.choice(len(sub), size=min(n_per_class, len(sub)), replace=False)]
        series = np.stack([ssim_series(np.load(Config.ROI_DIR / p)) for p in pick.clip_path])
        x = np.arange(1, series.shape[1] + 1)
        m, s = series.mean(0), series.std(0)
        ax.plot(x, m, color=color, label=f'{name} (클립 {len(pick)}개)')
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
    ax.set_xlabel('프레임 인덱스 (10 FPS)')
    ax.set_ylabel('연속 프레임 SSIM')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.6)
    save(fig, out)


def system_chart(out, name='cnn_lstm'):
    """방법별 재현율과 오경보율(95% CI)."""
    j = json.load(open(R / f'system_eval_{name}_conf0.4.json', encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(METHODS))
    fpr = [j[m]['fpr'] for m, _ in METHODS]
    err = np.array([[j[m]['fpr'] - j[m]['fpr_ci'][0], j[m]['fpr_ci'][1] - j[m]['fpr']] for m, _ in METHODS]).T
    rec = [j[m]['recall'] for m, _ in METHODS]
    ax.bar(x - 0.2, fpr, 0.4, yerr=err, capsize=4, color='#d9534f', label='비화재 영상 오경보율 (통제된 불꽃 제외, 95% CI)')
    ax.bar(x + 0.2, rec, 0.4, color='#5cb85c', label='화재 영상 검출률')
    for i, (f, r) in enumerate(zip(fpr, rec)):
        ax.text(i - 0.2, f + 0.03 + err[1][i], f'{f:.0%}', ha='center', fontsize=9)
        ax.text(i + 0.2, r + 0.01, f'{r:.0%}', ha='center', fontsize=9)
    ax.set_xticks(x, [n for _, n in METHODS])
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper center', ncol=2, fontsize=9)
    ax.grid(axis='y', ls=':', alpha=0.5)
    save(fig, out)


def category_chart(out, name='cnn_lstm'):
    """비화재 카테고리별 오경보 영상 수(막대 끝에 k/n 표기, 0도 표시)."""
    df = pd.read_csv(R / f'system_eval_{name}_conf0.4.csv')
    df['category'] = df.video_id.map(pd.read_csv(Config.MANIFEST, keep_default_na=False).set_index('video_id').category)
    df = df[df.y == 0]
    cf = Config.CONTROLLED_FLAME_CATEGORIES
    df['cat'] = df.label + '/' + df.category + np.where(df.category.isin(cf), ' *', '')
    cats = sorted(df.cat.unique())
    fig, ax = plt.subplots(figsize=(9, 6.5))
    h = 0.26
    for k, ((m, nm), c) in enumerate(zip(METHODS, COLORS)):
        for i, cat in enumerate(cats):
            g = df[df.cat == cat]
            hit, n = int(g[m].notna().sum()), len(g)
            y = i + (k - 1) * h
            ax.barh(y, max(hit / n, 0.008), h, color=c, label=nm if i == 0 else None)
            ax.text(hit / n + 0.01, y, f'{hit}/{n}', va='center', fontsize=7)
    ax.set_yticks(np.arange(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlabel('오경보가 난 영상 수 / 영상 수  (* 통제된 불꽃: 주 지표에서 분리)')
    ax.set_xlim(0, 1.15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.grid(axis='x', ls=':', alpha=0.5)
    save(fig, out)


def periodicity_chart(out):
    """SSIM 시계열의 주기성 지표를 화재/비화재 영상 단위로 비교(핵심 가설의 실증 검증).

    peak_ratio: 스펙트럼에서 가장 큰 성분의 비중(클수록 한 주파수에 몰린 주기적 신호)
    entropy   : 정규화 스펙트럼 엔트로피(클수록 광대역/불규칙)
    클립이 아닌 영상 단위로 평균해 검정한다(같은 영상의 클립은 독립이 아님)."""
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    from models.features import FEATURE_NAMES, clip_features

    cache = R / 'clip_features.csv'
    idx = pd.read_csv(Config.ROI_INDEX)
    if cache.exists():
        feats = pd.read_csv(cache)
    else:
        X = np.stack([clip_features(np.load(Config.ROI_DIR / p)) for p in idx.clip_path])
        feats = pd.DataFrame(X, columns=FEATURE_NAMES)
        feats.insert(0, 'clip_path', idx.clip_path.values)
        feats.to_csv(cache, index=False)
    d = idx.merge(feats, on='clip_path')
    cat = pd.read_csv(Config.MANIFEST, keep_default_na=False).set_index('video_id').category
    v = d.groupby('video_id')[FEATURE_NAMES].mean()
    v['y'] = d.groupby('video_id').y.first()
    v['grp'] = [('fire' if y else f'non-fire/{cat[i]}') for i, y in zip(v.index, v.y)]

    stats = {}
    for f in FEATURE_NAMES:
        a, b = v[v.y == 1][f], v[v.y == 0][f]
        stats[f] = dict(fire_median=float(a.median()), nonfire_median=float(b.median()),
                        auc_video_level=float(roc_auc_score(v.y, v[f])),
                        mannwhitney_p=float(mannwhitneyu(a, b).pvalue), n_fire=len(a), n_nonfire=len(b))
    (R / 'periodicity_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')

    groups = ['fire'] + sorted(g for g in v.grp.unique() if g != 'fire')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1.3, 1.3, 1.6]})
    for ax, f, ttl in ((axes[0], 'ssim_peak_ratio', '스펙트럼 최대 성분 비중 (클수록 주기적)'),
                       (axes[1], 'ssim_spec_entropy', '스펙트럼 엔트로피 (클수록 불규칙)')):
        data = [v[v.grp == g][f].values for g in groups]
        ax.boxplot(data, vert=False, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='#dfe7f1'), medianprops=dict(color='k'))
        for i, g in enumerate(data):
            ax.scatter(g, np.full(len(g), i + 1) + np.random.default_rng(0).uniform(-0.12, 0.12, len(g)),
                       s=14, color='#d9534f' if groups[i] == 'fire' else '#337ab7', zorder=3)
        ax.set_yticks(np.arange(1, len(groups) + 1), [f'{g} (n={len(d_)})' for g, d_ in zip(groups, data)],
                      fontsize=8)
        ax.set_title(f'{ttl}\n영상 단위 AUC={stats[f]["auc_video_level"]:.2f}, p={stats[f]["mannwhitney_p"]:.2f}',
                     fontsize=9)
        ax.grid(axis='x', ls=':', alpha=0.5)
    # 예시: 그룹별로 peak_ratio 가 중앙값에 가장 가까운 클립 하나의 SSIM 시계열
    ex = axes[2]
    for g, col in (('fire', '#d9534f'), ('non-fire/strobe', '#337ab7'), ('non-fire/welding', '#5cb85c')):
        vids = v[v.grp == g].index
        sub = d[d.video_id.isin(vids)]
        if sub.empty:
            continue
        row = sub.iloc[(sub.ssim_peak_ratio - sub.ssim_peak_ratio.median()).abs().argmin()]
        ex.plot(np.arange(1, 30), ssim_series(np.load(Config.ROI_DIR / row.clip_path)), color=col, label=g)
    ex.set_title('예시 클립의 SSIM 시계열 (그룹 중앙값에 가장 가까운 클립)', fontsize=9)
    ex.set_xlabel('프레임 인덱스 (10 FPS)')
    ex.set_ylabel('연속 프레임 SSIM')
    ex.legend(fontsize=8)
    ex.grid(True, ls=':', alpha=0.5)
    save(fig, out)


def seqlen_chart(out):
    """시퀀스 길이별 오경보율과 경보까지 걸리는 시간의 trade-off."""
    runs = [('L=10', 'cnn_lstm_L10', 10), ('L=20', 'cnn_lstm_L20', 20), ('L=30', 'cnn_lstm', 30)]
    offsets = {'L=10': (10, 10), 'L=20': (-85, 14), 'L=30': (12, -34)}      # 라벨 겹침 방지
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for lab, n, L in runs:
        p = R / f'system_eval_{n}_conf0.4.json'
        if not p.exists():
            continue
        r = json.load(open(p, encoding='utf-8'))['stage2_c1']
        ax.errorbar(r['median_sec'], r['fpr'], yerr=[[r['fpr'] - r['fpr_ci'][0]], [r['fpr_ci'][1] - r['fpr']]],
                    fmt='o', capsize=4, markersize=8)
        ax.annotate(f'{lab}\n재현율 {r["recall"]:.0%}', (r['median_sec'], r['fpr']), textcoords='offset points',
                    xytext=offsets[lab], fontsize=9)
    ax.set_xlabel('화재 경보까지 걸린 시간 (중앙값, 초)')
    ax.set_ylabel('비화재 영상 오경보율')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 0.6)
    ax.grid(True, ls=':', alpha=0.5)
    save(fig, out)


def examples_chart(out):
    """파이프라인 검출 예시. 각 영상의 Stage 2 판정 구간 중앙에 가장 가까운 저장 프레임을 사용한다.
    성공(화재 검출, 외란 필터링)과 실패(용접 오경보) 사례를 함께 싣는다."""
    import cv2
    base = R / 'figures' / 'examples'
    cases = [('fire_building', '화재 영상 (정탐)'), ('strobe', '스트로브 (정상 필터링)'),
             ('glass_reflection', '유리 반사광 (정상 필터링)'), ('welding', '가스 토치 불꽃 (오경보 사례, 실제 화염)')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (d, ttl) in zip(axes.ravel(), cases):
        j = json.load(open(base / d / 'run.json', encoding='utf-8'))
        ev = j['events']
        frames = sorted(int(p.stem[6:]) for p in (base / d).glob('frame_*.jpg'))
        mid = np.median([e['frame'] for e in ev]) if ev else frames[len(frames) // 2]
        pick = min(frames, key=lambda f: abs(f - mid))
        img = cv2.cvtColor(cv2.imread(str(base / d / f'frame_{pick:06d}.jpg')), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        n = len(ev)
        nf = sum(e['is_fire'] for e in ev)
        ax.set_title(f'{ttl}\n화재 판정 윈도우 {nf}/{n}', fontsize=10)
        ax.axis('off')
    save(fig, out)


def latency_chart(out):
    """pipeline 실행 시 실측한 단계별 평균 지연시간."""
    files = sorted(glob.glob(str(R / 'benchmark_*.json')))
    comp = [('stage1_detect_ms', 'Stage 1 탐지'), ('track_crop_ms', '추적/크롭'),
            ('stage2_classify_ms', 'Stage 2 분류'), ('total_ms', '프레임당 합계')]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    w = 0.2
    for i, (key, nm) in enumerate(comp):
        vals = []
        for f in files:
            t = json.load(open(f, encoding='utf-8'))['summary']['timing_ms']
            vals.append(t[key]['mean'] if key in t else 0)
        ax.bar(np.arange(len(files)) + (i - 1.5) * w, vals, w, label=nm)
    ax.set_xticks(np.arange(len(files)), [os.path.basename(f)[10:-5] for f in files])
    ax.set_ylabel('평균 지연시간 (ms)')
    ax.legend()
    ax.grid(axis='y', ls=':', alpha=0.5)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(R / 'figures'))
    a = ap.parse_args()
    setup_font()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fn, name in ((ssim_chart, 'ssim_dynamic_analysis'), (periodicity_chart, 'periodicity_analysis'),
                     (system_chart, 'system_comparison'),
                     (category_chart, 'category_false_alarm'), (seqlen_chart, 'seqlen_tradeoff'),
                     (latency_chart, 'latency_breakdown'), (examples_chart, 'detection_examples')):
        try:
            fn(out / f'{name}.png')
            print('저장:', name)
        except FileNotFoundError as e:
            print(f'[건너뜀] {name}: {e}')
    print(f'그림 폴더: {out}')


if __name__ == '__main__':
    main()
