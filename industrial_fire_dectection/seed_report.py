"""시드 반복 결과 집계: 같은 설정을 시드만 바꿔 반복한 system_eval 결과의 평균과 표준편차.

    python seed_report.py --names cnn_lstm cnn_lstm_seed1 cnn_lstm_seed2
출력: evaluation_results/seeds_summary.json, seeds_summary.md
"""
import argparse
import json

import numpy as np

from config import Config

METHODS = ['yolo_only', 'persistence', 'stage2_c1', 'stage2_c2', 'stage2_c3']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', nargs='+', default=['cnn_lstm', 'cnn_lstm_seed1', 'cnn_lstm_seed2'])
    ap.add_argument('--conf', default='0.4')
    a = ap.parse_args()

    runs = {n: json.load(open(Config.RESULTS_DIR / f'system_eval_{n}_conf{a.conf}.json', encoding='utf-8'))
            for n in a.names}
    out, lines = {}, [f'# 시드 반복 ({len(a.names)}개 시드, YOLO conf {a.conf})', '',
                      '| 방법 | 재현율 평균±표준편차 | 주 오경보율 평균±표준편차 | 시드별 오경보율 | 시드별 재현율 |',
                      '|---|---|---|---|---|']
    for m in METHODS:
        fpr = np.array([r[m]['fpr'] for r in runs.values()])
        rec = np.array([r[m]['recall'] for r in runs.values()])
        out[m] = dict(fpr_mean=float(fpr.mean()), fpr_std=float(fpr.std(ddof=1)), fpr_per_seed=fpr.tolist(),
                      recall_mean=float(rec.mean()), recall_std=float(rec.std(ddof=1)), recall_per_seed=rec.tolist())
        lines.append(f'| {m} | {rec.mean():.3f} ± {rec.std(ddof=1):.3f} | {fpr.mean():.3f} ± {fpr.std(ddof=1):.3f} | '
                     + ', '.join(f'{x:.3f}' for x in fpr) + ' | ' + ', '.join(f'{x:.3f}' for x in rec) + ' |')
    lines += ['', '(표준편차는 시드 간 표본 표준편차. 영상 수가 적어 시드별 값 자체도 크게 흔들릴 수 있다.)']
    (Config.RESULTS_DIR / 'seeds_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    (Config.RESULTS_DIR / 'seeds_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
