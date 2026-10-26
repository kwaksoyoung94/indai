"""k-fold 교차검증 결과 집계. 모든 영상이 정확히 한 번 test 로 쓰이므로 예측을 합쳐 전체 성능을 낸다.

    python cv_report.py --name cnn_lstm --k 5
    python cv_report.py --name rule_ssim --k 5      # baselines.py 결과
출력: evaluation_results/cv_<name>.json, cv_<name>.md
"""
import argparse
import json

import pandas as pd

from config import Config
from dataset import manifest
from evaluation.metrics import binary_metrics, bootstrap_ci, video_level


def fmt(v):
    return '-' if v is None else f'{v:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cnn_lstm')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--threshold', type=float, default=Config.FIRE_THRESHOLD)
    a = ap.parse_args()

    cat = manifest.load().set_index('video_id').category
    parts = []
    for f in range(a.k):
        p = Config.RESULTS_DIR / f'{a.name}_fold{f}_test' / 'predictions.csv'
        if not p.exists():
            raise SystemExit(f'fold {f} 결과가 없습니다: {p}')
        d = pd.read_csv(p)
        d['fold'] = f
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df['category'] = df.video_id.map(cat)
    dup = df.groupby('video_id').fold.nunique().max()
    assert dup == 1, '한 영상이 여러 fold의 test에 등장했습니다(누수)'

    thr = a.threshold
    # 주 지표는 통제된 불꽃(벽난로, 가스 토치)을 뺀 집합. 통제된 불꽃은 따로 보고한다.
    cf = df[df.category.isin(Config.CONTROLLED_FLAME_CATEGORIES)]
    full, df = df, df[~df.category.isin(Config.CONTROLLED_FLAME_CATEGORIES)]
    pooled = binary_metrics(df.y, df.prob, thr)
    out = dict(name=a.name, k=a.k, threshold=thr, n_videos=int(df.video_id.nunique()),
               pooled_clip_level=pooled, ci95_video_bootstrap=bootstrap_ci(df, thr),
               video_level_mean=binary_metrics(*[video_level(df, 'mean')[c] for c in ('y', 'prob')], thr),
               controlled_flame=dict(videos=int(cf.video_id.nunique()), clips=len(cf),
                                     clip_alarm_rate=float((cf.prob >= thr).mean()) if len(cf) else None,
                                     video_mean_prob={v: float(g.prob.mean()) for v, g in cf.groupby('video_id')}),
               all_nonfire_pooled_clip_level=binary_metrics(full.y, full.prob, thr),
               per_fold={}, per_category={})
    for f, g in df.groupby('fold'):
        m = binary_metrics(g.y, g.prob, thr)
        out['per_fold'][int(f)] = {k: m[k] for k in ('n', 'n_pos', 'precision', 'recall', 'fpr', 'auc')}
    for (c, pos), g in full.groupby(['category', 'y']):   # 'firesense'는 fire/normal 양쪽에 있으므로 y로도 구분
        rate = float((g.prob >= thr).mean())
        key = f'{c} ({"fire" if pos else "non-fire"})'
        out['per_category'][key] = dict(label='fire' if pos else 'non-fire', videos=int(g.video_id.nunique()),
                                        clips=len(g), **({'recall': rate} if pos else {'fpr': rate}))
    (Config.RESULTS_DIR / f'cv_{a.name}.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    ci = out['ci95_video_bootstrap']
    md = [f'# {a.name} / {a.k}-fold 교차검증 (임계값 {thr})', '',
          f'주 집합: 영상 {out["n_videos"]}개, 클립 {pooled["n"]}개 (영상 단위 분할, 영상은 정확히 한 번 test). '
          f'통제된 불꽃 {out["controlled_flame"]["videos"]}개({out["controlled_flame"]["clips"]}클립)는 제외하고 '
          f'따로 보고: 클립 오경보율 {fmt(out["controlled_flame"]["clip_alarm_rate"])}', '',
          '| 지표 | 값 | 95% CI(영상 부트스트랩) |', '|---|---|---|']
    for k in ('precision', 'recall', 'f1', 'fpr', 'auc'):
        md.append(f'| {k} | {fmt(pooled[k])} | ' +
                  (f'[{ci[k][0]:.3f}, {ci[k][1]:.3f}]' if k in ci and ci[k] else '-') + ' |')
    md += ['', '## fold별', '| fold | 클립 | precision | recall | FPR | AUC |', '|---|---|---|---|---|---|']
    md += [f'| {f} | {v["n"]} | {fmt(v["precision"])} | {fmt(v["recall"])} | {fmt(v["fpr"])} | {fmt(v["auc"])} |'
           for f, v in out['per_fold'].items()]
    md += ['', '## 카테고리별 (화재=재현율, 비화재=오탐률)', '| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |',
           '|---|---|---|---|---|']
    md += [f'| {c} | {v["label"]} | {v["videos"]} | {v["clips"]} | '
           f'{fmt(v.get("recall", v.get("fpr")))} |' for c, v in sorted(out['per_category'].items())]
    (Config.RESULTS_DIR / f'cv_{a.name}.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
