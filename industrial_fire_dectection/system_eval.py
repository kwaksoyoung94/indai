"""시스템(YOLO + 추적 + CNN-LSTM) 영상 단위 평가와 단계별 기여 분리(절제).

각 영상을 실제 파이프라인과 같은 경로(후보 탐지 -> IoU 추적 -> 동적 크롭 -> 트랙별 시퀀스 분류)로
처리하고 영상 단위 '경보' 여부를 비교한다. Stage 2 모델은 그 영상이 test fold인 fold 모델을 쓰므로
학습에 쓰인 영상으로 평가하지 않는다. YOLO 탐지는 yolo_candidate_rate.py 의 캐시(conf>=0.05)를 재사용한다.

방법별 경보 규칙
  yolo_only        : 후보가 K프레임 연속 (Stage 1 단독, 기존 단일 프레임 탐지기에 해당)
  yolo+persistence : 한 트랙이 T프레임(시퀀스 길이) 이상 지속 (Stage 2 없이 지속성 필터만)
  yolo+stage2(c)   : 한 트랙에서 CNN-LSTM 화재 확률 >= 임계값이 c회 연속 (제안 시스템)
지속성 필터와 Stage 2의 효과를 분리해 보고하기 위해 둘을 따로 낸다.

    python system_eval.py --name cnn_lstm --conf 0.4
"""
import argparse
import json
from collections import deque

import cv2
import numpy as np
import pandas as pd
import torch

from config import Config
from dataset import manifest
from dataset.build_roi_clips import frame_skip_for
from dataset.data_loader import clip_to_tensor
from models.cnn_lstm import load_checkpoint
from models.tracker import IoUTracker
from models.yolo_detector import crop_roi

CONSEC = (1, 2, 3)


def run_video(path, dets, model, a):
    cap = cv2.VideoCapture(str(path))
    skip = frame_skip_for(cap)
    T = Config.SEQUENCE_LENGTH
    tracker, bufs, cons = IoUTracker(), {}, {}
    first = {'yolo_only': None, 'persistence': None, **{f'stage2_c{c}': None for c in CONSEC}}
    run_det, i, idx, max_p = 0, 0, 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip == 0:
            if i >= len(dets):
                break
            boxes = [tuple(d[:4]) for d in dets[i] if d[4] >= a.conf]
            run_det = run_det + 1 if boxes else 0
            if first['yolo_only'] is None and run_det >= a.k:
                first['yolo_only'] = i
            tracks = tracker.update(boxes)
            alive = {tid for tid, _ in tracks}
            for tid in list(bufs):
                if tid not in alive:
                    del bufs[tid]
                    cons.pop(tid, None)
            ready = []
            for tid, box in tracks:
                crop = crop_roi(frame, box)
                if crop is None:
                    continue
                b = bufs.setdefault(tid, deque(maxlen=T))
                b.append(crop)
                if len(b) == T:
                    ready.append(tid)
                    if first['persistence'] is None:
                        first['persistence'] = i
            if ready and i % a.classify_every == 0:
                x = torch.stack([clip_to_tensor(np.stack(bufs[t])[-(a.input_len or T):]) for t in ready]).to(Config.DEVICE)
                with torch.inference_mode():
                    p = torch.sigmoid(model(x).float()).cpu().numpy()
                max_p = max(max_p, float(p.max()))
                for tid, pv in zip(ready, p):
                    cons[tid] = cons.get(tid, 0) + 1 if pv >= a.thr else 0
                    for c in CONSEC:
                        key = f'stage2_c{c}'
                        if first[key] is None and cons[tid] >= c:
                            first[key] = i
            i += 1
        idx += 1
    cap.release()
    return first, max_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cnn_lstm')
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--conf', type=float, default=Config.CONF_THRESHOLD, help='운영 YOLO 임계값')
    ap.add_argument('--thr', type=float, default=Config.FIRE_THRESHOLD)
    ap.add_argument('--k', type=int, default=3, help='yolo_only 경보에 필요한 연속 후보 프레임 수')
    ap.add_argument('--classify-every', type=int, default=5, help='N 분석프레임마다 Stage 2 실행')
    ap.add_argument('--seq-len', type=int, default=Config.SEQUENCE_LENGTH,
                    help='Stage 2 입력 길이(학습 때와 동일해야 함). persistence 규칙의 T도 이 값')
    ap.add_argument('--input-len', type=int, default=None,
                    help='분류기에 넣는 마지막 프레임 수. 기본은 seq-len 전체. 버퍼와 persistence 의 T 는 seq-len 그대로(프레임 1장 대조군용)')
    ap.add_argument('--resummarize', action='store_true', help='저장된 영상별 CSV로 리포트만 재생성')
    a = ap.parse_args()
    Config.SEQUENCE_LENGTH = a.seq_len

    m = manifest.load()
    m = m[(m.split != 'excluded') & (m.fold != '')].copy()
    m['fold'] = m.fold.astype(int)
    tag = f'{a.name}_conf{a.conf}'    # 시퀀스 길이는 모델 이름(예: cnn_lstm_L10)으로 구분
    csv_path = Config.RESULTS_DIR / f'system_eval_{tag}.csv'
    if a.resummarize:                 # 영상별 결과 CSV로 리포트만 다시 만든다(카테고리 재지정 반영용)
        summarize(pd.read_csv(csv_path), tag, a)
        return
    rows = []
    for f in range(a.k_folds):
        model, _ = load_checkpoint(Config.WEIGHTS_DIR / f'{a.name}_fold{f}' / 'best.pt', Config.DEVICE)
        for r in m[m.fold == f].itertuples():
            dets = json.loads((Config.RESULTS_DIR / 'yolo_candidates' / f'{r.video_id}.json').read_text())
            first, max_p = run_video(Config.VIDEO_DIR / r.path, dets, model, a)
            rows.append(dict(video_id=r.video_id, label=r.label, category=r.category, fold=f, y=int(r.y),
                             max_prob=max_p, **{k: (None if v is None else v / Config.FPS_LIMIT)
                                                for k, v in first.items()}))
            print(f'[fold {f}] {r.video_id[:46]:46s} ' +
                  ' '.join(f'{k}={"-" if v is None else f"{v / Config.FPS_LIMIT:.1f}s"}' for k, v in first.items()),
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    summarize(df, tag, a)


def summarize(df, tag, a):
    """주 지표는 '통제된 불꽃(벽난로, 가스 토치)'을 제외한 비화재 영상의 오경보율이다.
    통제된 불꽃은 실제 화염이라 별도 행으로 보고하고, 전부 합친 값은 민감도로 함께 낸다."""
    df = df.copy()
    df['category'] = df.video_id.map(manifest.load().set_index('video_id').category)   # 재지정 반영
    methods = ['yolo_only', 'persistence'] + [f'stage2_c{c}' for c in CONSEC]
    fire, non = df[df.y == 1], df[df.y == 0]
    cf = non[non.category.isin(Config.CONTROLLED_FLAME_CATEGORIES)]
    main = non[~non.category.isin(Config.CONTROLLED_FLAME_CATEGORIES)]
    rng = np.random.default_rng(0)

    def ci(sub, mth):
        boots = [sub.sample(len(sub), replace=True, random_state=int(s))[mth].notna().mean()
                 for s in rng.integers(0, 1 << 30, 1000)]
        return [float(x) for x in np.percentile(boots, [2.5, 97.5])]

    out = {}
    lines = [f'# 시스템 단위 평가: {a.name}, YOLO conf {a.conf}, Stage2 임계값 {a.thr}', '',
             f'영상 {len(df)}개: 화재 {len(fire)}, 비화재 주 집합 {len(main)} (통제된 불꽃 {len(cf)}개 분리)', '',
             f'| 방법 | 화재 검출(재현율) | **주 오경보율** (n={len(main)}) | 95% CI | 통제된 불꽃 오경보 (n={len(cf)}) '
             f'| 전체 비화재 오경보율 (n={len(non)}) | 화재 경보까지 중앙값(초) |', '|---|---|---|---|---|---|---|']
    for mth in methods:
        rec, fpr = fire[mth].notna().mean(), main[mth].notna().mean()
        lo, hi = ci(main, mth)
        med = fire[mth].dropna().median() if fire[mth].notna().any() else float('nan')
        out[mth] = dict(recall=float(rec), fpr=float(fpr), fpr_ci=[lo, hi],
                        tp=int(fire[mth].notna().sum()), fn=int(fire[mth].isna().sum()),
                        fp=int(main[mth].notna().sum()), tn=int(main[mth].isna().sum()), median_sec=float(med),
                        controlled_flame=dict(alarms=int(cf[mth].notna().sum()), n=len(cf)),
                        fpr_all_nonfire=float(non[mth].notna().mean()), fpr_all_ci=ci(non, mth))
        lines.append(f'| {mth} | {rec:.3f} ({out[mth]["tp"]}/{len(fire)}) | **{fpr:.3f}** ({out[mth]["fp"]}/{len(main)}) '
                     f'| [{lo:.2f}, {hi:.2f}] | {int(cf[mth].notna().sum())}/{len(cf)} '
                     f'| {out[mth]["fpr_all_nonfire"]:.3f} ({int(non[mth].notna().sum())}/{len(non)}) | {med:.1f} |')
    lines += ['', '## 카테고리별 오경보 영상 수 / 영상 수 (비화재)', '| 카테고리 | ' + ' | '.join(methods) + ' |',
              '|---|' + '---|' * len(methods)]
    for c, g in non.groupby(['label', 'category']):
        flag = ' (통제된 불꽃)' if c[1] in Config.CONTROLLED_FLAME_CATEGORIES else ''
        lines.append(f'| {c[0]}/{c[1]}{flag} | ' + ' | '.join(f'{int(g[x].notna().sum())}/{len(g)}' for x in methods) + ' |')
    lines += ['', '## 화재 영상 검출 (카테고리별 검출 영상 수 / 영상 수)', '| 카테고리 | ' + ' | '.join(methods) + ' |',
              '|---|' + '---|' * len(methods)]
    for c, g in fire.groupby('category'):
        lines.append(f'| fire/{c} | ' + ' | '.join(f'{int(g[x].notna().sum())}/{len(g)}' for x in methods) + ' |')
    (Config.RESULTS_DIR / f'system_eval_{tag}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    (Config.RESULTS_DIR / f'system_eval_{tag}.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
