"""Stage 1(YOLO)이 영상에서 후보를 얼마나 내는지 임계값별로 실측한다. ('YOLO 단독' 기준선)

모든 영상을 conf=0.05로 한 번만 추론해 박스별 신뢰도를 저장하고, 임계값별 지표는 그 결과에서 계산한다.
영상 단위 경보 규칙: 분석 프레임(10FPS)에서 후보가 K프레임 연속 나오면 '경보'(기본 K=3).
  - 화재 영상이 경보 -> 재현율(검출), 비화재 영상(interference/normal)이 경보 -> 오탐
YOLO는 D-Fire로만 학습했고 이 영상들은 학습에 쓰이지 않았으므로 누수가 없다.

    python yolo_candidate_rate.py
"""
import argparse
import json

import cv2
import pandas as pd
from tqdm import tqdm

from config import Config
from dataset import manifest
from dataset.build_roi_clips import frame_skip_for

THRESHOLDS = (0.1, 0.15, 0.25, 0.4, 0.5)


def detect_video(model, path, floor):
    cap = cv2.VideoCapture(str(path))
    skip, idx, frames = frame_skip_for(cap), 0, []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip == 0:
            r = model.predict(frame, conf=floor, device=Config.DEVICE, verbose=False)[0]
            frames.append([[*map(int, b), float(c), int(k)] for b, c, k in
                           zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())])
        idx += 1
    cap.release()
    return frames


def max_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--floor', type=float, default=0.05)
    ap.add_argument('--k', type=int, default=3, help='경보에 필요한 연속 후보 프레임 수')
    ap.add_argument('--fire-class', type=int, default=1, help='D-Fire: 0=smoke, 1=fire (-1이면 모든 클래스)')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(Config.YOLO_MODEL_PATH))
    m = manifest.load()
    m = m[m.split != 'excluded']
    out_dir = Config.RESULTS_DIR / 'yolo_candidates'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in tqdm(list(m.itertuples()), desc='videos'):
        cache = out_dir / f'{r.video_id}.json'
        if cache.exists():
            frames = json.loads(cache.read_text())
        else:
            frames = detect_video(model, Config.VIDEO_DIR / r.path, a.floor)
            cache.write_text(json.dumps(frames))
        for t in THRESHOLDS:
            flags = [any(d[4] >= t and (a.fire_class < 0 or d[5] == a.fire_class) for d in f) for f in frames]
            rows.append(dict(video_id=r.video_id, label=r.label, category=r.category, thr=t,
                             frames=len(frames), frac_frames=sum(flags) / max(len(frames), 1),
                             max_run=max_run(flags), alarm=int(max_run(flags) >= a.k)))
    df = pd.DataFrame(rows)
    df.to_csv(Config.RESULTS_DIR / 'yolo_candidate_rate.csv', index=False)

    pos = df[df.label == 'fire'].groupby('thr').alarm.mean()
    neg = df[df.label != 'fire'].groupby('thr').alarm.mean()
    intf = df[df.label == 'interference'].groupby('thr').alarm.mean()
    nor = df[df.label == 'normal'].groupby('thr').alarm.mean()
    summ = pd.DataFrame({'fire 영상 경보율(재현율)': pos, '비화재 영상 경보율(오탐)': neg,
                         '  외란만': intf, '  normal만': nor})
    print(f'\n영상 단위 경보율 (후보 {a.k}프레임 연속, 화재 클래스={a.fire_class})')
    print(summ.round(3).to_string())
    cat = df[df.thr.isin([0.15, 0.4])].pivot_table(index=['label', 'category'], columns='thr', values='alarm',
                                                    aggfunc=['sum', 'count'])
    print('\n카테고리별 경보 영상 수 / 영상 수 (thr 0.15, 0.4)')
    print(cat.to_string())
    summ.to_csv(Config.RESULTS_DIR / 'yolo_candidate_summary.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
