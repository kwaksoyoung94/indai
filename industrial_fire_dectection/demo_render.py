"""시연용 영상 생성: 같은 영상을 왼쪽 '탐지 단계 단독', 오른쪽 '제안(탐지 + 검증)'으로 나란히 처리해 보여 준다.

평가와 같은 조건을 쓴다: YOLO 후보는 yolo_candidate_rate.py 가 저장한 캐시(conf >= 0.4), 검증 모델은 그 영상이 시험 fold 인
fold 모델이라 학습에 쓰인 영상이 아니다. 경보 규칙도 system_eval.py 와 같다.
  탐지 단계 단독 : 후보가 3 분석프레임(10 FPS 기준 0.3초) 연속 검출되면 경보
  제안           : 한 트랙에 ROI 30프레임이 쌓이고 CNN-LSTM 화재 확률이 0.75 이상이면 경보 (5 분석프레임마다 판정)

    python demo_render.py --video fire__building__BOxpCa7mH1o --end 20 --out 시연/1_화재.mp4
"""
import argparse
import json
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from config import Config
from dataset import manifest
from dataset.build_roi_clips import frame_skip_for
from dataset.data_loader import clip_to_tensor
from models.cnn_lstm import load_checkpoint
from models.tracker import IoUTracker
from models.yolo_detector import crop_roi

W, H = 1280, 720
HEAD, FOOT = 84, 92
FONT = r'C:\Windows\Fonts\malgunbd.ttf'
FONT_R = r'C:\Windows\Fonts\malgun.ttf'
NAVY, RED, BLUE, GRAY, GREEN = (23, 55, 94), (217, 72, 15), (31, 90, 158), (140, 140, 140), (34, 139, 34)


def font(size, bold=True):
    return ImageFont.truetype(FONT if bold else FONT_R, size)


def draw_text(img, items):
    """items: [(xy, text, size, rgb, bold)] 를 한 번에 그린다(OpenCV 는 한글을 못 그리므로 PIL 사용)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for xy, t, s, c, b in items:
        d.text(xy, t, font=font(s, b), fill=c)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def fit(frame, bw, bh):
    h, w = frame.shape[:2]
    s = min(bw / w, bh / h)
    return s, cv2.resize(frame, (int(w * s), int(h * s)))


def place(canvas, panel, x0, y0, bw, bh):
    h, w = panel.shape[:2]
    ox, oy = x0 + (bw - w) // 2, y0 + (bh - h) // 2
    canvas[oy:oy + h, ox:ox + w] = panel
    return ox, oy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='manifest 의 video_id')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=20.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conf', type=float, default=Config.CONF_THRESHOLD)
    ap.add_argument('--thr', type=float, default=Config.FIRE_THRESHOLD)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--classify-every', type=int, default=5)
    ap.add_argument('--title', default='')
    a = ap.parse_args()

    m = manifest.load().set_index('video_id')
    row = m.loc[a.video]
    fold = int(row.fold)
    model, _ = load_checkpoint(Config.WEIGHTS_DIR / f'cnn_lstm_fold{fold}' / 'best.pt', Config.DEVICE)
    dets = json.loads((Config.RESULTS_DIR / 'yolo_candidates' / f'{a.video}.json').read_text())
    cap = cv2.VideoCapture(str(Config.VIDEO_DIR / row.path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = frame_skip_for(cap)
    T = Config.SEQUENCE_LENGTH
    bw, bh = W // 2 - 20, H - HEAD - FOOT - 10
    out_fps = min(src_fps, 30.0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    vw = cv2.VideoWriter(a.out, fourcc, out_fps, (W, H))

    tracker, bufs, cons = IoUTracker(), {}, {}
    run_det = i = idx = 0
    boxes, tracks_state = [], {}           # 마지막 분석 프레임의 결과를 다음 분석 전까지 그대로 그린다
    alarm_l = alarm_r = None
    max_p = 0.0
    out_every = max(1, int(round(src_fps / out_fps)))
    n_written = 0
    while True:
        ok, frame = cap.read()
        t = idx / src_fps
        if not ok or t > a.end:
            break
        if idx % skip == 0:
            if i >= len(dets):
                break
            boxes = [tuple(d[:4]) for d in dets[i] if d[4] >= a.conf]
            run_det = run_det + 1 if boxes else 0
            if alarm_l is None and run_det >= a.k:
                alarm_l = t
            tracks = tracker.update(boxes)
            alive = {tid for tid, _ in tracks}
            for tid in list(bufs):
                if tid not in alive:
                    del bufs[tid]
                    cons.pop(tid, None)
            tracks_state = {tid: dict(box=box, n=len(bufs.get(tid, [])), p=None) for tid, box in tracks}
            ready = []
            for tid, box in tracks:
                crop = crop_roi(frame, box)
                if crop is None:
                    continue
                b = bufs.setdefault(tid, deque(maxlen=T))
                b.append(crop)
                tracks_state[tid]['n'] = len(b)
                if len(b) == T:
                    ready.append(tid)
            if ready and i % a.classify_every == 0:
                x = torch.stack([clip_to_tensor(np.stack(bufs[tid])) for tid in ready]).to(Config.DEVICE)
                with torch.inference_mode():
                    p = torch.sigmoid(model(x).float()).cpu().numpy()
                for tid, pv in zip(ready, p):
                    cons[tid] = pv
                    max_p = max(max_p, float(pv))
                    if alarm_r is None and pv >= a.thr:
                        alarm_r = t
            for tid in ready:
                if tid in cons:
                    tracks_state[tid]['p'] = float(cons[tid])
            i += 1
        idx += 1
        if t < a.start or (idx - 1) % out_every:
            continue

        canvas = np.full((H, W, 3), 255, np.uint8)
        canvas[:HEAD] = NAVY[::-1]
        canvas[H - FOOT:] = (245, 245, 245)
        s, left = fit(frame, bw, bh)
        right = left.copy()
        for bx in boxes:                                     # 왼쪽: 탐지 단계가 찾은 후보를 모두 표시
            x1, y1, x2, y2 = [int(v * s) for v in bx]
            cv2.rectangle(left, (x1, y1), (x2, y2), (0, 140, 255), 3)
        labels = []
        for tid, st in tracks_state.items():                 # 오른쪽: 트랙별 검증 상태
            x1, y1, x2, y2 = [int(v * s) for v in st['box']]
            if st['n'] < T:
                col, txt = GRAY, f'수집 중 {st["n"]}/{T}'
            elif st['p'] is not None and st['p'] >= a.thr:
                col, txt = RED, f'화재 {st["p"]:.2f}'
            else:
                col, txt = BLUE, f'비화재 {st["p"]:.2f}' if st['p'] is not None else '판정 중'
            cv2.rectangle(right, (x1, y1), (x2, y2), col[::-1], 3)
            ty = max(0, y1 - 34)
            tw = int(font(22).getlength(txt)) + 14
            cv2.rectangle(right, (x1, ty), (x1 + tw, ty + 32), col[::-1], -1)     # 글자가 배경에 묻히지 않게 색 띠를 깐다
            labels.append(((x1 + 7, ty + 1), txt, 22, (255, 255, 255), True))
        lx, ly = place(canvas, left, 10, HEAD + 5, bw, bh)
        rx, ry = place(canvas, right, W // 2 + 10, HEAD + 5, bw, bh)
        canvas = draw_text(canvas, [((24, 18), a.title or a.video, 34, (255, 255, 255), True),
                                    ((W - 190, 26), f'{t - a.start:5.1f} s', 30, (255, 255, 255), True),
                                    ((24, H - FOOT + 10), '탐지 단계 단독', 26, NAVY, True),
                                    ((W // 2 + 24, H - FOOT + 10), '제안: 탐지 + 검증', 26, NAVY, True)] +
                            [((rx + xy[0], ry + xy[1]), tx, sz, c, b) for xy, tx, sz, c, b in labels])
        stat_l = (f'경보 ({alarm_l:.1f}초)', RED) if alarm_l is not None and t >= alarm_l else ('경보 없음', GREEN)
        stat_r = (f'경보 ({alarm_r:.1f}초)', RED) if alarm_r is not None and t >= alarm_r else ('경보 없음', GREEN)
        canvas = draw_text(canvas, [((24, H - FOOT + 44), stat_l[0], 34, stat_l[1], True),
                                    ((W // 2 + 24, H - FOOT + 44), stat_r[0], 34, stat_r[1], True)])
        cv2.line(canvas, (W // 2, HEAD), (W // 2, H - FOOT), (200, 200, 200), 2)
        vw.write(canvas)
        n_written += 1
    cap.release()
    vw.release()
    print(f'{a.out}: {n_written}프레임, 탐지 단계 경보={alarm_l}, 제안 경보={alarm_r}, 최대 확률={max_p:.2f}')


if __name__ == '__main__':
    main()
