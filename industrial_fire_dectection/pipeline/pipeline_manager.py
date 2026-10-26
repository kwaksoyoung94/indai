import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from config import Config
from dataset.data_loader import clip_to_tensor
from models.cnn_lstm import load_checkpoint
from models.tracker import IoUTracker
from models.yolo_detector import crop_roi


def _sync(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


class FireDetectionPipeline:
    """2-Stage 파이프라인: 후보 탐지 -> ROI 추적/동적 크롭 -> 트랙별 시퀀스 CNN-LSTM 검증.

    - 시퀀스 버퍼는 트랙(객체)별로 분리되어 서로 다른 객체의 ROI가 섞이지 않는다.
    - 단계별 지연시간을 실측해 summary 로 반환한다. (고정 문자열/상수 수치 없음)
    """

    def __init__(self, detector, checkpoint=None, threshold=None, device=None):
        self.device = device or Config.DEVICE
        self.detector = detector
        self.model, ck = load_checkpoint(checkpoint or Config.CNN_LSTM_PATH, self.device)
        self.seq_len = Config.SEQUENCE_LENGTH
        self.threshold = Config.FIRE_THRESHOLD if threshold is None else threshold
        self.tracker = IoUTracker()
        self.buffers = {}
        self.timing = defaultdict(list)
        self.events = []

    def warmup(self, n=3):
        x = torch.zeros(1, self.seq_len, 3, *Config.IMG_SIZE, device=self.device)
        with torch.inference_mode():
            for _ in range(n):
                self.model(x)
        _sync(self.device)

    def process_frame(self, frame, frame_idx=0):
        t0 = time.perf_counter()
        boxes = self.detector.detect(frame)
        t1 = time.perf_counter()

        tracks = self.tracker.update(boxes)
        alive = {tid for tid, _ in tracks}
        for tid in list(self.buffers):
            if tid not in alive:
                del self.buffers[tid]
        for tid, box in tracks:
            crop = crop_roi(frame, box)
            if crop is not None:
                self.buffers.setdefault(tid, deque(maxlen=self.seq_len)).append(crop)
        t2 = time.perf_counter()

        ready = [(tid, box) for tid, box in tracks
                 if tid in self.buffers and len(self.buffers[tid]) == self.seq_len]
        probs = {}
        if ready:
            x = torch.stack([clip_to_tensor(np.stack(self.buffers[tid])) for tid, _ in ready]).to(self.device)
            with torch.inference_mode():
                p = torch.sigmoid(self.model(x).float()).cpu().numpy()
            _sync(self.device)
            probs = {tid: float(v) for (tid, _), v in zip(ready, p)}
        t3 = time.perf_counter()

        self.timing['stage1_detect_ms'].append((t1 - t0) * 1000)
        self.timing['track_crop_ms'].append((t2 - t1) * 1000)
        if ready:
            self.timing['stage2_classify_ms'].append((t3 - t2) * 1000)
        self.timing['total_ms'].append((t3 - t0) * 1000)

        results = []
        for tid, box in tracks:
            prob = probs.get(tid)
            fire = prob is not None and prob >= self.threshold
            results.append(dict(track_id=tid, box=box, prob=prob, is_fire=fire))
            if prob is not None:
                self.events.append(dict(frame=frame_idx, track_id=tid, prob=prob, is_fire=fire))
        return results

    @staticmethod
    def _draw(frame, results):
        for r in results:
            x1, y1, x2, y2 = r['box']
            if r['prob'] is None:
                color, text = (160, 160, 160), 'collecting'
            elif r['is_fire']:
                color, text = (0, 0, 255), f'FIRE {r["prob"]:.2f}'
            else:
                color, text = (255, 128, 0), f'non-fire {r["prob"]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'#{r["track_id"]} {text}', (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run(self, source, show=False, max_frames=None, out_json=None, save_frames_dir=None,
            save_every=30):
        """source: RTSP URL, 영상 파일 경로. 실측 요약(summary)을 반환한다."""
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f'영상/스트림을 열 수 없습니다: {source}')
        if str(source).startswith('rtsp'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1 else 30.0
        skip = max(1, int(round(fps / Config.FPS_LIMIT)))
        if save_frames_dir:
            Path(save_frames_dir).mkdir(parents=True, exist_ok=True)

        self.warmup()
        n_read = n_proc = 0
        wall0 = time.perf_counter()
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or (max_frames and n_read >= max_frames):
                break
            if n_read % skip == 0:
                res = self.process_frame(frame, n_read)
                n_proc += 1
                if show or save_frames_dir:
                    self._draw(frame, res)
                if save_frames_dir and n_proc % save_every == 0:
                    cv2.imwrite(str(Path(save_frames_dir) / f'frame_{n_read:06d}.jpg'), frame)
            if show:
                cv2.imshow('fire-detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            n_read += 1
        cap.release()
        if show:
            cv2.destroyAllWindows()

        summary = self.summary(n_read, n_proc, time.perf_counter() - wall0, fps, skip)
        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(
                dict(summary=summary, events=self.events), indent=2, ensure_ascii=False), encoding='utf-8')
        return summary

    def summary(self, n_read, n_proc, wall_sec, src_fps, skip):
        stats = {}
        for k, v in self.timing.items():
            a = np.asarray(v)
            stats[k] = dict(n=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                            p95=float(np.percentile(a, 95)))
        total = stats.get('total_ms')
        fire_tracks = sorted({e['track_id'] for e in self.events if e['is_fire']})
        return dict(
            device=self.device, source_fps=src_fps, frame_skip=skip,
            frames_read=n_read, frames_analyzed=n_proc,
            analyzed_fps_compute_only=(1000.0 / total['mean']) if total else None,
            wall_clock_sec=wall_sec, timing_ms=stats,
            classified_windows=len(self.events), fire_windows=sum(e['is_fire'] for e in self.events),
            fire_track_ids=fire_tracks, threshold=self.threshold)
