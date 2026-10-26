"""전체 경로(manifest -> ROI 클립 -> 학습 -> 평가 -> baseline -> 파이프라인 -> 그림) 동작 검증용.

임시 폴더에 합성 영상을 만들어 실행하므로 결과 수치는 어떤 연구 결과로도 사용할 수 없다.
    python tests/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config  # noqa: E402


def make_video(path, kind, seed, n=240, fps=30, size=(320, 240)):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    cx, cy = int(rng.integers(100, 220)), int(rng.integers(80, 160))
    for t in range(n):
        f = rng.integers(20, 50, (size[1], size[0], 3), dtype=np.uint8)
        if kind == 'interference':      # 주기적으로 점멸하는 주황 광원
            amp = 0.5 + 0.5 * np.sin(2 * np.pi * t / 12)
            r = 28
        else:                            # 불규칙하게 요동하는 불꽃
            amp = float(rng.uniform(0.3, 1.0))
            r = int(rng.integers(18, 40))
        color = (int(30 * amp), int(140 * amp + 60), int(255 * amp))
        cv2.circle(f, (cx + int(rng.integers(-4, 5)), cy), r, color, -1)
        w.write(f)
    w.release()


def run(mod_main, argv):
    sys.argv = argv
    mod_main()


def main():
    tmp = Path(tempfile.mkdtemp(prefix='fire_smoke_'))
    Config.set_root(tmp / 'data', tmp / 'weights', tmp / 'results')
    Config.NUM_WORKERS = 0
    Config.SEQUENCE_LENGTH, Config.SEQUENCE_STRIDE = 10, 5
    print('임시 폴더:', tmp)

    for i in range(6):
        make_video(Config.VIDEO_DIR / 'fire' / 'synthetic' / f'f{i}.mp4', 'fire', i)
        make_video(Config.VIDEO_DIR / 'interference' / 'synthetic' / f'i{i}.mp4', 'interference', 100 + i)

    from dataset import manifest, build_roi_clips
    manifest.scan()
    manifest.split()
    run(build_roi_clips.main, ['x', '--detector', 'bright', '--max-clips-per-video', '30'])

    import train, evaluate, baselines
    run(train.main, ['x', '--name', 'smoke', '--epochs', '3'])
    run(evaluate.main, ['x', '--name', 'smoke', '--split', 'test', '--seq-len', '10'])
    run(baselines.main, ['x', '--name', 'rule_smoke', '--seq-len', '10'])

    # k-fold 경로: fold 배정 -> fold별 학습/평가 -> 집계
    import cv_report
    manifest.assign_folds(3)
    for f in range(3):
        run(train.main, ['x', '--name', 'smokecv', '--epochs', '2', '--fold', str(f)])
        run(evaluate.main, ['x', '--name', 'smokecv', '--fold', str(f), '--seq-len', '10'])
    run(cv_report.main, ['x', '--name', 'smokecv', '--k', '3'])

    from models.candidate_detector import BrightRegionDetector
    from pipeline.pipeline_manager import FireDetectionPipeline
    pipe = FireDetectionPipeline(BrightRegionDetector(), Config.WEIGHTS_DIR / 'smoke' / 'best.pt')
    pipe.seq_len = 10
    vid = next((Config.VIDEO_DIR / 'fire').rglob('*.mp4'))
    s = pipe.run(vid, out_json=tmp / 'results' / 'pipeline_run.json', save_frames_dir=tmp / 'frames')
    print('pipeline:', {k: s[k] for k in ('frames_read', 'frames_analyzed', 'classified_windows',
                                          'analyzed_fps_compute_only')})

    import generate_report_images as gri
    gri.setup_font()
    gri.ssim_chart('test', 5, tmp / 'ssim.png')
    gri.latency_chart(tmp / 'results' / 'pipeline_run.json', tmp / 'lat.png')
    print('OK - 산출물:', sorted(p.name for p in (tmp / 'results').rglob('*') if p.is_file())[:12])


if __name__ == '__main__':
    main()
