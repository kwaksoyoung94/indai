import argparse
import json

from config import Config
from pipeline.pipeline_manager import FireDetectionPipeline


def make_detector(name):
    if name == 'yolo':
        from models.yolo_detector import YoloFireDetector
        return YoloFireDetector()
    from models.candidate_detector import BrightRegionDetector
    return BrightRegionDetector()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='2-Stage 화재 감지 파이프라인 실행(영상 파일 또는 RTSP)')
    ap.add_argument('source', nargs='?', default=Config.RTSP_URL)
    ap.add_argument('--detector', choices=['yolo', 'bright'], default='yolo')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--out', default=None, help='실측 결과(JSON) 저장 경로')
    ap.add_argument('--save-frames', default=None, help='주석 프레임 저장 폴더(논문 그림용)')
    a = ap.parse_args()

    pipe = FireDetectionPipeline(make_detector(a.detector), a.checkpoint, a.threshold)
    summary = pipe.run(a.source, show=a.show, max_frames=a.max_frames, out_json=a.out,
                       save_frames_dir=a.save_frames)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
