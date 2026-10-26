import cv2
import numpy as np

from config import Config


def crop_roi(frame_bgr, box, margin=None, size=None):
    """프레임에서 박스 영역을 마진 포함해 잘라 RGB uint8 (H, W, 3)로 반환한다."""
    margin = Config.ROI_MARGIN if margin is None else margin
    size = Config.IMG_SIZE if size is None else size
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    roi = frame_bgr[max(0, y1 - margin):min(h, y2 + margin),
                    max(0, x1 - margin):min(w, x2 + margin)]
    if roi.size == 0:
        return None
    roi = cv2.resize(roi, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)


class YoloFireDetector:
    """Stage 1: YOLOv8로 화재 후보 박스를 탐지한다. 반환: [(x1, y1, x2, y2), ...]"""

    def __init__(self, model_path=None, conf=None, device=None):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path or Config.YOLO_MODEL_PATH))
        self.conf = Config.CONF_THRESHOLD if conf is None else conf
        self.device = device or Config.DEVICE

    def detect(self, frame_bgr):
        res = self.model.predict(frame_bgr, conf=self.conf, device=self.device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int) if len(res.boxes) else np.empty((0, 4), int)
        return [tuple(int(v) for v in b) for b in boxes]
