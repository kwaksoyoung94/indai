import cv2
import numpy as np


class BrightRegionDetector:
    """YOLO 화재 가중치 없이 쓸 수 있는 규칙 기반 후보 영역 탐지기.

    고휘도 영역 또는 주황-적색 영역을 후보로 뽑는다. (1) YOLO 학습 이전 단계에서 ROI
    시퀀스 데이터셋을 만들 때, (2) 'YOLO 없이 규칙 기반 Stage 1' 비교 실험에 사용한다.
    경광등, 용접광, 불꽃이 모두 후보로 잡히도록 설계되어 있어 판별은 Stage 2의 몫이다.
    """

    def __init__(self, v_min=200, orange_v_min=150, s_min=80, min_area_ratio=0.0005, max_boxes=3):
        self.v_min = v_min
        self.orange_v_min = orange_v_min
        self.s_min = s_min
        self.min_area_ratio = min_area_ratio
        self.max_boxes = max_boxes
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        bright = v >= self.v_min
        orange = ((h <= 35) | (h >= 170)) & (s >= self.s_min) & (v >= self.orange_v_min)
        mask = (bright | orange).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_ratio * frame_bgr.shape[0] * frame_bgr.shape[1]
        cands = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                x, y, w, hh = cv2.boundingRect(c)
                cands.append((area, (x, y, x + w, y + hh)))
        cands.sort(reverse=True)
        return [b for _, b in cands[:self.max_boxes]]
