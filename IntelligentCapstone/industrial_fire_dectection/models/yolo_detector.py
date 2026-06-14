import cv2
from ultralytics import YOLO
from config import Config

class YoloFireDetector:
    def __init__(self):
        self.model = YOLO(Config.YOLO_MODEL_PATH)
        
    def extract_rois(self, frame):
        """
        전체 프레임에서 화재 후보 영역(Bounding Box)을 찾아 크롭합니다.
        교수님 피드백: 전체 연산량 감소를 위한 ROI 추출 핵심 로직
        """
        results = self.model.predict(frame, conf=Config.CONF_THRESHOLD, verbose=False)
        rois = []
        bboxes = []
        
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                # 안전 마진을 포함하여 크롭
                roi = frame[max(0, y1-10):min(frame.shape[0], y2+10), 
                            max(0, x1-10):min(frame.shape[1], x2+10)]
                
                if roi.size > 0:
                    roi_resized = cv2.resize(roi, Config.IMG_SIZE)
                    roi_normalized = roi_resized / 255.0
                    rois.append(roi_normalized)
                    bboxes.append((x1, y1, x2, y2))
                    
        return rois, bboxes