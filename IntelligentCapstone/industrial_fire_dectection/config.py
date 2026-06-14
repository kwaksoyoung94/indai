import torch

class Config:
    # 하이퍼파라미터 및 이미지 설정
    IMG_SIZE = (224, 224)
    SEQUENCE_LENGTH = 10
    FPS_LIMIT = 10              # 초당 10프레임 이하 샘플링으로 연산량 최적화
    CONF_THRESHOLD = 0.4        # YOLO 후보 영역 탐지 임계값
    FIRE_THRESHOLD = 0.75       # CNN-LSTM 최종 화재 판정 임계값
    
    # 학습 설정
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 저장 경로
    YOLO_MODEL_PATH = 'models/weights/yolov8n_fire.pt'
    CNN_LSTM_PATH = 'models/weights/spatiotemporal_classifier.h5'
    
    # 실증 테스트용 공장 CCTV RTSP / WebRTC 미디어 서버 URL
    # 예시: 'rtsp://admin:password@192.168.1.100:554/stream1' 또는 WebRTC 로컬 매핑 주소
    RTSP_URL = "rtsp://localhost:8554/factory_cctv"