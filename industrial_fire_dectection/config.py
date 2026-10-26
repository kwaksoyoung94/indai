from pathlib import Path

import torch


class Config:
    ROOT = Path(__file__).resolve().parent
    SEED = 42

    # 입력/시퀀스 설정
    IMG_SIZE = (128, 128)       # ROI를 리사이즈하는 크기 (H, W)
    SEQUENCE_LENGTH = 30        # Stage 2 입력 프레임 수 (10 FPS 기준 3초)
    SEQUENCE_STRIDE = 15        # 학습용 클립 생성 시 슬라이딩 윈도우 간격
    FPS_LIMIT = 10              # 분석 프레임레이트 상한
    ROI_MARGIN = 10             # 크롭 안전 마진(px)

    # Stage 1 / Stage 2 임계값
    CONF_THRESHOLD = 0.4        # YOLO 후보 영역 탐지 임계값
    FIRE_THRESHOLD = 0.75       # CNN-LSTM 최종 화재 판정 임계값
    TRACK_IOU = 0.3             # ROI 추적 IoU 임계값
    TRACK_MAX_AGE = 10          # 탐지가 끊겨도 트랙을 유지하는 최대 프레임 수 (10FPS 기준 1초)
    TRAIN_CONF_THRESHOLD = 0.15  # Stage 2 학습/평가용 후보 생성 임계값 (운영 CONF_THRESHOLD 보다 낮게)

    # Stage 2 모델/학습 설정
    BACKBONE = 'scratch'        # scratch | mobilenet_v3_small | resnet18
    PRETRAINED = False
    LSTM_HIDDEN = 64
    DROPOUT = 0.5
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10               # val loss 기준 early stopping
    NUM_WORKERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 경로 (set_root로 재지정 가능)
    DATA_DIR = ROOT / 'dataset'
    VIDEO_DIR = DATA_DIR / 'videos'          # videos/<label>/<category>/*.mp4
    ROI_DIR = DATA_DIR / 'roi_clips'
    MANIFEST = DATA_DIR / 'manifest.csv'     # 영상 단위 메타데이터와 split
    ROI_INDEX = DATA_DIR / 'roi_index.csv'   # ROI 클립 단위 인덱스
    WEIGHTS_DIR = ROOT / 'models' / 'weights'
    RESULTS_DIR = ROOT / 'evaluation_results'
    YOLO_MODEL_PATH = WEIGHTS_DIR / 'yolov8n_fire.pt'
    CNN_LSTM_PATH = WEIGHTS_DIR / 'cnn_lstm' / 'best.pt'

    # 실증 테스트용 RTSP URL
    RTSP_URL = 'rtsp://localhost:8554/factory_cctv'

    LABELS = ('fire', 'interference', 'normal')   # 화재=1, 나머지=0
    EXCLUDE_CATEGORIES = ('vfx_overlay',)         # 그린스크린 합성 영상 등 현실성이 낮아 제외
    # 실제 화염이 있는 통제된 불꽃(벽난로, 가스 토치). 경보 대상이 아닌 비화재이지만 시각적으로는 화염이므로
    # 주 지표(외란 오경보율)에서 분리해 따로 보고한다.
    CONTROLLED_FLAME_CATEGORIES = ('fireplace', 'gas_torch')
    # 폴더(영상 ID)는 그대로 두고 카테고리만 재지정한다. 영상 ID를 바꾸면 탐지 캐시, ROI 클립, 예측 결과의
    # 참조가 모두 깨지기 때문. CeRRsDlQkxA 는 아크 용접이 아니라 가스 토치 불꽃임을 예시 프레임으로 확인했다.
    CATEGORY_OVERRIDES = {'interference__welding__CeRRsDlQkxA': 'gas_torch'}

    @classmethod
    def set_root(cls, data_dir=None, weights_dir=None, results_dir=None):
        """테스트나 다른 데이터셋 사용 시 파생 경로를 일괄 재계산한다."""
        if data_dir is not None:
            cls.DATA_DIR = Path(data_dir)
            cls.VIDEO_DIR = cls.DATA_DIR / 'videos'
            cls.ROI_DIR = cls.DATA_DIR / 'roi_clips'
            cls.MANIFEST = cls.DATA_DIR / 'manifest.csv'
            cls.ROI_INDEX = cls.DATA_DIR / 'roi_index.csv'
        if weights_dir is not None:
            cls.WEIGHTS_DIR = Path(weights_dir)
            cls.YOLO_MODEL_PATH = cls.WEIGHTS_DIR / 'yolov8n_fire.pt'
            cls.CNN_LSTM_PATH = cls.WEIGHTS_DIR / 'cnn_lstm' / 'best.pt'
        if results_dir is not None:
            cls.RESULTS_DIR = Path(results_dir)
