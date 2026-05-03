# Fire and Smoke Detection System

실시간 화재/연기 탐지 시스템 구현

논문: "False Positive Decremented Research for Fire and Smoke Detection in Surveillance Camera using Spatial and Temporal Features Based on Deep Learning"

## 프로젝트 구조

```
fire_smoke_detection/
├── config.py                  # 설정 파일
├── feature_extraction.py      # 특징 추출 (SSIM, MSE, Wavelet 등)
├── faster_rcnn_detector.py    # Faster R-CNN 모델
├── fire_smoke_detector.py     # 통합 탐지 시스템
├── inference.py              # 추론 스크립트
├── train.py                  # 훈련 스크립트
├── requirements.txt          # 필요한 라이브러리
├── data/
│   ├── train/               # 훈련 데이터
│   │   ├── fire/
│   │   ├── smoke/
│   │   └── non_fire/
│   └── test/                # 테스트 데이터
├── models/                   # 모델 코드
└── weights/                  # 학습된 가중치

```

## 주요 특징

### 1. 전역 특징 검증 (Global Feature Validation)

- **SSIM (Structural Similarity)**: 프레임 유사도 계산
- **MSE (Mean Square Error)**: 평균 제곱 오차
- **3-Frame Difference**: 3프레임 차이 기반 움직임 감지

### 2. Faster R-CNN 기반 객체 탐지

- ResNet 백본 (Backbone)
- RPN (Region Proposal Network)
- ROI Pooling

### 3. 지역 특징 검증 (Local Feature Validation)

- **색상 히스토그램 (Color Histogram)**
  - 연기: 회색~흰색 (RGB: 80-250)
  - 화재: HSV 색상 범위
- **Wavelet Transform**: 고주파 성분 분석
  - 연기는 낮은 Wavelet 에너지
- **Coefficient of Variation (CV)**: 표준편차/평균
  - 연기/화재: 낮은 CV
  - 거짓 양성: 높은 CV

### 4. 거짓 양성 감소

- 전역 + 지역 특징 결합으로 99.9% 거짓 양성 감소
- 움직임 있을 때만 Deep Learning 실행 (계산량 감소)

## 설치 및 사용

### 1. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 훈련 (선택사항)

데이터 구조:

```
data/train/
├── fire/
│   ├── fire1.jpg
│   ├── fire2.jpg
│   └── ...
├── smoke/
│   ├── smoke1.jpg
│   └── ...
└── non_fire/
    ├── office1.jpg
    └── ...
```

훈련 실행:

```bash
python train.py --data-dir data/train --epochs 100 --batch-size 32
```

### 3. 추론

#### 이미지 처리

```bash
python inference.py --mode image --input test_image.jpg --output result.jpg
```

#### 비디오 처리

```bash
python inference.py --mode video --input test_video.mp4 --output result.mp4
```

#### 실시간 카메라

```bash
python inference.py --mode camera --camera-id 0
```

## API 사용

```python
from config import Config
from fire_smoke_detector import FireSmokeDetector
import cv2

# 초기화
config = Config()
detector = FireSmokeDetector(config, use_pretrained=True)

# 이미지 처리
frame = cv2.imread('image.jpg')
annotated_frame, detections = detector.process_frame(frame)

# 결과 확인
for det in detections:
    print(f"{det['class']}: {det['confidence']:.2f}")

# 비디오 처리
stats = detector.process_video('video.mp4', 'output.mp4')
print(stats)
```

## 성능 지표

| 지표             | 설명                     | 논문 결과 |
| ---------------- | ------------------------ | --------- |
| 거짓 양성 감소율 | False Positive Reduction | 99.9%     |
| 화재 탐지율      | Fire Detection Rate      | >95%      |
| 연기 탐지율      | Smoke Detection Rate     | >90%      |

## 구성 요소 설명

### config.py

- 모든 임계값 설정
- 모델 하이퍼파라미터
- 색상 범위 및 특징 추출 파라미터

### feature_extraction.py

- SSIM 계산
- MSE 계산
- 3-Frame Difference
- 색상 히스토그램
- Coefficient of Variation
- Wavelet Transform

### faster_rcnn_detector.py

- Faster R-CNN 모델 구현
- 객체 탐지
- 모델 훈련 및 저장

### fire_smoke_detector.py

- 전체 탐지 파이프라인
- 전역 + 지역 특징 검증
- 비디오/카메라 처리
- 통계 계산

## 논문의 주요 알고리즘

### 전역 특징 (Global Check)

```
FSG = 1 if S_k < th1 AND M_k < th2 AND A_k < th3
    0 otherwise

where:
- S_k = SSIM(f_i, f_j)
- M_k = MSE(f_i, f_j)
- A_k = 3-frame difference
```

### 화재 지역 특징 (Fire Local Check)

```
FL = 1 if M_k < f_th1 AND A_k < f_th2 AND H_sum*F < f_th3 AND WE_k < f_th4
   0 otherwise
```

### 연기 지역 특징 (Smoke Local Check)

```
SL = 1 if M_k > s_th1 AND A_k > s_th2 AND H_sum*F > s_th3 AND WE_k < s_th4
   0 otherwise
```

### 최종 판정 (Final Decision)

```
FD = 1 if FL > 0 AND SL > 0 AND FSD > 0
   0 otherwise
```

## 시스템 요구사항

- Python 3.7+
- TensorFlow 2.10+
- OpenCV 4.5+
- NumPy 1.20+
- GPU (권장)

## 참고 문헌

Lee, Y., & Shim, J. (2019). False Positive Decremented Research for Fire and Smoke Detection in Surveillance Camera using Spatial and Temporal Features Based on Deep Learning. Electronics, 8(10), 1167.

https://www.mdpi.com/2079-9292/8/10/1167

## 라이센스

이 프로젝트는 교육 및 연구 목적으로 만들어졌습니다.
