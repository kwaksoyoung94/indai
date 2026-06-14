# Optical Fire False Alarm Detection

제조·물류 환경의 광학적 간섭을 고려한 화재 감지 및 오탐 억제 연구 프로젝트입니다.

이 저장소는 기존 fire_smoke_detection 프로젝트를 기반으로 복제한 뒤, 용접 불꽃·반사광·조명 변화 같은 간섭 요소를 분리하는 후속 연구용 작업공간으로 사용합니다.

## 프로젝트 목적

- CCTV 기반 실시간 화재 감지
- 광학적 간섭 요소에 의한 오탐 감소
- RGB + Temporal Feature 기반 판별 구조 실험
- 기존 화재/연기 탐지 파이프라인과의 비교 실험
- 실제 추론은 RTSP 스트림을 입력으로 받아 WebRTC로 결과를 전달하는 구조
- 학습 데이터는 YouTube 영상과 공개 데이터셋을 중심으로 구성

## 현재 포함된 구성

```
optical_fire_false_alarm_detection/
├── config.py
├── feature_extraction.py
├── faster_rcnn_detector.py
├── fire_smoke_detector.py
├── inference.py
├── train.py
├── presentation_outline.md
├── requirements.txt
├── data/
├── models/
└── weights/
```

## 연구 방향

### 1. 1차 화재 후보 탐지

- YOLOv8, RT-DETR, Faster R-CNN 비교
- 화재 후보 영역을 빠르게 추출
- 기존 단일 프레임 탐지 대비 탐지 안정성 확보

### 2. 광학 간섭 오탐 억제

- 용접 불꽃, 반사광, 헤드라이트, 조명 깜빡임 등 분류
- CNN + LSTM, Temporal Attention, Optical Flow 적용 검토
- 프레임 간 변화 패턴을 이용한 오탐 필터링

### 3. 발표 및 논문 정리

- 연구 배경과 문제 정의 정리
- 실험 구성, 평가 지표, 한계점, 향후 연구 방향 정리
- 발표 자료 초안은 presentation_outline.md에서 관리

## 기존 코드 활용 방식

현재 프로젝트는 기존 탐지 코드를 유지한 상태로 시작합니다. 즉, 코드 실행은 기존 fire_smoke_detection과 유사한 방식으로 동작하지만, 논문 및 발표에서는 광학적 간섭 억제와 시계열 검증을 핵심 기여점으로 설명하는 구조입니다.

## 실행 참고

기존 프로젝트와 동일하게 의존성을 설치한 뒤 학습 또는 추론 스크립트를 실행할 수 있습니다.

```bash
pip install -r requirements.txt
python train.py
python inference.py --mode video --input sample.mp4 --output result.mp4
```

운영 단계에서는 RTSP URL로 들어오는 CCTV 영상을 받아 실시간 추론을 수행하고, 결과는 WebRTC 기반 스트리밍으로 확인하는 구성을 목표로 합니다.

## 참고 문헌

- Lee, Y., & Shim, J. (2019). False Positive Decremented Research for Fire and Smoke Detection in Surveillance Camera using Spatial and Temporal Features Based on Deep Learning. Electronics, 8(10), 1167.
- Muhammad, K., et al. Early Fire Detection using Convolutional Neural Networks during Surveillance for Smart Cities. Expert Systems with Applications (2018).
- Sharma, J., et al. Deep Learning Based Real-time Fire Detection for Video Surveillance. ICIP (2017).
