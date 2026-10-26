# industrial_fire_dectection

제조·물류 현장의 광학적 외란(용접 불꽃, 반사광, 조명 깜빡임, 증기 등)으로 인한 화재 감지 오탐을 줄이기 위한 **2-Stage 화재 감지 시스템**입니다. 충북대학교 산업인공지능학과 석사학위논문 연구 코드입니다.

## 시스템 개요

| 단계 | 역할 | 코드 |
|---|---|---|
| Stage 1 | 프레임에서 화재 후보 영역(ROI) 탐지 (YOLO 또는 밝기 기반) | `models/yolo_detector.py`, `models/candidate_detector.py` |
| 추적 | 후보 ROI를 IoU 기반으로 추적하며 시퀀스 구성 | `models/tracker.py` |
| 특징 | 시공간 특징(SSIM 등) 추출 | `models/features.py` |
| Stage 2 | ROI 시퀀스(30프레임)를 CNN-LSTM(PyTorch)으로 화재/비화재 분류 | `models/cnn_lstm.py` |
| 통합 | 위 단계를 묶어 영상/RTSP 입력을 처리 | `pipeline/pipeline_manager.py`, `main.py` |

주요 설정(이미지 크기, 시퀀스 길이, 임계값, 학습 하이퍼파라미터)은 `config.py`에 모여 있습니다.

## 폴더 구성

- `config.py`, `main.py` : 설정 및 파이프라인 실행 진입점
- `models/`, `pipeline/` : 검출기·추적기·특징·분류기·파이프라인
- `dataset/` : 영상 수집/ROI 클립 생성/로더 스크립트 (영상·클립 자체는 git 제외)
- `train.py`, `train_yolo.py`, `evaluate.py`, `evaluation/` : 학습과 평가 지표
- `baselines.py`, `run_compare.py` : 비교 실험(프레임 1장, ResNet18, R3D-18 등)
- `run_ablation.py`, `temporal_order_test.py`, `seed_report.py`, `cv_report.py`, `system_eval.py`, `yolo_candidate_rate.py` : 보강 실험 및 리포트
- `evaluation_results/` : 실험 결과(json/csv/png). 논문의 수치와 그림은 여기 실측값에서만 생성
- `demo_render.py`, `make_diagram.py`, `generate_report_images.py` : 시연 영상·구조도·보고서 이미지
- `tests/smoke_test.py` : 스모크 테스트
- `legacy/` : 초기 TensorFlow 기반 버전(`_v0`), 현재 코드와 별개

## 실행

```bash
pip install -r requirements.txt
# RTX 50 시리즈는 CUDA 12.8 빌드 필요: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python main.py <영상 또는 RTSP> --detector yolo --checkpoint <가중치> --out result.json
python tests/smoke_test.py
```

## 평가 원칙

- 데이터 분할은 **영상 단위 5-fold**입니다. 같은 영상의 클립이 학습/평가에 섞이는 클립 단위 분할은 쓰지 않습니다.
- 주 결과는 사전에 정한 설정(seed 42)으로 보고하고, 다른 설정과 시드 반복은 민감도·변동성 분석으로 따로 둡니다.

## git에 포함되지 않는 것

데이터셋 영상·ROI 클립, 모델 가중치(`*.pt`, `*.h5`), `docs/`(논문 양식·면담자료), 실험 로그와 PDF. 각 규칙은 `.gitignore`를 참고하세요.
