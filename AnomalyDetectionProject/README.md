# 기계 예측 유지보수 분류

센서 데이터를 이용한 기계 상태 분류 및 이상 탐지 프로젝트입니다. 지도학습과 비지도학습 모델을 비교하여 기계의 정상/비정상 상태를 예측합니다.

## 개요

### 프로젝트 목표
- 기계 센서 데이터를 분석하여 기계의 정상/이상 상태 판별
- 여러 머신러닝 모델의 성능 비교
- 데이터 전처리 및 클래스 불균형 처리 방법 학습
- 예측 유지보수(Predictive Maintenance)의 실제 응용

### 데이터 설명
- **특성**: 온도, 진동, 압력, 습도, 모터 속도 등 8개 센서 값
- **샘플 수**: 약 10,000개
- **클래스**: 정상(0) / 이상(1)

## 프로젝트 구조

### 분석 단계

1. **데이터 로드 및 탐색**
   - 데이터셋 로드 및 기본 통계
   - 클래스 분포 확인
   - 특성별 분포 분석
   - 상관관계 분석

2. **데이터 전처리**
   - 결측치 처리
   - 특성 표준화
   - 80/20 비율 데이터 분리
   - SMOTE를 이용한 클래스 불균형 처리

3. **모델 개발**
   - 비지도학습: K-Means, GMM, Isolation Forest
   - 지도학습: SVM, Random Forest, Gradient Boosting, XGBoost

4. **모델 평가**
   - 정확도, 정밀도, 재현율, F1-Score
   - 혼동 행렬 및 ROC-AUC
   - 모델 간 성능 비교

5. **분석 및 해석**
   - 특성 중요도 분석
   - 모델별 성능 요약

## 필수 패키지

```bash
pip install -r requirements.txt
```

주요 라이브러리:
- `pandas`, `numpy`: 데이터 처리
- `scikit-learn`: 머신러닝 모델
- `xgboost`: XGBoost 모델
- `matplotlib`, `seaborn`: 시각화
- `imbalanced-learn`: SMOTE

## 실행 방법

### 1. Kaggle 데이터 다운로드 (선택사항)

```bash
kaggle datasets download -d stephanmatzka/predictive-maintenance-classification
unzip predictive-maintenance-classification.zip
```

샘플 데이터가 없으면 자동으로 생성됩니다.

### 2. Jupyter Notebook 실행

```bash
jupyter notebook Machine_Predictive_Maintenance.ipynb
```

각 셀을 순서대로 실행하면 됩니다.

## 모델 성능 요약

| 모델 | 정확도 | 특징 |
|------|------|------|
| **지도학습** | ~95% | 레이블 데이터 활용, 높은 성능 |
| **비지도학습** | ~70% | 레이블 없이 패턴 자동 발견 |

### 최고 성능 모델
- XGBoost, Gradient Boosting 등 트리 기반 앙상블 모델
- 높은 정확도와 재현율 제공

## 주요 발견사항

1. **지도학습의 우수성**: 레이블이 있을 때 지도학습이 훨씬 좋은 성능 제공
2. **클래스 불균형 처리의 중요성**: SMOTE 적용으로 재현율 향상
3. **앙상블 모델의 효과성**: Random Forest, Gradient Boosting의 안정적인 성능
4. **모델 해석 가능성**: 특성 중요도를 통한 의사결정 과정 이해

## 참고자료

- [Kaggle 데이터셋](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-classification)
- 예측 유지보수 (Predictive Maintenance)
- 산업용 IoT 및 센서 기술

## 프로젝트 특징

- 실무적인 머신러닝 파이프라인
- 다양한 모델 구현 및 비교
- 명확한 분석 구조
- 풍부한 시각화

---

**Last Updated**: 2025년 12월
