# PowerForecast

전력 수요 예측 및 관련 머신러닝 모델 코드가 포함된 폴더입니다.

## 주요 파일 및 폴더
- [img/](./img)  
  : 예측 결과, 데이터 시각화 등 이미지 파일 저장 폴더
- [read_data.py](./read_data.py)  
  : 데이터 읽기 및 전처리 스크립트
- [requirements.txt](./requirements.txt)  
  : 의존 패키지 목록
- [temp_1d-cnn.py](./temp_1d-cnn.py)  
  : 1D-CNN을 활용한 예측 모델 코드
- [temp_ar_ma.py](./temp_ar_ma.py)  
  : AR, MA 기반 예측 코드
- [temp_arima.py](./temp_arima.py)  
  : ARIMA 모델 예측 코드
- [temp_arima_pmd.py](./temp_arima_pmd.py)  
  : PMD 기반 ARIMA 예측 코드
- [temp_lstm.py](./temp_lstm.py)  
  : LSTM 모델 예측 코드
- [전력 수요 예측 모델 개발.pdf](./전력 수요 예측 모델 개발.pdf)  
  : 프로젝트 설명 및 결과 문서

## 목적
- 전력 수요 데이터 기반 예측 모델 실습 및 비교

## 사용 방법
1. `requirements.txt`로 필요한 패키지 설치
2. 각 모델별 `.py` 파일 실행하여 예측 결과 확인
