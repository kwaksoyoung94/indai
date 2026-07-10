import os
import tensorflow as tf
from config import Config
from dataset.data_loader import FireDatasetLoader
from models.cnn_lstm import build_cnn_lstm_model

def main():
    print("=======================================================")
    print("      고정밀 화재 감지 시스템: CNN-LSTM 모델 학습")
    print("=======================================================")
    
    # 1. 하드웨어(GPU) 가속 확인
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU 가속 구동 가능: {gpus}")
        # 메모리 증가 옵션 설정 (OOM 방지)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("GPU를 찾을 수 없습니다. CPU로 학습을 진행합니다. (속도가 느릴 수 있음)")

    # 2. 데이터 로더 정의 (이전에 만든 data_loader.py 활용)
    print("\n[1/3] 데이터셋 로드 및 전처리 시작...")
    train_loader = FireDatasetLoader(data_dir='dataset/train', shuffle=True)
    val_loader = FireDatasetLoader(data_dir='dataset/val', shuffle=False)

    # 3. 모델 빌드 (이전에 만든 cnn_lstm.py 활용)
    print("\n[2/3] CNN-LSTM 시공간 복합 모델 구조 정의 중...")
    model = build_cnn_lstm_model()
    model.summary() # 모델 아키텍처 구조 시각화 출력

    # 4. 모델 저장 및 체크포인트 설정 폴더 생성
    weights_dir = os.path.dirname(Config.CNN_LSTM_PATH)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # 최고 성능 모델 자동 저장을 위한 콜백 정의
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=Config.CNN_LSTM_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # 5. 실제 모델 학습 수행 (성능 개선 및 정량 지표 추출 과정)
    print("\n[3/3] 모델 학습 및 성능 지표 검증 가동...")
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=Config.EPOCHS,
        callbacks=[checkpoint]
    )

    print("\n=======================================================")
    print("학습 완료! 최고 성능의 가중치 파일이 저장되었습니다.")
    print(f"저장 경로: {Config.CNN_LSTM_PATH}")
    print("=======================================================")

if __name__ == "__main__":
    main()