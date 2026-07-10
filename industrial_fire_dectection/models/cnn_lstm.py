import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from config import Config

def build_cnn_lstm_model():
    """
    용접 불꽃, 경광등 등의 주기적/일시적 외란 광원을 
    시계열 패턴 분석으로 필터링하는 고도화 모델
    """
    model = Sequential()
    
    # TimeDistributed CNN: 공간적 특징 추출
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), 
                              input_shape=(Config.SEQUENCE_LENGTH, Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    
    # LSTM: 시간적 움직임 및 확산성 학습
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # 화재 여부 이진 분류
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model