import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 데이터 준비
# MNIST 데이터 로드
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 정규화 (0~255 값을 0~1 범위로 스케일링)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 라벨 데이터 One-hot Encoding 처리 (0~9 → 벡터 형태)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 2. 모델 정의
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1D 벡터로 변환
    Dense(128, activation='relu'),  # 은닉층 1: 노드 128개, 활성화 함수 ReLU
    Dense(64, activation='relu'),  # 은닉층 2: 노드 64개, 활성화 함수 ReLU
    Dense(10, activation='softmax')  # 출력층: 10개 클래스, 소프트맥스 활성화
])

# 3. 모델 컴파일
model.compile(optimizer='adam',  # 옵티마이저: Adam
              loss='categorical_crossentropy',  # 손실 함수: 카테고리 크로스엔트로피
              metrics=['accuracy'])  # 평가지표: 정확도

# 4. 모델 학습
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# 5. 학습 결과 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 6. 예측
predictions = model.predict(X_test[:5])  # 테스트 데이터의 처음 5개 샘플로 예측
predicted_labels = tf.argmax(predictions, axis=1)
true_labels = tf.argmax(y_test[:5], axis=1)

print("Predicted Labels:", predicted_labels.numpy())
print("True Labels:", true_labels.numpy())