import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 데이터 정규화 (0~1로 스케일링)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 라벨을 One-hot 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# CNN 모델 생성
model = models.Sequential([
    # Conv2D + MaxPooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Fully Connected
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_data=(x_test, y_test))

# 테스트 데이터로 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc * 100:.2f}%')

# CIFAR-10 라벨 목록
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 테스트 샘플에서 10개 이미지 선택
num_samples = 10
sample_images = x_test[:num_samples]
sample_labels = np.argmax(y_test[:num_samples], axis=1)

# 예측 수행
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(sample_images[i])
    true_label = class_names[sample_labels[i]]
    pred_label = class_names[predicted_labels[i]]
    color = 'blue' if true_label == pred_label else 'red'
    plt.xlabel(f"True: {true_label}\nPred: {pred_label}", color=color)
plt.tight_layout()
plt.show()