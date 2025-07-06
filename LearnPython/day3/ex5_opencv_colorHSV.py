import cv2
import numpy as np

# 이미지 읽기 (Lenna.png가 현재 디렉토리에 있다고 가정)
image = cv2.imread('Lenna.png')

# BGR에서 HSV로 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# HSV 분리
h, s, v = cv2.split(hsv_image)

# 결과 출력
print("Hue (H) 성분:")
print(h)

print("\nSaturation (S) 성분:")
print(s)

print("\nValue (V) 성분:")
print(v)

# 이미지 시각화 (선택적으로 표시 가능)
cv2.imshow('Hue', h)
cv2.imshow('Saturation', s)
cv2.imshow('Value', v)

cv2.waitKey(0)
cv2.destroyAllWindows()