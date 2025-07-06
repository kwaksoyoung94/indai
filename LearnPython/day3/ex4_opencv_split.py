import cv2
import numpy as np

# Lenna.png 이미지 읽기
img = cv2.imread('Lenna.png')  # OpenCV는 이미지를 기본적으로 BGR 형식으로 읽음

# 이미지의 각각의 성분(B, G, R)을 분리
b, g, r = cv2.split(img)

# 각 채널 성분 출력
print("Blue Channel:")
print(b)

print("\nGreen Channel:")
print(g)

print("\nRed Channel:")
print(r)

# 이미지 채널 확인을 위한 시각화 (필요하면 활성화하세요)
cv2.imshow('Blue Channel', b)
cv2.imshow('Green Channel', g)
cv2.imshow('Red Channel', r)

cv2.waitKey(0)  # 키 입력 기다림
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기