import cv2
import numpy as np

# 이미지를 읽어오기
image = cv2.imread('candies.png')

# 이미지를 HSV 색상 공간으로 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색의 HSV 범위 설정 (빨간색은 Hue 값이 0 근처에 있음)
lower_red1 = np.array([0, 120, 70])  # 첫번째 범위
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])  # 두번째 범위
upper_red2 = np.array([180, 255, 255])

# 범위 안에 들으면 흰색, 아니면 검정색으로 이진화
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# 두 개의 마스크를 합치기
red_mask = cv2.bitwise_or(mask1, mask2)

# 원본 이미지와 마스크를 활용해 빨간색 부분 추출
red_result = cv2.bitwise_and(image, image, mask=red_mask)

# 결과를 화면에 출력
cv2.imshow('Original Image', image)
cv2.imshow('Red Mask', red_mask)
cv2.imshow('Red Color Extracted', red_result)

# 키 입력 대기 후, 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()