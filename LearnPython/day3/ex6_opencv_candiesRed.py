import cv2
import numpy as np

# 이미지를 읽기
image = cv2.imread('Candies.png')

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 이미지를 BGR 채널로 분리
b, g, r = cv2.split(image)

# Red 성분이 50 이상인 범위를 설정
red_min = 150
red_mask = cv2.inRange(r, red_min, 255)

# 결과로 마스크와 원본 이미지를 결합하여 빨간색 성분만 추출
red_result = cv2.bitwise_and(image, image, mask=red_mask)

# 화면에 결과 출력
cv2.imshow("Original Image", image)  # 원본 이미지 출력
cv2.imshow("Red Pixels Only", red_result)  # 빨간색 성분만 출력

# 키 입력 대기
cv2.waitKey(0)
cv2.destroyAllWindows()