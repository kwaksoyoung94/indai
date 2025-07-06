import cv2
import numpy as np

# 이미지를 읽습니다
image = cv2.imread('Lenna.png', cv2.IMREAD_COLOR)

if image is None:
    print("이미지를 로드할 수 없습니다. 경로를 확인하세요!")
    exit()

# 1. 샤프닝 필터 적용
# 샤프닝 커널 정의
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

# 2. 평균값 필터(블러링) 적용
blurred_image = cv2.blur(image, (5, 5))  # 5x5 크기의 커널 사용

# 3. 라플라시안 필터 적용
# 라플라시안 필터는 엣지 감지에 자주 사용됨
laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
laplacian_image = cv2.convertScaleAbs(laplacian_image)  # 절대값 및 형 변환

# 결과 출력
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.imshow('Blurred Image (Average Filter)', blurred_image)
cv2.imshow('Laplacian Filtered Image', laplacian_image)

# 키 입력 대기 후 종료
cv2.waitKey(0)
cv2.destroyAllWindows()