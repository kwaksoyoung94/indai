import cv2
import matplotlib.pyplot as plt
import numpy as np

# candies.png 이미지 읽기
image = cv2.imread('candies.png')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 로드할 수 없습니다.")
else:
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # HSV에서 빨간색 범위 설정
    # HSV의 H(Hue) 범위는 0-180이고, 빨간색은 0-10과 170-180 범위
    lower_red1 = np.array([0, 100, 100])      # 낮은 빨간색 범위1 (0-10)
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 100, 100])    # 낮은 빨간색 범위2 (170-180)
    upper_red2 = np.array([180, 255, 255])
    
    # 마스크 생성
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    # 두 마스크를 합치기
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 마스크를 사용하여 붉은색 캔디만 추출
    red_candy = cv2.bitwise_and(image, image, mask=red_mask)
    
    print("✓ 빨간색 캔디 추출 완료!")
    print(f"원본 이미지 크기: {image.shape}")
    print(f"마스크의 흰색 픽셀 개수: {cv2.countNonZero(red_mask)}")
    print(f"전체 픽셀 대비 빨간색 비율: {(cv2.countNonZero(red_mask) / red_mask.size * 100):.2f}%")
    
    # 결과 표시
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 원본 이미지 (RGB로 변환)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # HSV 이미지 시각화 (H 채널만 표시)
    axes[0, 1].imshow(hsv_image[:, :, 0], cmap='hsv')
    axes[0, 1].set_title('HSV - Hue Channel', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 마스크
    axes[1, 0].imshow(red_mask, cmap='gray')
    axes[1, 0].set_title('Red Color Mask', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 추출된 빨간색 캔디 (RGB로 변환)
    red_candy_rgb = cv2.cvtColor(red_candy, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(red_candy_rgb)
    axes[1, 1].set_title('Extracted Red Candy', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('red_candy_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 추출된 이미지 저장
    cv2.imwrite('red_candy_extracted.png', red_candy)
    
    print("\n저장된 파일:")
    print("- red_candy_result.png (결과 비교 이미지)")
    print("- red_candy_extracted.png (추출된 빨간색 캔디)")
