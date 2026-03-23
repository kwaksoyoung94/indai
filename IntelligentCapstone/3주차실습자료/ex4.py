import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lenna.png 파일을 BGR로 읽기
image = cv2.imread('Lenna.png')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 로드할 수 없습니다.")
else:
    # BGR 채널 분리
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    
    print("✓ 이미지 로드 완료!")
    print(f"이미지 크기: {image.shape}")
    print(f"Blue 채널 범위: {blue_channel.min()} ~ {blue_channel.max()}")
    print(f"Green 채널 범위: {green_channel.min()} ~ {green_channel.max()}")
    print(f"Red 채널 범위: {red_channel.min()} ~ {red_channel.max()}")
    
    # 각 채널을 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 원본 이미지 (RGB로 변환)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image (BGR)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Blue 채널
    axes[0, 1].imshow(blue_channel, cmap='Blues')
    axes[0, 1].set_title('Blue Channel', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Green 채널
    axes[0, 2].imshow(green_channel, cmap='Greens')
    axes[0, 2].set_title('Green Channel', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Red 채널
    axes[1, 0].imshow(red_channel, cmap='Reds')
    axes[1, 0].set_title('Red Channel', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Blue 채널 히스토그램
    axes[1, 1].hist(blue_channel.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    axes[1, 1].set_title('Blue Channel Histogram', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Green 채널 히스토그램
    axes[1, 2].hist(green_channel.ravel(), 256, [0, 256], color='green', alpha=0.7)
    axes[1, 2].set_title('Green Channel Histogram', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('color_channels_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 각 채널을 개별 파일로 저장
    cv2.imwrite('channel_blue.png', blue_channel)
    cv2.imwrite('channel_green.png', green_channel)
    cv2.imwrite('channel_red.png', red_channel)
    
    print("\n✓ 모든 채널이 출력되었습니다.")
    print("저장된 파일:")
    print("- color_channels_result.png (채널 비교 이미지)")
    print("- channel_blue.png (Blue 채널)")
    print("- channel_green.png (Green 채널)")
    print("- channel_red.png (Red 채널)")
