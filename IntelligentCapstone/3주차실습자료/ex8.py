import cv2
import matplotlib.pyplot as plt
import numpy as np

# Lenna.png 파일을 BGR로 읽기
image_bgr = cv2.imread('Lenna.png')

# 이미지가 제대로 로드되었는지 확인
if image_bgr is None:
    print("이미지를 로드할 수 없습니다.")
else:
    # BGR을 RGB로 변환
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # RGB 채널 분리
    red_channel = image_rgb[:, :, 0]
    green_channel = image_rgb[:, :, 1]
    blue_channel = image_rgb[:, :, 2]
    
    print("✓ 이미지 로드 완료!")
    print(f"이미지 크기: {image_rgb.shape}")
    print(f"Red 채널 범위: {red_channel.min()} ~ {red_channel.max()}")
    print(f"Green 채널 범위: {green_channel.min()} ~ {green_channel.max()}")
    print(f"Blue 채널 범위: {blue_channel.min()} ~ {blue_channel.max()}")
    
    # RGB 채널별 히스토그램 계산
    red_hist = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    blue_hist = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    
    # 결과 표시
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 원본 이미지
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image (Lenna)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Red 채널 히스토그램
    axes[0, 1].plot(red_hist, color='red', linewidth=2)
    axes[0, 1].fill_between(range(256), red_hist.flatten(), alpha=0.3, color='red')
    axes[0, 1].set_xlim([0, 256])
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Red Channel Histogram', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Green 채널 히스토그램
    axes[1, 0].plot(green_hist, color='green', linewidth=2)
    axes[1, 0].fill_between(range(256), green_hist.flatten(), alpha=0.3, color='green')
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Green Channel Histogram', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Blue 채널 히스토그램
    axes[1, 1].plot(blue_hist, color='blue', linewidth=2)
    axes[1, 1].fill_between(range(256), blue_hist.flatten(), alpha=0.3, color='blue')
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Blue Channel Histogram', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rgb_histogram_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 세 채널을 함께 표시한 히스토그램
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(red_hist, color='red', label='Red', linewidth=2, alpha=0.8)
    ax.plot(green_hist, color='green', label='Green', linewidth=2, alpha=0.8)
    ax.plot(blue_hist, color='blue', label='Blue', linewidth=2, alpha=0.8)
    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Combined RGB Histogram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rgb_combined_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ RGB 히스토그램 출력 완료!")
    print("\n저장된 파일:")
    print("- rgb_histogram_result.png (R, G, B 채널별 히스토그램)")
    print("- rgb_combined_histogram.png (합쳐진 RGB 히스토그램)")
    
    # 채널별 통계 정보
    print("\n채널별 통계 정보:")
    print(f"Red   - Mean: {red_channel.mean():.2f}, Std: {red_channel.std():.2f}")
    print(f"Green - Mean: {green_channel.mean():.2f}, Std: {green_channel.std():.2f}")
    print(f"Blue  - Mean: {blue_channel.mean():.2f}, Std: {blue_channel.std():.2f}")
