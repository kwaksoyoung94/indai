import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lenna.png 파일을 BGR로 읽기
image = cv2.imread('Lenna.png')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 로드할 수 없습니다.")
else:
    # BGR을 YUV로 변환
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # YUV 채널 분리
    y_channel = yuv_image[:, :, 0]
    u_channel = yuv_image[:, :, 1]
    v_channel = yuv_image[:, :, 2]
    
    print("✓ 이미지 로드 및 YUV 변환 완료!")
    print(f"이미지 크기: {image.shape}")
    print(f"YUV 크기: {yuv_image.shape}")
    print(f"\nY 채널 범위: {y_channel.min()} ~ {y_channel.max()}")
    print(f"U 채널 범위: {u_channel.min()} ~ {u_channel.max()}")
    print(f"V 채널 범위: {v_channel.min()} ~ {v_channel.max()}")
    print("\n주의: OpenCV YUV에서 Y, U, V는 모두 0-255 범위입니다.")
    
    # 각 채널을 시각화
    fig = plt.figure(figsize=(15, 10))
    
    # 원본 이미지 (RGB로 변환)
    ax1 = plt.subplot(2, 3, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image (BGR)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Hue 채널 (색상)
    ax2 = plt.subplot(2, 3, 2)
    y_display = ax2.imshow(y_channel, cmap='gray')
    ax2.set_title('Y Channel (Luma)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(y_display, ax=ax2, label='Y (0-255)')
    
    # Saturation 채널 (채도)
    ax3 = plt.subplot(2, 3, 3)
    u_display = ax3.imshow(u_channel, cmap='gray')
    ax3.set_title('U Channel (Chrominance)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(u_display, ax=ax3, label='U (0-255)')
    
    # Value 채널 (명도)
    ax4 = plt.subplot(2, 3, 4)
    v_display = ax4.imshow(v_channel, cmap='gray')
    ax4.set_title('V Channel (Chrominance)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(v_display, ax=ax4, label='V (0-255)')
    
    # Y 채널 히스토그램
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(y_channel.ravel(), 256, [0, 256], color='gray', alpha=0.7, edgecolor='black')
    ax5.set_title('Y Channel Histogram', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Y Value (0-255)')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    # U와 V 채널 히스토그램 (함께 표시)
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(u_channel.ravel(), 256, [0, 256], alpha=0.5, label='U Channel', color='blue', edgecolor='black')
    ax6.hist(v_channel.ravel(), 256, [0, 256], alpha=0.5, label='V Channel', color='red', edgecolor='black')
    ax6.set_title('U & V Channel Histograms', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Pixel Value (0-255)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yuv_channels_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 각 채널을 개별 파일로 저장
    cv2.imwrite('channel_y.png', y_channel)
    cv2.imwrite('channel_u.png', u_channel)
    cv2.imwrite('channel_v.png', v_channel)
    
    # YUV 이미지 저장
    cv2.imwrite('lenna_yuv.png', yuv_image)
    
    print("\n✓ 모든 YUV 채널이 출력되었습니다.")
    print("\n저장된 파일:")
    print("- yuv_channels_result.png (채널 비교 이미지)")
    print("- channel_y.png (Y 채널)")
    print("- channel_u.png (U 채널)")
    print("- channel_v.png (V 채널)")
    print("- lenna_yuv.png (YUV 변환 이미지)")
