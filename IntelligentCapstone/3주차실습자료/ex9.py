import cv2
import matplotlib.pyplot as plt
import numpy as np

# dataspyt.JPG 파일 읽기
image = cv2.imread('dataspyt.JPG')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 로드할 수 없습니다.")
    print("주의: 파일명이 'dataspyt.JPG'인지 확인해주세요.")
else:
    # BGR을 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # filter2D를 사용한 엣지 검출
    # Sobel X 커널 정의
    kernel_sobelx = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=np.float32)
    
    # Sobel Y 커널 정의
    kernel_sobely = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]], dtype=np.float32)
    
    # Laplacian 커널 정의
    kernel_laplacian = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]], dtype=np.float32)
    
    # Prewitt 커널 정의 (X 방향)
    kernel_prewittx = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]], dtype=np.float32)
    
    # filter2D를 사용하여 엣지 검출
    edges_sobelx = cv2.filter2D(blurred, cv2.CV_32F, kernel_sobelx)
    edges_sobelx = np.uint8(np.absolute(edges_sobelx))
    
    edges_sobely = cv2.filter2D(blurred, cv2.CV_32F, kernel_sobely)
    edges_sobely = np.uint8(np.absolute(edges_sobely))
    
    edges_laplacian = cv2.filter2D(blurred, cv2.CV_32F, kernel_laplacian)
    edges_laplacian = np.uint8(np.absolute(edges_laplacian))
    
    edges_prewitt = cv2.filter2D(blurred, cv2.CV_32F, kernel_prewittx)
    edges_prewitt = np.uint8(np.absolute(edges_prewitt))
    
    # Sobel X와 Y 결합
    sobel_combined = cv2.addWeighted(edges_sobelx, 0.5, edges_sobely, 0.5, 0)
    
    print("✓ 이미지 로드 및 엣지 검출 완료!")
    print(f"원본 이미지 크기: {image.shape}")
    print(f"그레이스케일 이미지 크기: {gray_image.shape}")
    
    # 결과 표시
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 원본 이미지 (RGB로 변환)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 그레이스케일 이미지
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 1].set_title('Grayscale Image', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 블러 이미지
    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title('Gaussian Blurred', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Canny 엣지 검출
    axes[1, 0].imshow(edges_laplacian, cmap='gray')
    axes[1, 0].set_title('Laplacian (filter2D)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Sobel 엣지 검출
    axes[1, 1].imshow(sobel_combined, cmap='gray')
    axes[1, 1].set_title('Sobel (filter2D)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Laplacian 엣지 검출
    axes[1, 2].imshow(edges_prewitt, cmap='gray')
    axes[1, 2].set_title('Prewitt (filter2D)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_detection_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 개별 엣지 이미지 저장
    cv2.imwrite('edge_laplacian_filter2d.png', edges_laplacian)
    cv2.imwrite('edge_sobel_filter2d.png', sobel_combined)
    cv2.imwrite('edge_prewitt_filter2d.png', edges_prewitt)
    
    print("\n✓ 엣지 검출 완료!")
    print("\n저장된 파일:")
    print("- edge_detection_result.png (모든 엣지 검출 방법 비교)")
    print("- edge_laplacian_filter2d.png (Laplacian 엣지)")
    print("- edge_sobel_filter2d.png (Sobel 엣지)")
    print("- edge_prewitt_filter2d.png (Prewitt 엣지)")
    print("\nfilter2D를 사용한 엣지 검출 방법:")
    print("1. Sobel: 그래디언트 기반의 엣지 검출 (3x3 커널)")
    print("2. Laplacian: 2차 미분 기반의 엣지 검출 (3x3 커널)")
    print("3. Prewitt: Sobel과 유사한 엣지 검출 (3x3 커널)")
