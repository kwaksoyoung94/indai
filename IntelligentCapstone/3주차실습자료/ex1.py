import cv2

# Lenna.png 파일 읽기
image = cv2.imread('Lenna.png')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 로드할 수 없습니다.")
else:
    # 이미지 정보 출력
    print("✓ 이미지 로드 완료!")
    print(f"이미지 크기: {image.shape}")
    print(f"가로: {image.shape[1]}px, 세로: {image.shape[0]}px, 채널: {image.shape[2]}")
    
    # 이미지 출력
    cv2.imshow('Lenna Image', image)
    
    print("\n윈도우가 열렸습니다. 아무 키를 눌러주세요.")
    
    # 키 입력 대기 (0은 무한 대기)
    cv2.waitKey(0)
    
    # 이미지 저장
    cv2.imwrite('lenna_saved.png', image)
    print("✓ 이미지가 'lenna_saved.png'로 저장되었습니다.")
    
    # 윈도우 종료
    cv2.destroyAllWindows()
