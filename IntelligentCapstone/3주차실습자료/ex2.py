import cv2

# 비디오 파일 열기
cap = cv2.VideoCapture('test_video.mp4')

# 비디오가 제대로 열렸는지 확인
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
else:
    # 비디오 정보 출력
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ 비디오 파일 로드 완료!")
    print(f"해상도: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"전체 프레임 수: {frame_count}")
    print(f"영상 길이: {frame_count / fps:.2f}초")
    print("\n재생 중... (종료하려면 'q' 키를 눌러주세요)")
    
    frame_number = 0
    
    # 비디오 프레임 읽고 출력
    while True:
        ret, frame = cap.read()
        
        # 프레임을 제대로 읽지 못했으면 종료
        if not ret:
            print("\n✓ 비디오 재생 완료!")
            break
        
        frame_number += 1
        
        # 프레임 정보 표시
        text = f"Frame: {frame_number}/{frame_count}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # 프레임 출력
        cv2.imshow('Video Playback', frame)
        
        # 'q' 키를 누르면 종료 (1ms 대기)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            print("\n✓ 사용자가 재생을 중단했습니다.")
            break
    
    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
