import cv2

# 웹캠 열기 (0은 기본 카메라)
cap = cv2.VideoCapture(0)

# 웹캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
else:
    # 웹캠 정보 출력
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ 웹캠 연결 완료!")
    print(f"해상도: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print("\n웹캠 출력 중... (종료하려면 'q' 키를 눌러주세요)")
    
    frame_count = 0
    
    # 웹캠 프레임 읽고 출력
    while True:
        ret, frame = cap.read()
        
        # 프레임을 제대로 읽지 못했으면 종료
        if not ret:
            print("\n✓ 웹캠을 종료합니다.")
            break
        
        frame_count += 1
        
        # 프레임 정보 표시
        text = f"Frame: {frame_count}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # 해상도 정보 표시
        resolution_text = f"Resolution: {frame_width}x{frame_height}"
        cv2.putText(frame, resolution_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # 안내문 표시
        guide_text = "Press 'q' to exit"
        cv2.putText(frame, guide_text, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)
        
        # 프레임 출력
        cv2.imshow('Webcam', frame)
        
        # 'q' 키를 누르면 종료 (1ms 대기)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n✓ 사용자가 웹캠을 종료했습니다. (총 {frame_count}프레임 캡처됨)")
            break
    
    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
