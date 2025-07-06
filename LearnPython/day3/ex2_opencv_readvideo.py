import cv2

# 비디오 파일 경로
video_path = 'test_video.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 비디오가 열리지 않는 경우 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 프레임을 한 장씩 읽고 보여줌
while True:
    ret, frame = cap.read()
    if not ret:
        # 더 이상 프레임이 없으면 종료
        print("Video has ended.")
        break

    # 프레임 표시
    cv2.imshow('Video', frame)

    # 키보드에서 'q'를 누르면 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("Playback terminated by user.")
        break

# 비디오 캡처와 모든 창 닫기
cap.release()
cv2.destroyAllWindows()