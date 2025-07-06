import cv2

# 카메라를 열기 (기본 내장 카메라: 0, 외장 카메라가 있다면 번호를 변경)
cap = cv2.VideoCapture(0)

# 카메라가 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    # 카메라로부터 프레임을 읽기
    ret, frame = cap.read()

    # 프레임 읽기가 실패한 경우 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 프레임을 화면에 출력
    cv2.imshow("Camera Feed", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 작업이 끝나면 자원 해제
cap.release()
cv2.destroyAllWindows()