import os
import cv2
import yt_dlp
from config import Config # 이전 단계에서 정의한 Config 클래스 활용

def download_youtube_video(video_url, output_filename='temp_video'):
    """ 유튜브 영상을 최고 화질로 다운로드합니다. """
    outtmpl = f"{output_filename}.%(ext)s"
    ydl_opts = {
        'format': 'bestvideo/best',
        'outtmpl': outtmpl,
        'merge_output_format': 'mp4',
    }
    
    print(f"유튜브 영상 다운로드 중: {video_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info)
        # 확장자가 바뀔 수 있으므로 실제 저장된 파일명 반환
        if not os.path.exists(filename) and os.path.exists(f"{output_filename}.mp4"):
            filename = f"{output_filename}.mp4"
        return filename

def split_video_into_clips(video_path, output_dir, prefix='clip'):
    """ 
    다운로드한 영상을 프레임 단위로 쪼개어, 
    Config.SEQUENCE_LENGTH (10프레임) 단위의 독립된 비디오 클립 파일로 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 비디오 저장을 위한 코덱 설정 (MP4V)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    frame_buffer = []
    clip_count = 0
    frame_idx = 0
    
    print(f"영상을 프레임 단위로 분해 및 {Config.SEQUENCE_LENGTH}프레임 클립 생성 중...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 초당 10프레임(Low-FPS) 수준으로 프레임을 쪼개어 수집하기 위해 원본이 30fps라면 3프레임마다 1장씩만 샘플링하여 데이터 효율 극대화
        frame_skip = max(1, int(fps / Config.FPS_LIMIT))
        if frame_idx % frame_skip == 0:
            frame_buffer.append(frame)
            
            # 버퍼에 10프레임이 쌓이면 하나의 독립된 비디오 클립(.mp4)으로 저장
            if len(frame_buffer) == Config.SEQUENCE_LENGTH:
                clip_path = os.path.join(output_dir, f"{prefix}_{clip_count:04d}.mp4")
                
                # 새로운 클립 파일 생성
                out = cv2.VideoWriter(clip_path, fourcc, Config.FPS_LIMIT, (width, height))
                for f in frame_buffer:
                    out.write(f)
                out.release()
                
                clip_count += 1
                # 슬라이딩 윈도우 방식이 아닌 중복 없는 클립 생성을 위해 버퍼 비움
                frame_buffer = [] 
                
        frame_idx += 1
        
    cap.release()
    print(f"총 {clip_count}개의 외란 광원 클립이 {output_dir} 폴더에 생성되었습니다.")

if __name__ == "__main__":
    # 1. 수집하고자 하는 유튜브 외란 영상 URL 리스트
    # 공장 용접(Welding Arc), 경광등(Strobe Light)
    urls_to_collect = [
        {"url": "https://www.youtube.com/watch?v=CeRRsDlQkxA", "prefix": "welding"},
        {"url": "https://www.youtube.com/watch?v=5GOmuoyH-7o", "prefix": "strobe"},
        {"url":"https://www.youtube.com/watch?v=srB_ormB7e8", "prefix": "strobe"},
        {"url":"https://www.youtube.com/watch?v=d86HtVaitEU", "prefix": "strobe"},

    ]
    
    # 2. 저장할 대상 폴더 설정 (data_loader가 바라보는 경로)
    target_dir = "dataset/train/normal_interference"
    
    for i, item in enumerate(urls_to_collect):
        try:
            temp_file = f"temp_download_{i}"
            # 다운로드 수행
            downloaded_file = download_youtube_video(item["url"], output_filename=temp_file)
            
            # 10프레임 단위 쪼개기 수행
            split_video_into_clips(downloaded_file, target_dir, prefix=item["prefix"])
            
            # 임시 원본 파일 삭제 (용량 확보)
            if os.path.exists(downloaded_file):
                os.remove(downloaded_file)
                
        except Exception as e:
            print(f"{item['url']} 처리 중 오류 발생: {e}")