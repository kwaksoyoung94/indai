import sys
from pipeline.pipeline_manager import FireDetectionPipeline
from config import Config

if __name__ == "__main__":
    print("=======================================================")
    print("  제조 현장 외란 광원 대응 고정밀 화재 감지 파이프라인 실증")
    print("=======================================================")
    
    # 실행 시 매개변수로 RTSP 주소를 넘기거나 기본 설정을 활용합니다.
    source = Config.RTSP_URL
    if len(sys.argv) > 1:
        source = sys.argv[1]
        
    pipeline = FireDetectionPipeline()
    
    # 실시간 WebRTC/RTSP 주소로부터 영상을 불러와 프레임 단위로 쪼개어 구동
    success = pipeline.run_live_inference(source_url=source)
    
    if not success:
        print("시스템이 비정상 종료되었습니다. 네트워크 및 스트림 URL을 확인하세요.")