"""
Extract frames from video files
영상에서 프레임 추출
"""

import cv2
import os
import sys
from pathlib import Path


def extract_frames(video_path, output_dir, class_name='fire', fps=1):
    """
    Extract frames from video at specified fps
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        class_name: Class name (fire, smoke, non_fire)
        fps: Frames per second to extract (1 = 1 frame per second)
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ 오류: {video_path} 파일을 찾을 수 없습니다")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 오류: {video_path}를 열 수 없습니다")
        return False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    print(f"\n📹 영상 정보:")
    print(f"  파일: {video_path}")
    print(f"  원본 FPS: {original_fps}")
    print(f"  총 프레임: {total_frames}")
    print(f"  재생 시간: {duration:.1f}초")
    
    # Calculate interval
    if original_fps > 0:
        interval = int(original_fps / fps)
    else:
        interval = 1
    
    print(f"\n⚙️  설정:")
    print(f"  추출 FPS: {fps}")
    print(f"  간격: {interval} 프레임마다 1개 추출")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    extracted = 0
    
    print(f"\n📂 저장 위치: {output_dir}")
    print(f"🔄 프레임 추출 중...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % interval == 0:
            filename = os.path.join(output_dir, f"{class_name}_{extracted:05d}.jpg")
            cv2.imwrite(filename, frame)
            extracted += 1
            
            # Progress
            if extracted % 50 == 0:
                print(f"  ✓ {extracted}개 프레임 추출 완료...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n✅ 완료!")
    print(f"  추출된 프레임: {extracted}개")
    print(f"  저장 위치: {output_dir}")
    
    return True


def process_videos(videos_dict):
    """
    Process multiple videos
    
    Args:
        videos_dict: Dictionary with format:
        {
            'class_name': [
                ('video_path', 'output_dir'),
                ...
            ]
        }
    """
    for class_name, video_list in videos_dict.items():
        print(f"\n{'='*60}")
        print(f"클래스: {class_name}")
        print(f"{'='*60}")
        
        for i, (video_path, output_dir) in enumerate(video_list, 1):
            print(f"\n[{i}/{len(video_list)}]")
            extract_frames(video_path, output_dir, class_name, fps=1)


def main():
    """Main function"""
    
    print("="*60)
    print("🎬 영상에서 프레임 추출")
    print("="*60)
    
    # Get arguments
    if len(sys.argv) < 2:
        print("\n사용 방법:")
        print("  python extract_frames.py <video_path> [output_dir] [class_name]")
        print("\n예시:")
        print("  python extract_frames.py fire_video.mp4 data/train/fire fire")
        print("  python extract_frames.py smoke_video.mp4 data/train/smoke smoke")
        print("  python extract_frames.py office_video.mp4 data/train/non_fire non_fire")
        
        print("\n또는 현재 폴더의 모든 영상 처리:")
        print("  1. extract_all()을 사용하거나")
        print("  2. videos_dict를 설정 후 process_videos() 호출")
        
        return
    
    # Parse arguments
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else f"data/train/{Path(video_path).stem}"
    class_name = sys.argv[3] if len(sys.argv) > 3 else Path(video_path).stem
    
    # Extract frames
    success = extract_frames(video_path, output_dir, class_name, fps=1)
    
    if success:
        # Show dataset statistics
        print(f"\n📊 현재 데이터셋 상태:")
        print_dataset_stats()


def extract_all():
    """Extract all videos in current directory"""
    videos = {
        'fire': [],
        'smoke': [],
        'non_fire': []
    }
    
    # Find video files
    current_dir = os.getcwd()
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    
    for filename in os.listdir(current_dir):
        if filename.lower().endswith(video_extensions):
            filepath = os.path.join(current_dir, filename)
            
            # Determine class based on filename
            if 'fire' in filename.lower():
                videos['fire'].append((filepath, 'data/train/fire'))
            elif 'smoke' in filename.lower():
                videos['smoke'].append((filepath, 'data/train/smoke'))
            else:
                videos['non_fire'].append((filepath, 'data/train/non_fire'))
    
    if sum(len(v) for v in videos.values()) == 0:
        print("현재 폴더에서 영상 파일을 찾을 수 없습니다")
        return
    
    process_videos(videos)


def print_dataset_stats():
    """Print current dataset statistics"""
    base_dir = 'data/train'
    
    if not os.path.exists(base_dir):
        print("  아직 데이터셋이 없습니다")
        return
    
    total = 0
    for class_name in ['fire', 'smoke', 'non_fire']:
        class_dir = os.path.join(base_dir, class_name)
        
        if not os.path.exists(class_dir):
            count = 0
        else:
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        total += count
        print(f"  {class_name}: {count}개 이미지")
    
    print(f"  {'─'*30}")
    print(f"  합계: {total}개 이미지")


if __name__ == "__main__":
    main()
