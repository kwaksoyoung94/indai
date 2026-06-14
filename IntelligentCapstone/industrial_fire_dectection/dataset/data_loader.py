import os
import cv2
import numpy as np
import tensorflow as tf
from config import Config

class FireDatasetLoader(tf.keras.utils.Sequence):
    """
    성능 검증을 위한 시계열 ROI 데이터 로더
    - 화재(Positive) 및 광학 외란/정상(Negative) 영상을 Sequence 형태로 로드
    - 메모리 오버헤드를 방지하기 위해 배치를 제너레이터 형태로 dynamic 로드
    """
    def __init__(self, data_dir, batch_size=Config.BATCH_SIZE, seq_len=Config.SEQUENCE_LENGTH, img_size=Config.IMG_SIZE, shuffle=True):
        self.data_dir = data_dir          # 데이터셋 최상위 경로 (train 또는 val)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.img_size = img_size
        self.shuffle = shuffle
        
        self.classes = ['normal_interference', 'real_fire'] # 0: 외란(정상), 1: 실제화재
        self.data_list = [] # (video_path, label) 튜플 리스트
        
        self._prepare_data_list()
        self.on_epoch_end()

    def _prepare_data_list(self):
        """ 폴더 구조를 탐색하여 영상 파일과 라벨을 매핑 """
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for video_name in os.listdir(class_dir):
                if video_name.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(class_dir, video_name)
                    self.data_list.append((video_path, label_idx))
        
        print(f" 총 {len(self.data_list)}개의 비디오 클립이 성공적으로 로드되었습니다.")

    def __len__(self):
        """ 에포크당 배치 총 개수 계산 """
        return int(np.floor(len(self.data_list) / self.batch_size))

    def on_epoch_end(self):
        """ 한 에포크가 끝날 때마다 데이터 셔플링 (학습 고도화) """
        self.indices = np.arange(len(self.data_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """ 배치 인덱스에 해당하는 10-Sequence 데이터 생성 및 반환 """
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []
        
        for idx in batch_indices:
            video_path, label = self.data_list[idx]
            frames = self._load_and_preprocess_video(video_path)
            
            # 비디오가 깨졌거나 프레임 수가 부족한 경우 예외 처리
            if frames is not None and len(frames) == self.seq_len:
                batch_x.append(frames)
                batch_y.append(label)
                
        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def _load_and_preprocess_video(self, video_path):
        """ 
        하나의 비디오 파일에서 균일한 간격으로 Config.SEQUENCE_LENGTH(10) 만큼의 프레임을 추출
        ROI 크롭 및 균일 샘플링(Low-FPS) 반영 
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.seq_len:
            cap.release()
            return None
        
        # 전체 영상에서 10개의 프레임을 균일한 간격으로 추출하기 위한 인덱스 계산
        frame_indices = np.linspace(0, total_frames - 1, self.seq_len, dtype=int)
        
        sampled_frames = []
        current_frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame_idx in frame_indices:
                # 1. 이미지 크기 조정 (224, 224)
                # 실제 현장 파이프라인에서는 YOLO ROI가 들어오지만, 
                # 학습 단계에서는 미리 크롭되어 저장된 외란/화재 영상 데이터를 정규화합니다.
                frame_resized = cv2.resize(frame, self.img_size)
                
                # 2. 이미지 정규화 (Min-Max Scaling)
                frame_normalized = frame_resized / 255.0
                sampled_frames.append(frame_normalized)
                
                if len(sampled_frames) == self.seq_len:
                    break
                    
            current_frame_idx += 1
            
        cap.release()
        
        if len(sampled_frames) != self.seq_len:
            return None
            
        return np.array(sampled_frames) # (10, 224, 224, 3) 형태로 리턴