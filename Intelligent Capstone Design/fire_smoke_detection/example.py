"""
Example Usage of Fire and Smoke Detection System
간단한 사용 예제
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from config import Config
from fire_smoke_detector import FireSmokeDetector
from feature_extraction import FeatureExtractor


def example_1_basic_detection():
    """
    예제 1: 기본 탐지 사용
    Basic detection example
    """
    print("=" * 60)
    print("예제 1: 기본 화재/연기 탐지")
    print("=" * 60)
    
    # 초기화
    config = Config()
    detector = FireSmokeDetector(config)
    
    # 더미 이미지 생성 (실제로는 실제 이미지 사용)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 프레임 처리
    annotated_frame, detections = detector.process_frame(dummy_frame)
    
    print(f"\n탐지된 객체 수: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  [{i+1}] 클래스: {det['class']}")
        print(f"      신뢰도: {det['confidence']:.2f}")
        print(f"      위치: {det['bbox']}")


def example_2_feature_extraction():
    """
    예제 2: 특징 추출
    Feature extraction example
    """
    print("\n" + "=" * 60)
    print("예제 2: 특징 추출")
    print("=" * 60)
    
    config = Config()
    extractor = FeatureExtractor(config)
    
    # 더미 프레임
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 전역 특징 추출
    print("\n1. 전역 특징 (Global Features)")
    print("-" * 40)
    
    ssim_val = extractor.calculate_ssim(frame1, frame2)
    print(f"SSIM 점수: {ssim_val:.4f}")
    
    mse_val = extractor.calculate_mse(frame1, frame2)
    print(f"MSE 값: {mse_val:.2f}")
    
    diff_val = extractor.calculate_frame_difference(frame1, frame2)
    print(f"프레임 차이: {diff_val:.4f}")
    
    # 지역 특징 추출
    print("\n2. 지역 특징 (Local Features)")
    print("-" * 40)
    
    smoke_hist = extractor.calculate_color_histogram(frame1, color_type='smoke')
    fire_hist = extractor.calculate_color_histogram(frame1, color_type='fire')
    print(f"연기 색상 히스토그램: {smoke_hist:.4f}")
    print(f"화재 색상 히스토그램: {fire_hist:.4f}")
    
    cv_value = extractor.calculate_coefficient_of_variation(frame1)
    print(f"Coefficient of Variation: {cv_value:.4f}")
    
    wavelet_energy = extractor.calculate_wavelet_energy(frame1)
    print(f"Wavelet 에너지: {wavelet_energy:.4f}")


def example_3_batch_processing():
    """
    예제 3: 배치 처리
    Batch processing example
    """
    print("\n" + "=" * 60)
    print("예제 3: 배치 프레임 처리")
    print("=" * 60)
    
    config = Config()
    detector = FireSmokeDetector(config)
    
    # 10개의 더미 프레임 생성
    num_frames = 10
    print(f"\n{num_frames}개의 프레임 처리 중...")
    
    all_detections = []
    
    for i in range(num_frames):
        # 프레임 생성
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 처리
        _, detections = detector.process_frame(frame)
        all_detections.append(detections)
        
        print(f"  프레임 {i+1}/{num_frames}: {len(detections)}개 탐지")
    
    # 통계
    stats = detector.get_statistics()
    print(f"\n처리 통계:")
    print(f"  총 프레임: {stats['total_frames_processed']}")
    print(f"  화재 탐지: {stats['fire_detections']}")
    print(f"  연기 탐지: {stats['smoke_detections']}")
    print(f"  화재 탐지율: {stats['fire_detection_rate']*100:.1f}%")
    print(f"  연기 탐지율: {stats['smoke_detection_rate']*100:.1f}%")


def example_4_config_explanation():
    """
    예제 4: 설정 파라미터 설명
    Configuration parameters explanation
    """
    print("\n" + "=" * 60)
    print("예제 4: 설정 파라미터")
    print("=" * 60)
    
    config = Config()
    
    print("\n1. 전역 특징 임계값 (Global Thresholds)")
    print("-" * 40)
    print(f"SSIM 임계값: {config.SSIM_THRESHOLD}")
    print(f"  → SSIM < {config.SSIM_THRESHOLD}이면 움직임 있음")
    
    print(f"MSE 임계값: {config.MSE_THRESHOLD}")
    print(f"  → MSE < {config.MSE_THRESHOLD}이면 움직임 있음")
    
    print(f"프레임 차이 임계값: {config.FRAME_DIFF_THRESHOLD}")
    print(f"  → 차이 < {config.FRAME_DIFF_THRESHOLD}이면 움직임 있음")
    
    print("\n2. 화재 지역 임계값 (Fire Local Thresholds)")
    print("-" * 40)
    print(f"MSE 임계값: {config.FIRE_MSE_THRESHOLD}")
    print(f"프레임 차이 임계값: {config.FIRE_FRAME_DIFF_THRESHOLD}")
    print(f"히스토그램 임계값: {config.FIRE_HISTOGRAM_THRESHOLD}")
    print(f"Wavelet 에너지 임계값: {config.FIRE_WAVELET_ENERGY_THRESHOLD}")
    
    print("\n3. 연기 지역 임계값 (Smoke Local Thresholds)")
    print("-" * 40)
    print(f"MSE 임계값: {config.SMOKE_MSE_THRESHOLD}")
    print(f"프레임 차이 임계값: {config.SMOKE_FRAME_DIFF_THRESHOLD}")
    print(f"히스토그램 임계값: {config.SMOKE_HISTOGRAM_THRESHOLD}")
    print(f"Wavelet 에너지 임계값: {config.SMOKE_WAVELET_ENERGY_THRESHOLD}")
    
    print("\n4. 색상 범위")
    print("-" * 40)
    print(f"연기 RGB 범위: {config.SMOKE_RGB_MIN}-{config.SMOKE_RGB_MAX}")
    print(f"화재 HSV H 범위: {config.FIRE_HSV_H_MIN}-{config.FIRE_HSV_H_MAX}")
    print(f"화재 HSV S 범위: {config.FIRE_HSV_S_MIN}-{config.FIRE_HSV_S_MAX}")
    print(f"화재 HSV V 범위: {config.FIRE_HSV_V_MIN}-{config.FIRE_HSV_V_MAX}")


def example_5_algorithm_flow():
    """
    예제 5: 알고리즘 흐름 설명
    Algorithm flow explanation
    """
    print("\n" + "=" * 60)
    print("예제 5: 알고리즘 흐름 (Algorithm Flow)")
    print("=" * 60)
    
    flow = """
    
1. 프레임 입력 (Frame Input)
   ↓
2. 전역 특징 검증 (Global Feature Validation)
   - SSIM 계산: S_k = SSIM(f_i, f_j)
   - MSE 계산: M_k = MSE(f_i, f_j)
   - 3-Frame Difference: A_k = diff(f_i, f_j)
   - 결정: FSG = 1 if S_k < th1 AND M_k < th2 AND A_k < th3
   ↓
3. 움직임 감지 여부?
   ├─ NO → 프레임 버퍼에만 저장, 다음 프레임
   └─ YES ↓
4. Faster R-CNN 객체 탐지 (Object Detection)
   - 화재/연기 후보 영역 생성
   - 신뢰도 기반 필터링
   ↓
5. 지역 특징 검증 (Local Feature Validation)
   각 탐지 영역(ROI)에 대해:
   
   화재 검증 (Fire):
   - MSE: M_k < f_th1?
   - 프레임 차이: A_k < f_th2?
   - 히스토그램: H_sum*F < f_th3? (HSV H,S,V 범위)
   - Wavelet: WE_k < f_th4?
   → 모두 만족하면 FL = 1
   
   연기 검증 (Smoke):
   - MSE: M_k > s_th1?
   - 프레임 차이: A_k > s_th2?
   - 히스토그램: H_sum*F > s_th3? (RGB 회색 범위)
   - Wavelet: WE_k < s_th4?
   → 모두 만족하면 SL = 1
   ↓
6. 최종 판정 (Final Decision)
   FD = 1 if FL > 0 AND SL > 0
   
   FD = 1 → 화재/연기 탐지 (Detection)
   FD = 0 → 거짓 양성 제거 (False Positive Removal)
   ↓
7. 탐지 결과 반환
    
거짓 양성 감소율: 약 99.9% (유지: 화재/연기 탐지율 > 95%)
    """
    print(flow)


def main():
    """Run all examples"""
    print("\n")
    print("#" * 60)
    print("# 화재/연기 탐지 시스템 - 예제 코드")
    print("# Fire and Smoke Detection System - Examples")
    print("#" * 60)
    
    try:
        example_1_basic_detection()
        example_2_feature_extraction()
        example_3_batch_processing()
        example_4_config_explanation()
        example_5_algorithm_flow()
        
        print("\n" + "#" * 60)
        print("# 모든 예제 실행 완료!")
        print("# All examples completed successfully!")
        print("#" * 60)
        print()
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
