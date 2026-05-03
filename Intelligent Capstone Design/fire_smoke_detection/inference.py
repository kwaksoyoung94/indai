"""
Inference Script
Run fire and smoke detection on images, videos, or live camera feed
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from fire_smoke_detector import FireSmokeDetector


def main():
    parser = argparse.ArgumentParser(
        description='Fire and Smoke Detection System - Inference'
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['image', 'video', 'camera'],
                       help='Mode: image, video, or camera')
    
    parser.add_argument('--input', type=str,
                       help='Input file path (for image or video mode)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (optional)')
    
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID (for camera mode)')
    
    parser.add_argument('--display', type=bool, default=True,
                       help='Display results during processing')
    
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model weights')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    
    # Update model path if specified
    if args.model_path:
        config.MODEL_SAVE_PATH = args.model_path
    
    # Initialize detector
    print("Initializing Fire and Smoke Detection System...")
    detector = FireSmokeDetector(config, use_pretrained=os.path.exists(config.MODEL_SAVE_PATH))
    
    print(f"Mode: {args.mode}")
    
    if args.mode == 'image':
        if not args.input or not os.path.exists(args.input):
            print("Error: Input image path required and must exist")
            return
        
        print(f"Processing image: {args.input}")
        output_path = args.output or f"output_{Path(args.input).stem}.jpg"
        
        annotated_frame, detections = detector.process_image(args.input, output_path)
        
        if annotated_frame is not None:
            print(f"\nDetections found: {len(detections)}")
            for det in detections:
                print(f"  - {det['class'].upper()}: {det['confidence']:.2f}")
            print(f"Output saved to: {output_path}")
        
    elif args.mode == 'video':
        if not args.input or not os.path.exists(args.input):
            print("Error: Input video path required and must exist")
            return
        
        print(f"Processing video: {args.input}")
        output_path = args.output or f"output_{Path(args.input).stem}.mp4"
        
        stats = detector.process_video(args.input, output_path, args.display)
        
        if stats:
            print(f"\n=== Video Processing Statistics ===")
            print(f"Total Frames: {stats['total_frames']}")
            print(f"Fire Frames: {stats['fire_frames']} ({stats['fire_detection_rate']*100:.1f}%)")
            print(f"Smoke Frames: {stats['smoke_frames']} ({stats['smoke_detection_rate']*100:.1f}%)")
            print(f"FPS: {stats['fps']}")
            print(f"Resolution: {stats['resolution']}")
            print(f"Output saved to: {output_path}")
    
    elif args.mode == 'camera':
        print(f"Starting live camera feed (Camera ID: {args.camera_id})")
        output_path = args.output or "camera_output.mp4"
        
        stats = detector.process_camera(args.camera_id, output_path)
        
        if stats:
            print(f"\n=== Camera Processing Statistics ===")
            print(f"Total Frames: {stats['total_frames']}")
            print(f"Fire Detections: {stats['fire_frames']}")
            print(f"Smoke Detections: {stats['smoke_frames']}")
            if output_path:
                print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
