"""
Configuration for Fire and Smoke Detection System
Based on the paper: "False Positive Decremented Research for Fire and Smoke Detection 
in Surveillance Camera using Spatial and Temporal Features Based on Deep Learning"
"""

class Config:
    # Model Configuration
    MODEL_NAME = "faster_rcnn_resnet"
    INPUT_IMAGE_SIZE = (300, 300)
    NUM_CLASSES = 3  # background, fire, smoke
    
    # Training Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    VALIDATION_SPLIT = 0.3
    TRAIN_DATA_RATIO = 0.7
    
    # Detection Thresholds (Global)
    SSIM_THRESHOLD = 0.9          # S_k < th1
    MSE_THRESHOLD = 1000          # M_k < th2
    FRAME_DIFF_THRESHOLD = 0.05   # A_k < th3
    
    # Fire Detection Thresholds (Local)
    FIRE_MSE_THRESHOLD = 800.0                # f_th1
    FIRE_FRAME_DIFF_THRESHOLD = 0.03         # f_th2
    FIRE_HISTOGRAM_THRESHOLD = 0.5            # f_th3
    FIRE_WAVELET_ENERGY_THRESHOLD = 100.0    # f_th4
    
    # Smoke Detection Thresholds (Local)
    SMOKE_MSE_THRESHOLD = 500.0               # s_th1
    SMOKE_FRAME_DIFF_THRESHOLD = 0.08        # s_th2
    SMOKE_HISTOGRAM_THRESHOLD = 0.6           # s_th3
    SMOKE_WAVELET_ENERGY_THRESHOLD = 50.0    # s_th4
    SMOKE_SD_THRESHOLD_1 = 0.3               # s_th5
    SMOKE_SD_THRESHOLD_2 = 0.4               # s_th6
    SMOKE_SD_THRESHOLD_3 = 0.2               # s_th7
    SMOKE_SD_THRESHOLD_4 = 5.0               # s_th8
    
    # Color Space Ranges for Smoke (Gray to White)
    SMOKE_RGB_MIN = 80       # τ1
    SMOKE_RGB_MAX = 250      # τ2 (upper)
    
    # Color Space Ranges for Fire (HSV)
    FIRE_HSV_H_MIN = 0
    FIRE_HSV_H_MAX = 40
    FIRE_HSV_S_MIN = 100
    FIRE_HSV_S_MAX = 255
    FIRE_HSV_V_MIN = 80
    FIRE_HSV_V_MAX = 255
    
    # Feature Extraction
    WAVELET_TYPE = 'db1'       # Daubechies wavelet
    FRAME_BUFFER_SIZE = 10     # Number of frames to keep in buffer
    
    # Detection Confidence
    DETECTION_CONFIDENCE_THRESHOLD = 0.3  # Faster R-CNN confidence threshold
    
    # Post-processing
    NMS_THRESHOLD = 0.5  # Non-maximum suppression threshold
    MIN_BBOX_AREA = 20   # Minimum bounding box area (pixels)
    
    # Paths
    MODEL_SAVE_PATH = "weights/fire_smoke_model.h5"
    FROZEN_GRAPH_PATH = "weights/frozen_inference_graph.pb"
    
    # Logging
    LOG_LEVEL = "INFO"
