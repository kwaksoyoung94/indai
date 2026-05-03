"""
Feature Extraction Module
Implements spatial and temporal features for reducing false positives:
- SSIM (Structural Similarity)
- MSE (Mean Square Error)
- 3-Frame Difference
- Color Histogram
- Wavelet Transform
- Coefficient of Variation
"""

import numpy as np
import cv2
from scipy import signal
from skimage.metrics import structural_similarity as ssim
import pywt


class FeatureExtractor:
    """Extracts spatial and temporal features from video frames"""
    
    def __init__(self, config):
        self.config = config
        self.frame_buffer = []
    
    # ==================== GLOBAL FEATURES ====================
    
    def calculate_ssim(self, frame1, frame2):
        """
        Calculate Structural Similarity Index (SSIM) between two frames
        
        SSIM = [L(x,y)]^α * [M(x,y)]^β * [N(x,y)]^γ
        where α = β = γ = 1.0
        
        Returns:
            float: SSIM value between 0 and 1 (1 = identical)
        """
        # Convert to grayscale if color image
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize to same size if needed
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Calculate SSIM
        score, _ = ssim(frame1, frame2, full=True)
        return float(score)
    
    def calculate_mse(self, frame1, frame2):
        """
        Calculate Mean Square Error between two frames
        
        MSE = (1/N) * Σ(x_i - y_i)^2
        
        Returns:
            float: MSE value (0 = identical)
        """
        # Convert to grayscale if color image
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize to same size if needed
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Calculate MSE
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        return float(mse)
    
    def calculate_frame_difference(self, frame1, frame2, threshold=30):
        """
        Calculate frame difference using 3-frame difference algorithm
        
        A_k = diff(f_i, f_j)
        
        Returns:
            float: Ratio of different pixels (0 to 1)
        """
        # Convert to grayscale if color image
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize to same size if needed
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Calculate absolute difference
        diff = cv2.absdiff(frame1, frame2)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate ratio of different pixels
        ratio = np.sum(thresh > 0) / thresh.size
        return float(ratio)
    
    # ==================== LOCAL FEATURES ====================
    
    def calculate_color_histogram(self, frame, roi=None, color_type='smoke'):
        """
        Calculate RGB/HSV color histogram for smoke or fire detection
        
        For Smoke: Gray to white colors (C_L: 80-150, C_H: 180-250)
        For Fire: HSV range (H: 0-40, S: 100-255, V: 80-255)
        
        Args:
            frame: Input frame
            roi: Region of interest (bounding box) or None for whole frame
            color_type: 'smoke' or 'fire'
        
        Returns:
            float: Histogram score (0 to 1)
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
        
        if color_type == 'smoke':
            return self._smoke_histogram(frame)
        elif color_type == 'fire':
            return self._fire_histogram(frame)
        else:
            return 0.0
    
    def _smoke_histogram(self, frame):
        """Calculate smoke color histogram (grayscale range)"""
        # Convert to RGB if BGR
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate average RGB
        if len(frame.shape) == 3:
            b, g, r = cv2.split(frame)
            c = (r.astype(float) + g.astype(float) + b.astype(float)) / 3.0
        else:
            c = frame.astype(float)
        
        # Check gray range: 80-250
        mask = (c >= self.config.SMOKE_RGB_MIN) & (c <= self.config.SMOKE_RGB_MAX)
        
        # Calculate histogram
        bins = np.arange(256)
        hist = np.histogram(c[mask], bins=256, range=(0, 256))[0]
        
        # Normalize by image size
        h_sum = np.sum(hist) / (frame.shape[0] * frame.shape[1])
        return float(h_sum)
    
    def _fire_histogram(self, frame):
        """Calculate fire color histogram (HSV range)"""
        # Convert to HSV
        if len(frame.shape) == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            return 0.0
        
        # Extract HSV channels
        h, s, v = cv2.split(hsv)
        
        # Apply HSV ranges
        mask = (h >= self.config.FIRE_HSV_H_MIN) & (h <= self.config.FIRE_HSV_H_MAX) & \
               (s >= self.config.FIRE_HSV_S_MIN) & (s <= self.config.FIRE_HSV_S_MAX) & \
               (v >= self.config.FIRE_HSV_V_MIN) & (v <= self.config.FIRE_HSV_V_MAX)
        
        # Calculate histogram
        bins = np.arange(256)
        hist = np.histogram(h[mask], bins=256, range=(0, 256))[0]
        
        # Normalize by image size
        h_sum = np.sum(hist) / (frame.shape[0] * frame.shape[1])
        return float(h_sum)
    
    def calculate_coefficient_of_variation(self, frame, roi=None, color_channel='R'):
        """
        Calculate Coefficient of Variation (CV)
        
        CV = σ / m
        where σ is standard deviation and m is mean
        
        Lower CV indicates smoke/fire, higher CV indicates false alarm
        
        Returns:
            float: CV value
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
        
        # Extract color channel
        if len(frame.shape) == 3:
            if color_channel == 'R':
                channel = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[:, :, 0]
            elif color_channel == 'Y':
                # Convert to YCbCr and extract Y
                ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                channel = ycbcr[:, :, 0]
            else:
                channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            channel = frame
        
        # Calculate CV
        mean = np.mean(channel.astype(float))
        if mean == 0:
            return 0.0
        
        std = np.std(channel.astype(float))
        cv = std / mean
        return float(cv)
    
    def calculate_wavelet_energy(self, frame, roi=None, wavelet_type='db1'):
        """
        Calculate Wavelet Transform Energy
        
        E(x,y) = sqrt(LH(x,y)^2 + HL(x,y)^2 + HH(x,y)^2)
        where LH, HL, HH are high-frequency components
        
        Returns:
            float: Average wavelet energy
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply 2D Discrete Wavelet Transform
        try:
            # Use 'db1' (Daubechies) wavelet
            coeffs = pywt.dwt2(gray.astype(float), wavelet_type)
            cA, (cH, cV, cD) = coeffs
            
            # Calculate energy: sqrt(cH^2 + cV^2 + cD^2)
            energy = np.sqrt(np.mean(cH**2) + np.mean(cV**2) + np.mean(cD**2))
            return float(energy)
        except Exception as e:
            print(f"Wavelet transform error: {e}")
            return 0.0
    
    def add_frame_to_buffer(self, frame):
        """Add frame to buffer (keep last N frames)"""
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.config.FRAME_BUFFER_SIZE:
            self.frame_buffer.pop(0)
    
    def get_frame_buffer(self):
        """Get current frame buffer"""
        return self.frame_buffer.copy()
    
    def clear_buffer(self):
        """Clear frame buffer"""
        self.frame_buffer.clear()


class GlobalFeatureValidator:
    """Validates global frame features to detect motion"""
    
    def __init__(self, config):
        self.config = config
        self.extractor = FeatureExtractor(config)
    
    def validate_global_features(self, frame_current, frame_prev):
        """
        Validate global features (motion detection)
        
        FSG = 1 if S_k < th1 AND M_k < th2 AND A_k < th3, else 0
        where:
        - S_k = SSIM(f_i, f_j)
        - M_k = MSE(f_i, f_j)
        - A_k = diff(f_i, f_j)
        """
        if frame_prev is None:
            return False
        
        # Calculate features
        ssim_val = self.extractor.calculate_ssim(frame_prev, frame_current)
        mse_val = self.extractor.calculate_mse(frame_prev, frame_current)
        diff_val = self.extractor.calculate_frame_difference(frame_prev, frame_current)
        
        # Check thresholds
        is_motion = (ssim_val < self.config.SSIM_THRESHOLD and
                    mse_val < self.config.MSE_THRESHOLD and
                    diff_val < self.config.FRAME_DIFF_THRESHOLD)
        
        return is_motion


class LocalFeatureValidator:
    """Validates local features (ROI) for fire and smoke detection"""
    
    def __init__(self, config):
        self.config = config
        self.extractor = FeatureExtractor(config)
    
    def validate_fire_local(self, roi_frame, roi_bbox, frame_history):
        """
        Validate local fire features
        
        FL = 1 if M_k < f_th1 AND A_k < f_th2 AND H_sum*F < f_th3 AND WE_k < f_th4
        """
        if frame_history is None or len(frame_history) < 2:
            return 0
        
        frame_prev = frame_history[-2]
        frame_curr = frame_history[-1]
        
        # Calculate local features
        mse_val = self.extractor.calculate_mse(frame_prev[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]],
                                               frame_curr[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]])
        
        diff_val = self.extractor.calculate_frame_difference(
            frame_prev[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]],
            frame_curr[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]])
        
        hist_val = self.extractor.calculate_color_histogram(roi_frame, color_type='fire')
        wavelet_val = self.extractor.calculate_wavelet_energy(roi_frame, wavelet_type='db1')
        
        # Check thresholds
        is_fire = (mse_val < self.config.FIRE_MSE_THRESHOLD and
                   diff_val < self.config.FIRE_FRAME_DIFF_THRESHOLD and
                   hist_val > self.config.FIRE_HISTOGRAM_THRESHOLD and
                   wavelet_val < self.config.FIRE_WAVELET_ENERGY_THRESHOLD)
        
        return 1 if is_fire else 0
    
    def validate_smoke_local(self, roi_frame, roi_bbox, frame_history):
        """
        Validate local smoke features with complex formula
        
        SL = 1 if M_k > s_th1 AND A_k > s_th2 AND H_sum*F > s_th3 AND WE_k < s_th4
        """
        if frame_history is None or len(frame_history) < 2:
            return 0
        
        frame_prev = frame_history[-2]
        frame_curr = frame_history[-1]
        
        # Calculate local features
        mse_val = self.extractor.calculate_mse(frame_prev[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]],
                                               frame_curr[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]])
        
        diff_val = self.extractor.calculate_frame_difference(
            frame_prev[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]],
            frame_curr[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]])
        
        hist_val = self.extractor.calculate_color_histogram(roi_frame, color_type='smoke')
        wavelet_val = self.extractor.calculate_wavelet_energy(roi_frame, wavelet_type='db1')
        
        cv_smoke = self.extractor.calculate_coefficient_of_variation(roi_frame, color_channel='Y')
        
        # Check thresholds
        is_smoke = (mse_val > self.config.SMOKE_MSE_THRESHOLD and
                    diff_val > self.config.SMOKE_FRAME_DIFF_THRESHOLD and
                    hist_val > self.config.SMOKE_HISTOGRAM_THRESHOLD and
                    wavelet_val < self.config.SMOKE_WAVELET_ENERGY_THRESHOLD)
        
        return 1 if is_smoke else 0
