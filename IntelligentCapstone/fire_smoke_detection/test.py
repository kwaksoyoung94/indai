"""
Unit Tests for Fire and Smoke Detection System
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
import cv2
from config import Config
from fire_smoke_detector import FireSmokeDetector
from feature_extraction import FeatureExtractor, GlobalFeatureValidator


class TestConfig(unittest.TestCase):
    """Test configuration"""
    
    def test_config_initialization(self):
        """Test config can be initialized"""
        config = Config()
        self.assertIsNotNone(config)
        self.assertEqual(config.NUM_CLASSES, 3)
    
    def test_thresholds_exist(self):
        """Test all thresholds are defined"""
        config = Config()
        self.assertGreater(config.SSIM_THRESHOLD, 0)
        self.assertGreater(config.MSE_THRESHOLD, 0)
        self.assertGreater(config.FIRE_MSE_THRESHOLD, 0)
        self.assertGreater(config.SMOKE_MSE_THRESHOLD, 0)


class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction methods"""
    
    def setUp(self):
        self.config = Config()
        self.extractor = FeatureExtractor(self.config)
        self.frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_ssim_calculation(self):
        """Test SSIM calculation"""
        ssim_val = self.extractor.calculate_ssim(self.frame1, self.frame2)
        self.assertIsInstance(ssim_val, float)
        self.assertGreaterEqual(ssim_val, -1.0)
        self.assertLessEqual(ssim_val, 1.0)
    
    def test_ssim_identical_frames(self):
        """Test SSIM with identical frames"""
        ssim_val = self.extractor.calculate_ssim(self.frame1, self.frame1)
        self.assertGreater(ssim_val, 0.99)  # Should be very similar
    
    def test_mse_calculation(self):
        """Test MSE calculation"""
        mse_val = self.extractor.calculate_mse(self.frame1, self.frame2)
        self.assertIsInstance(mse_val, float)
        self.assertGreaterEqual(mse_val, 0)
    
    def test_mse_identical_frames(self):
        """Test MSE with identical frames"""
        mse_val = self.extractor.calculate_mse(self.frame1, self.frame1)
        self.assertEqual(mse_val, 0.0)
    
    def test_frame_difference(self):
        """Test frame difference calculation"""
        diff_val = self.extractor.calculate_frame_difference(self.frame1, self.frame2)
        self.assertIsInstance(diff_val, float)
        self.assertGreaterEqual(diff_val, 0)
        self.assertLessEqual(diff_val, 1)
    
    def test_color_histogram_smoke(self):
        """Test smoke color histogram"""
        hist_val = self.extractor.calculate_color_histogram(self.frame1, color_type='smoke')
        self.assertIsInstance(hist_val, float)
        self.assertGreaterEqual(hist_val, 0)
    
    def test_color_histogram_fire(self):
        """Test fire color histogram"""
        hist_val = self.extractor.calculate_color_histogram(self.frame1, color_type='fire')
        self.assertIsInstance(hist_val, float)
        self.assertGreaterEqual(hist_val, 0)
    
    def test_coefficient_of_variation(self):
        """Test coefficient of variation"""
        cv_val = self.extractor.calculate_coefficient_of_variation(self.frame1)
        self.assertIsInstance(cv_val, float)
        self.assertGreaterEqual(cv_val, 0)
    
    def test_wavelet_energy(self):
        """Test wavelet energy calculation"""
        wavelet_val = self.extractor.calculate_wavelet_energy(self.frame1)
        self.assertIsInstance(wavelet_val, float)
        self.assertGreaterEqual(wavelet_val, 0)
    
    def test_frame_buffer(self):
        """Test frame buffer management"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.extractor.add_frame_to_buffer(frame)
        buffer = self.extractor.get_frame_buffer()
        self.assertEqual(len(buffer), 1)


class TestGlobalFeatureValidator(unittest.TestCase):
    """Test global feature validation"""
    
    def setUp(self):
        self.config = Config()
        self.validator = GlobalFeatureValidator(self.config)
        self.frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_validate_global_features_identical(self):
        """Test global validation with identical frames (no motion)"""
        is_motion = self.validator.validate_global_features(self.frame1, self.frame1)
        self.assertFalse(is_motion)
    
    def test_validate_global_features_different(self):
        """Test global validation with different frames (has motion)"""
        is_motion = self.validator.validate_global_features(self.frame1, self.frame2)
        # Could be True or False depending on random frames
        self.assertIsInstance(is_motion, (bool, np.bool_))
    
    def test_validate_global_features_no_previous(self):
        """Test global validation with no previous frame"""
        is_motion = self.validator.validate_global_features(self.frame1, None)
        self.assertFalse(is_motion)


class TestFireSmokeDetector(unittest.TestCase):
    """Test main detector"""
    
    def setUp(self):
        self.config = Config()
        self.detector = FireSmokeDetector(self.config)
        self.frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.detector)
        self.assertIsNotNone(self.detector.feature_extractor)
    
    def test_process_frame(self):
        """Test frame processing"""
        annotated_frame, detections = self.detector.process_frame(self.frame)
        
        self.assertIsNotNone(annotated_frame)
        self.assertIsInstance(detections, list)
        self.assertEqual(annotated_frame.shape, self.frame.shape)
    
    def test_detection_format(self):
        """Test detection format"""
        _, detections = self.detector.process_frame(self.frame)
        
        for det in detections:
            self.assertIn('class', det)
            self.assertIn('confidence', det)
            self.assertIn('bbox', det)
            self.assertIn('timestamp', det)
            
            self.assertIn(det['class'], ['fire', 'smoke'])
            self.assertIsInstance(det['confidence'], float)
            self.assertIsInstance(det['bbox'], tuple)
    
    def test_batch_processing(self):
        """Test processing multiple frames"""
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                 for _ in range(5)]
        
        all_detections = []
        for frame in frames:
            _, detections = self.detector.process_frame(frame)
            all_detections.append(detections)
        
        self.assertEqual(len(all_detections), 5)
    
    def test_statistics(self):
        """Test statistics calculation"""
        for _ in range(3):
            self.detector.process_frame(self.frame)
        
        stats = self.detector.get_statistics()
        
        self.assertIn('total_frames_processed', stats)
        self.assertIn('fire_detections', stats)
        self.assertIn('smoke_detections', stats)
        self.assertGreater(stats['total_frames_processed'], 0)
    
    def test_clear_history(self):
        """Test clearing detection history"""
        self.detector.process_frame(self.frame)
        self.detector.clear_history()
        
        stats = self.detector.get_statistics()
        self.assertEqual(stats, {})


class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def test_frame_processing_speed(self):
        """Test that frame processing is reasonably fast"""
        import time
        
        config = Config()
        detector = FireSmokeDetector(config)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        detector.process_frame(frame)
        elapsed_time = time.time() - start_time
        
        # Processing should be reasonably fast (< 5 seconds for this test)
        self.assertLess(elapsed_time, 5.0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestGlobalFeatureValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestFireSmokeDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
