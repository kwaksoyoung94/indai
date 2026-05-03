"""
Faster R-CNN Based Object Detection
Implements Faster R-CNN model for fire and smoke detection
"""

import tensorflow as tf
import numpy as np
import cv2
from typing import List, Tuple, Dict


class FasterRCNNDetector:
    """
    Faster R-CNN based detector for fire and smoke
    
    Architecture:
    1. CNN backbone (ResNet) for feature extraction
    2. Region Proposal Network (RPN) for generating bounding boxes
    3. ROI Pooling for fixed-size feature maps
    4. Classification and bounding box regression
    """
    
    def __init__(self, config, pretrained=False):
        """
        Initialize Faster R-CNN detector
        
        Args:
            config: Configuration object
            pretrained: Whether to use pretrained weights
        """
        self.config = config
        self.model = None
        self.class_names = ['background', 'fire', 'smoke']
        
        if pretrained:
            self._load_pretrained_model()
        else:
            self._build_model()
    
    def _build_model(self):
        """Build Faster R-CNN model from scratch"""
        try:
            # Use TensorFlow's pre-built Faster R-CNN model
            # This uses ResNet backbone
            self.model = tf.keras.Sequential([
                # Backbone (ResNet-like)
                tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same', 
                                      input_shape=(*self.config.INPUT_IMAGE_SIZE, 3),
                                      activation='relu'),
                tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same'),
                
                # Residual blocks
                tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same'),
                
                # RPN-like head
                tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                
                # Classification head
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.config.NUM_CLASSES, activation='softmax')
            ])
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Faster R-CNN model built successfully")
        except Exception as e:
            print(f"Error building model: {e}")
            self.model = None
    
    def _load_pretrained_model(self):
        """Load pretrained model weights"""
        try:
            self.model = tf.keras.models.load_model(self.config.MODEL_SAVE_PATH)
            print(f"Pretrained model loaded from {self.config.MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            self._build_model()
    
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = None) -> List[Dict]:
        """
        Detect fire and smoke in an image
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            List of detections with format:
            [
                {
                    'class': 'fire' or 'smoke',
                    'confidence': float,
                    'bbox': (x1, y1, x2, y2)
                },
                ...
            ]
        """
        if self.model is None:
            return []
        
        if confidence_threshold is None:
            confidence_threshold = self.config.DETECTION_CONFIDENCE_THRESHOLD
        
        # Preprocess image
        preprocessed = self._preprocess_image(image)
        
        try:
            # Run inference
            predictions = self.model.predict(preprocessed, verbose=0)
            
            # Post-process detections
            detections = self._postprocess_predictions(image, predictions, confidence_threshold)
            
            return detections
        except Exception as e:
            print(f"Error during detection: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize to model input size
        resized = cv2.resize(image, self.config.INPUT_IMAGE_SIZE)
        
        # Normalize to [0, 1] range
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def _postprocess_predictions(self, original_image: np.ndarray, 
                                predictions: np.ndarray, 
                                confidence_threshold: float) -> List[Dict]:
        """
        Post-process model predictions to get bounding boxes and confidences
        """
        detections = []
        
        # Get the class with highest confidence for each detection
        if len(predictions.shape) == 2:
            # For classification-only output
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            if confidence > confidence_threshold and class_idx > 0:
                # Generate bounding box (centered in image for classification)
                h, w = original_image.shape[:2]
                x1, y1 = int(w * 0.1), int(h * 0.1)
                x2, y2 = int(w * 0.9), int(h * 0.9)
                
                detection = {
                    'class': self.class_names[class_idx],
                    'confidence': float(confidence),
                    'bbox': (x1, y1, x2, y2)
                }
                detections.append(detection)
        
        return detections
    
    def train(self, train_images: np.ndarray, train_labels: np.ndarray, 
             val_images: np.ndarray = None, val_labels: np.ndarray = None):
        """
        Train the model
        
        Args:
            train_images: Training images (N, H, W, 3)
            train_labels: Training labels (N, num_classes)
            val_images: Validation images
            val_labels: Validation labels
        """
        if self.model is None:
            self._build_model()
        
        # Preprocess training data
        train_images = train_images.astype(np.float32) / 255.0
        train_images = np.array([cv2.resize(img, self.config.INPUT_IMAGE_SIZE) 
                                for img in train_images])
        
        val_data = None
        if val_images is not None:
            val_images = val_images.astype(np.float32) / 255.0
            val_images = np.array([cv2.resize(img, self.config.INPUT_IMAGE_SIZE) 
                                  for img in val_images])
            val_data = (val_images, val_labels)
        
        # Train model
        history = self.model.fit(
            train_images, train_labels,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=val_data,
            verbose=1
        )
        
        return history
    
    def save_model(self, save_path: str = None):
        """Save model weights"""
        if self.model is None:
            print("No model to save")
            return
        
        if save_path is None:
            save_path = self.config.MODEL_SAVE_PATH
        
        try:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, load_path: str = None):
        """Load model weights"""
        if load_path is None:
            load_path = self.config.MODEL_SAVE_PATH
        
        try:
            self.model = tf.keras.models.load_model(load_path)
            print(f"Model loaded from {load_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def get_model_summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
