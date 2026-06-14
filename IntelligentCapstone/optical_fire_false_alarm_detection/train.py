"""
Training Script
Train the Faster R-CNN model for fire and smoke detection
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import cv2

from config import Config
from faster_rcnn_detector import FasterRCNNDetector


class DataLoader:
    """Load and prepare training data"""
    
    def __init__(self, config):
        self.config = config
    
    def load_dataset(self, data_dir, split=0.7):
        """
        Load dataset from directory structure:
        data_dir/
            fire/
                image1.jpg
                image2.jpg
                ...
            smoke/
                image1.jpg
                ...
            non_fire/
                image1.jpg
                ...
        
        Returns:
            Tuple of (images, labels, val_images, val_labels)
        """
        images = []
        labels = []
        
        class_names = {
            'non_fire': 0,
            'fire': 1,
            'smoke': 2
        }
        
        print(f"Loading dataset from {data_dir}...")
        
        for class_name, class_id in class_names.items():
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found")
                continue
            
            # Load images from class directory
            for filename in os.listdir(class_dir):
                if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                filepath = os.path.join(class_dir, filename)
                try:
                    # Read image
                    img = cv2.imread(filepath)
                    if img is None:
                        continue
                    
                    # Resize to model input size
                    img = cv2.resize(img, self.config.INPUT_IMAGE_SIZE)
                    
                    # Create one-hot label
                    label = np.zeros(self.config.NUM_CLASSES)
                    label[class_id] = 1
                    
                    images.append(img)
                    labels.append(label)
                
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        if len(images) == 0:
            print("No images loaded!")
            return None, None, None, None
        
        print(f"Loaded {len(images)} images")
        
        # Split into train/validation
        num_train = int(len(images) * split)
        
        # Shuffle
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        
        train_images = images[:num_train]
        train_labels = labels[:num_train]
        
        val_images = images[num_train:]
        val_labels = labels[num_train:]
        
        print(f"Train: {len(train_images)}, Validation: {len(val_images)}")
        
        return train_images, train_labels, val_images, val_labels


def main():
    parser = argparse.ArgumentParser(
        description='Fire and Smoke Detection - Training'
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to training data directory')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    parser.add_argument('--output-model', type=str, default=None,
                       help='Output model save path')
    
    parser.add_argument('--split', type=float, default=0.7,
                       help='Train/validation split ratio')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    
    if args.output_model:
        config.MODEL_SAVE_PATH = args.output_model
    
    # Load dataset
    data_loader = DataLoader(config)
    train_images, train_labels, val_images, val_labels = data_loader.load_dataset(
        args.data_dir, 
        split=args.split
    )
    
    if train_images is None:
        print("Failed to load dataset")
        return
    
    # Initialize model
    print("\nInitializing model...")
    detector = FasterRCNNDetector(config, pretrained=False)
    
    # Train model
    print("\nStarting training...")
    history = detector.train(
        train_images, train_labels,
        val_images, val_labels
    )
    
    # Save model
    print("\nSaving model...")
    detector.save_model(config.MODEL_SAVE_PATH)
    
    print(f"Training completed. Model saved to {config.MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
