"""
Data loading utilities for training and testing.
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import logging
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for training and testing AI models."""
    
    def __init__(self, config: dict):
        """
        Initialize data loader.
        
        Args:
            config: Data loader configuration
        """
        self.config = config
        self.batch_size = config.get('batch_size', 16)
        self.shuffle = config.get('shuffle', True)
    
    def load_images_from_directory(self, directory: str,
                                  extensions: List[str] = ['.jpg', '.png', '.jpeg']) -> List[str]:
        """
        Load image file paths from directory.
        
        Args:
            directory: Directory path
            extensions: List of valid file extensions
            
        Returns:
            List of image file paths
        """
        image_paths = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))
        
        logger.info(f"Loaded {len(image_paths)} images from {directory}")
        return image_paths
    
    def load_image(self, path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Load single image from file.
        
        Args:
            path: Image file path
            target_size: Target size (width, height) or None for original size
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            image = cv2.imread(path)
            if image is None:
                logger.error(f"Failed to load image: {path}")
                return None
            
            if target_size:
                image = cv2.resize(image, target_size)
            
            return image
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None
    
    def load_video_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Load frames from video file.
        
        Args:
            video_path: Video file path
            max_frames: Maximum number of frames to load or None for all frames
            
        Returns:
            List of frames
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
            
            cap.release()
            logger.info(f"Loaded {len(frames)} frames from {video_path}")
            return frames
        except Exception as e:
            logger.error(f"Failed to load video {video_path}: {e}")
            return []
    
    def load_annotations(self, annotation_path: str) -> Optional[Dict]:
        """
        Load annotations from file.
        
        Args:
            annotation_path: Annotation file path
            
        Returns:
            Annotation dictionary or None if failed
        """
        try:
            import json
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            return annotations
        except Exception as e:
            logger.error(f"Failed to load annotations {annotation_path}: {e}")
            return None
    
    def create_batches(self, data: List, batch_size: Optional[int] = None) -> List[List]:
        """
        Create batches from data.
        
        Args:
            data: List of data items
            batch_size: Batch size or None to use default
            
        Returns:
            List of batches
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def split_data(self, data: List, train_ratio: float = 0.7,
                  val_ratio: float = 0.2) -> Tuple[List, List, List]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: List of data items
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.shuffle:
            import random
            random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return train_data, val_data, test_data


class DatasetGenerator:
    """Dataset generator for training models."""
    
    def __init__(self, image_paths: List[str], annotations: Dict,
                 batch_size: int = 16, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize dataset generator.
        
        Args:
            image_paths: List of image file paths
            annotations: Annotations dictionary
            batch_size: Batch size
            target_size: Target image size
        """
        self.image_paths = image_paths
        self.annotations = annotations
        self.batch_size = batch_size
        self.target_size = target_size
        self.index = 0
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.image_paths) // self.batch_size
    
    def __iter__(self):
        """Reset index and return self."""
        self.index = 0
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get next batch.
        
        Returns:
            Tuple of (images, labels)
        """
        if self.index >= len(self.image_paths):
            raise StopIteration
        
        batch_paths = self.image_paths[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        
        images = []
        labels = []
        
        for path in batch_paths:
            # Load image
            image = cv2.imread(path)
            if image is not None:
                image = cv2.resize(image, self.target_size)
                images.append(image)
                
                # Get annotation
                filename = os.path.basename(path)
                label = self.annotations.get(filename, [])
                labels.append(label)
        
        return np.array(images), np.array(labels)
