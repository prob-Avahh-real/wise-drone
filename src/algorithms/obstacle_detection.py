"""
Obstacle detection algorithms for drone navigation.
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ObstacleDetector:
    """Obstacle detection using depth and semantic segmentation."""
    
    def __init__(self, config: dict):
        """
        Initialize obstacle detector.
        
        Args:
            config: Obstacle detection configuration
        """
        self.config = config
        self.depth_threshold = config.get('depth_threshold', 5.0)
        self.safety_margin = config.get('safety_margin', 2.0)
        self.confidence = config.get('confidence', 0.5)
    
    def detect_from_depth(self, depth_image: np.ndarray) -> List[Dict]:
        """
        Detect obstacles from depth image.
        
        Args:
            depth_image: Depth image in meters
            
        Returns:
            List of obstacle detections
        """
        try:
            obstacles = []
            
            # Threshold depth image
            thresholded = np.where(depth_image < self.depth_threshold, 255, 0).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate average depth in region
                roi = depth_image[y:y+h, x:x+w]
                avg_depth = np.mean(roi[roi > 0]) if np.any(roi > 0) else self.depth_threshold
                
                # Create obstacle detection
                obstacle = {
                    'bbox': (x, y, w, h),
                    'depth': avg_depth,
                    'confidence': 1.0 - (avg_depth / self.depth_threshold),
                    'type': 'depth_based'
                }
                obstacles.append(obstacle)
            
            return obstacles
        except Exception as e:
            logger.error(f"Depth-based detection failed: {e}")
            return []
    
    def detect_from_rgb(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        Detect obstacles from RGB image using edge detection.
        
        Args:
            rgb_image: RGB image
            
        Returns:
            List of obstacle detections
        """
        try:
            obstacles = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create obstacle detection
                obstacle = {
                    'bbox': (x, y, w, h),
                    'depth': 0.0,  # Unknown depth
                    'confidence': 0.5,
                    'type': 'edge_based'
                }
                obstacles.append(obstacle)
            
            return obstacles
        except Exception as e:
            logger.error(f"RGB-based detection failed: {e}")
            return []
    
    def detect_from_rgbd(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> List[Dict]:
        """
        Detect obstacles from RGB-D image.
        
        Args:
            rgb_image: RGB image
            depth_image: Depth image
            
        Returns:
            List of obstacle detections
        """
        # Get depth-based detections
        depth_obstacles = self.detect_from_depth(depth_image)
        
        # Get RGB-based detections
        rgb_obstacles = self.detect_from_rgb(rgb_image)
        
        # Merge detections
        merged_obstacles = self._merge_detections(depth_obstacles, rgb_obstacles)
        
        return merged_obstacles
    
    def _merge_detections(self, depth_obstacles: List[Dict], 
                         rgb_obstacles: List[Dict]) -> List[Dict]:
        """
        Merge depth and RGB detections.
        
        Args:
            depth_obstacles: Depth-based detections
            rgb_obstacles: RGB-based detections
            
        Returns:
            Merged detections
        """
        merged = []
        
        # Add depth obstacles (higher confidence)
        for obs in depth_obstacles:
            merged.append(obs)
        
        # Add RGB obstacles that don't overlap with depth obstacles
        for rgb_obs in rgb_obstacles:
            overlaps = False
            for depth_obs in depth_obstacles:
                if self._calculate_overlap(rgb_obs['bbox'], depth_obs['bbox']) > 0.5:
                    overlaps = True
                    break
            
            if not overlaps:
                merged.append(rgb_obs)
        
        return merged
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int],
                          bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            Overlap ratio
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def get_obstacle_positions(self, obstacles: List[Dict], 
                               camera_intrinsics: Dict) -> List[Tuple[float, float, float]]:
        """
        Convert obstacle detections to 3D positions.
        
        Args:
            obstacles: List of obstacle detections
            camera_intrinsics: Camera intrinsic parameters
            
        Returns:
            List of 3D positions (x, y, z)
        """
        positions = []
        
        for obs in obstacles:
            bbox = obs['bbox']
            depth = obs.get('depth', 0.0)
            
            if depth <= 0:
                continue
            
            # Get center of bounding box
            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2
            
            # Convert to 3D using camera intrinsics
            fx = camera_intrinsics.get('fx', 500.0)
            fy = camera_intrinsics.get('fy', 500.0)
            cx0 = camera_intrinsics.get('cx', 320.0)
            cy0 = camera_intrinsics.get('cy', 240.0)
            
            # 3D coordinates
            X = (cx - cx0) * depth / fx
            Y = (cy - cy0) * depth / fy
            Z = depth
            
            positions.append((X, Y, Z))
        
        return positions
    
    def visualize(self, image: np.ndarray, obstacles: List[Dict]) -> np.ndarray:
        """
        Visualize obstacle detections on image.
        
        Args:
            image: Input image
            obstacles: List of obstacle detections
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        for obs in obstacles:
            bbox = obs['bbox']
            x, y, w, h = bbox
            confidence = obs.get('confidence', 0.5)
            depth = obs.get('depth', 0.0)
            
            # Color based on depth (closer = redder)
            if depth > 0:
                color_intensity = int(255 * (1 - min(depth / self.depth_threshold, 1.0)))
                color = (color_intensity, 0, 255 - color_intensity)
            else:
                color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"Obs: {confidence:.2f}"
            if depth > 0:
                label += f" d={depth:.1f}m"
            cv2.putText(vis_image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image


class SemanticSegmentation:
    """Semantic segmentation for obstacle classification."""
    
    def __init__(self, config: dict):
        """
        Initialize semantic segmentation.
        
        Args:
            config: Segmentation configuration
        """
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load semantic segmentation model."""
        try:
            # Placeholder for model loading
            # In production, load models like DeepLabV3, SegFormer, etc.
            logger.info("Semantic segmentation model not implemented (placeholder)")
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform semantic segmentation on image.
        
        Args:
            image: Input image
            
        Returns:
            Segmentation mask
        """
        # Placeholder implementation
        # In production, use actual segmentation model
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def get_obstacle_mask(self, segmentation: np.ndarray, 
                         obstacle_classes: List[int]) -> np.ndarray:
        """
        Get mask for obstacle classes.
        
        Args:
            segmentation: Segmentation mask
            obstacle_classes: List of obstacle class IDs
            
        Returns:
            Obstacle mask
        """
        mask = np.zeros_like(segmentation)
        for class_id in obstacle_classes:
            mask[segmentation == class_id] = 255
        return mask
