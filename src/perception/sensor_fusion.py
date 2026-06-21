"""
Multi-sensor data fusion for drone perception.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SensorFusion:
    """Multi-sensor data fusion for enhanced perception."""
    
    def __init__(self, config: dict):
        """
        Initialize sensor fusion system.
        
        Args:
            config: Sensor fusion configuration
        """
        self.config = config
        self.sensor_weights = {
            'camera': 1.0,
            'depth': 0.8,
            'imu': 0.6,
            'gps': 0.5
        }
        self._sensor_data = {}
    
    def add_sensor_data(self, sensor_type: str, data: np.ndarray, 
                       timestamp: float) -> None:
        """
        Add sensor data to fusion system.
        
        Args:
            sensor_type: Type of sensor (camera, depth, imu, gps)
            data: Sensor data
            timestamp: Data timestamp
        """
        self._sensor_data[sensor_type] = {
            'data': data,
            'timestamp': timestamp,
            'weight': self.sensor_weights.get(sensor_type, 1.0)
        }
    
    def fuse_depth_and_rgb(self, rgb_image: np.ndarray, 
                          depth_image: np.ndarray) -> np.ndarray:
        """
        Fuse RGB and depth images for 3D perception.
        
        Args:
            rgb_image: RGB image
            depth_image: Depth image
            
        Returns:
            Fused RGB-D image (4 channels)
        """
        try:
            # Normalize depth to 0-255
            depth_normalized = cv2.normalize(
                depth_image, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            # Stack RGB and depth
            rgbd = np.dstack([rgb_image, depth_normalized])
            return rgbd
        except Exception as e:
            logger.error(f"Failed to fuse depth and RGB: {e}")
            return rgb_image
    
    def fuse_imu_and_gps(self, imu_data: Dict, gps_data: Dict) -> Dict:
        """
        Fuse IMU and GPS data for improved position estimation.
        
        Args:
            imu_data: IMU sensor data (acceleration, angular velocity)
            gps_data: GPS sensor data (position, velocity)
            
        Returns:
            Fused state estimate
        """
        try:
            # Simple weighted fusion (in production, use Kalman filter)
            fused_state = {
                'position': gps_data.get('position', (0, 0, 0)),
                'velocity': gps_data.get('velocity', (0, 0, 0)),
                'acceleration': imu_data.get('acceleration', (0, 0, 0)),
                'angular_velocity': imu_data.get('angular_velocity', (0, 0, 0)),
                'orientation': imu_data.get('orientation', (0, 0, 0))
            }
            return fused_state
        except Exception as e:
            logger.error(f"Failed to fuse IMU and GPS: {e}")
            return {}
    
    def fuse_obstacle_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Fuse obstacle detections from multiple sensors.
        
        Args:
            detections: List of obstacle detections from different sensors
            
        Returns:
            Fused obstacle detections
        """
        try:
            if not detections:
                return []
            
            # Group detections by spatial proximity
            fused_detections = []
            used_indices = set()
            
            for i, det1 in enumerate(detections):
                if i in used_indices:
                    continue
                
                # Find nearby detections
                nearby = [det1]
                used_indices.add(i)
                
                for j, det2 in enumerate(detections):
                    if j in used_indices:
                        continue
                    
                    # Calculate distance between detections
                    pos1 = det1.get('position', (0, 0, 0))
                    pos2 = det2.get('position', (0, 0, 0))
                    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    
                    if distance < 1.0:  # 1 meter threshold
                        nearby.append(det2)
                        used_indices.add(j)
                
                # Fuse nearby detections
                if len(nearby) > 1:
                    fused = self._fuse_detection_group(nearby)
                    fused_detections.append(fused)
                else:
                    fused_detections.append(det1)
            
            return fused_detections
        except Exception as e:
            logger.error(f"Failed to fuse obstacle detections: {e}")
            return detections
    
    def _fuse_detection_group(self, detections: List[Dict]) -> Dict:
        """
        Fuse a group of nearby detections.
        
        Args:
            detections: List of nearby detections
            
        Returns:
            Fused detection
        """
        try:
            # Average positions
            positions = [d.get('position', (0, 0, 0)) for d in detections]
            avg_position = np.mean(positions, axis=0)
            
            # Average confidence
            confidences = [d.get('confidence', 0.5) for d in detections]
            avg_confidence = np.mean(confidences)
            
            # Keep most common class
            classes = [d.get('class', 'unknown') for d in detections]
            most_common_class = max(set(classes), key=classes.count)
            
            fused = {
                'position': tuple(avg_position),
                'confidence': avg_confidence,
                'class': most_common_class,
                'sensor_count': len(detections)
            }
            
            return fused
        except Exception as e:
            logger.error(f"Failed to fuse detection group: {e}")
            return detections[0] if detections else {}
    
    def get_fused_state(self) -> Optional[Dict]:
        """
        Get current fused sensor state.
        
        Returns:
            Fused state dictionary or None if no data available
        """
        if not self._sensor_data:
            return None
        
        try:
            # Get latest data from each sensor
            latest_data = {}
            for sensor_type, data in self._sensor_data.items():
                latest_data[sensor_type] = data
            
            return latest_data
        except Exception as e:
            logger.error(f"Failed to get fused state: {e}")
            return None
    
    def clear_old_data(self, max_age: float = 1.0) -> None:
        """
        Clear sensor data older than max_age seconds.
        
        Args:
            max_age: Maximum age in seconds
        """
        import time
        current_time = time.time()
        
        for sensor_type in list(self._sensor_data.keys()):
            data = self._sensor_data[sensor_type]
            if current_time - data['timestamp'] > max_age:
                del self._sensor_data[sensor_type]
                logger.debug(f"Cleared old data from {sensor_type}")
