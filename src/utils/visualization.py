"""
Visualization utilities for drone AI algorithms.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization tools for drone perception and control."""
    
    def __init__(self, config: dict):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.window_size = config.get('window_size', [1280, 720])
        self.show_tracking = config.get('show_tracking', True)
        self.show_obstacles = config.get('show_obstacles', True)
        self.show_path = config.get('show_path', True)
    
    def draw_tracking_results(self, frame: np.ndarray, tracks: List[Dict],
                             detections: List[Dict]) -> np.ndarray:
        """
        Draw target tracking results on frame.
        
        Args:
            frame: Input image frame
            tracks: List of track dictionaries
            detections: List of detection dictionaries
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw detections
        for detection in detections:
            bbox = detection.get('bbox', (0, 0, 0, 0))
            confidence = detection.get('confidence', 0.0)
            class_name = detection.get('class_name', 'unknown')
            
            x, y, w, h = bbox
            cv2.rectangle(vis_frame, (int(x), int(y)), (int(x + w), int(y + h)),
                         (0, 255, 0), 2)
            
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (int(x), int(y - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tracks
        for track in tracks:
            bbox = track.get('bbox', (0, 0, 0, 0))
            track_id = track.get('track_id', 0)
            class_name = track.get('class_name', 'unknown')
            hits = track.get('hits', 0)
            
            x, y, w, h = bbox
            color = (0, 0, 255) if hits >= 3 else (255, 0, 0)
            cv2.rectangle(vis_frame, (int(x), int(y)), (int(x + w), int(y + h)),
                         color, 2)
            
            label = f"ID {track_id}: {class_name}"
            cv2.putText(vis_frame, label, (int(x), int(y - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    def draw_obstacles(self, frame: np.ndarray, obstacles: List[Dict]) -> np.ndarray:
        """
        Draw obstacle detections on frame.
        
        Args:
            frame: Input image frame
            obstacles: List of obstacle dictionaries
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        for obstacle in obstacles:
            bbox = obstacle.get('bbox', (0, 0, 0, 0))
            depth = obstacle.get('depth', 0.0)
            confidence = obstacle.get('confidence', 0.0)
            
            x, y, w, h = bbox
            
            # Color based on depth (closer = redder)
            if depth > 0:
                color_intensity = int(255 * (1 - min(depth / 5.0, 1.0)))
                color = (color_intensity, 0, 255 - color_intensity)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"Obs: {confidence:.2f}"
            if depth > 0:
                label += f" d={depth:.1f}m"
            cv2.putText(vis_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    def draw_path(self, frame: np.ndarray, path: List[Tuple[float, float, float]],
                  current_position: Tuple[float, float, float]) -> np.ndarray:
        """
        Draw navigation path on frame.
        
        Args:
            frame: Input image frame
            path: List of waypoint positions
            current_position: Current drone position
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        if not path:
            return vis_frame
        
        # Draw path lines
        for i in range(len(path) - 1):
            pt1 = self._world_to_image(path[i], frame.shape)
            pt2 = self._world_to_image(path[i + 1], frame.shape)
            cv2.line(vis_frame, pt1, pt2, (0, 255, 255), 2)
        
        # Draw waypoints
        for i, waypoint in enumerate(path):
            pt = self._world_to_image(waypoint, frame.shape)
            color = (0, 255, 0) if i == 0 else (255, 0, 0) if i == len(path) - 1 else (0, 0, 255)
            cv2.circle(vis_frame, pt, 5, color, -1)
            cv2.putText(vis_frame, f"{i}", (pt[0] + 10, pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw current position
        current_pt = self._world_to_image(current_position, frame.shape)
        cv2.circle(vis_frame, current_pt, 8, (0, 255, 0), -1)
        
        return vis_frame
    
    def _world_to_image(self, position: Tuple[float, float, float],
                       image_shape: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        Convert world position to image coordinates.
        
        Args:
            position: World position (x, y, z)
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Image coordinates (x, y)
        """
        # Simple projection (in production, use camera intrinsics)
        height, width = image_shape[:2]
        x = int(position[0] * width / 100.0)
        y = int(position[1] * height / 100.0)
        return (x, y)
    
    def draw_drone_state(self, frame: np.ndarray, state: Dict) -> np.ndarray:
        """
        Draw drone state information on frame.
        
        Args:
            frame: Input image frame
            state: Drone state dictionary
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw state information
        y_offset = 30
        line_height = 25
        
        info_lines = [
            f"Position: {state.get('position', (0, 0, 0))}",
            f"Altitude: {state.get('position', (0, 0, 0))[2]:.1f}m",
            f"Velocity: {state.get('velocity', (0, 0, 0))}",
            f"Battery: {state.get('battery_level', 0):.1f}%",
            f"State: {state.get('flight_state', 'unknown')}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(vis_frame, line, (10, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def create_dashboard(self, frame: np.ndarray, tracks: List[Dict],
                        obstacles: List[Dict], path: List[Tuple[float, float, float]],
                        drone_state: Dict) -> np.ndarray:
        """
        Create comprehensive dashboard with all visualizations.
        
        Args:
            frame: Input image frame
            tracks: List of track dictionaries
            obstacles: List of obstacle dictionaries
            path: Navigation path
            drone_state: Drone state dictionary
            
        Returns:
            Dashboard image
        """
        vis_frame = frame.copy()
        
        if self.show_tracking:
            vis_frame = self.draw_tracking_results(vis_frame, tracks, [])
        
        if self.show_obstacles:
            vis_frame = self.draw_obstacles(vis_frame, obstacles)
        
        if self.show_path and path:
            current_pos = drone_state.get('position', (0, 0, 0))
            vis_frame = self.draw_path(vis_frame, path, current_pos)
        
        vis_frame = self.draw_drone_state(vis_frame, drone_state)
        
        return vis_frame
    
    def display_frame(self, frame: np.ndarray, window_name: str = "Drone AI Dashboard") -> None:
        """
        Display frame in window.
        
        Args:
            frame: Image frame to display
            window_name: Window name
        """
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    
    def save_frame(self, frame: np.ndarray, output_path: str) -> bool:
        """
        Save frame to file.
        
        Args:
            frame: Image frame to save
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cv2.imwrite(output_path, frame)
            return True
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False
