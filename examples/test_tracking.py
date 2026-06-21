"""
Test script for target tracking with synthetic video.
"""
import sys
import os
import yaml
import logging
import cv2
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.target_tracking import TargetTrackingSystem
from utils.visualization import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_frame(frame_num, width=640, height=480):
    """
    Generate a synthetic test frame with a moving object.
    
    Args:
        frame_num: Current frame number
        width: Frame width
        height: Frame height
        
    Returns:
        Synthetic frame as numpy array
    """
    # Create black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some background noise
    frame = cv2.rectangle(frame, (0, 0), (width, height), (50, 50, 50), -1)
    
    # Create moving object (simulating a person)
    # Move in a circular pattern
    center_x = width // 2
    center_y = height // 2
    radius = 100
    angle = frame_num * 0.05
    
    obj_x = int(center_x + radius * np.cos(angle))
    obj_y = int(center_y + radius * np.sin(angle))
    
    # Draw object (person-like shape)
    obj_width = 40
    obj_height = 80
    
    # Body
    cv2.rectangle(frame, (obj_x - obj_width//2, obj_y - obj_height//2),
                 (obj_x + obj_width//2, obj_y + obj_height//2), (0, 255, 0), -1)
    
    # Head
    cv2.circle(frame, (obj_x, obj_y - obj_height//2 - 15), 15, (0, 255, 0), -1)
    
    # Add frame number
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def main():
    """Main test function."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize target tracking system
    logger.info("Initializing target tracking system...")
    tracker = TargetTrackingSystem(config['target_tracking'])
    
    # Initialize visualizer
    logger.info("Initializing visualizer...")
    visualizer = Visualizer(config['visualization'])
    
    logger.info("Starting test with synthetic video. Press 'q' to quit.")
    
    try:
        frame_count = 0
        fps_start_time = time.time()
        max_frames = 300  # Run for 300 frames
        
        while frame_count < max_frames:
            # Generate synthetic frame
            frame = generate_test_frame(frame_count)
            
            # Track objects
            tracks, detections = tracker.track(frame)
            
            # Visualize results
            vis_frame = tracker.visualize(frame, tracks, detections)
            
            # Display frame
            cv2.imshow("Target Tracking Test", vis_frame)
            
            # Calculate FPS
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_start_time)
                logger.info(f"FPS: {fps:.2f}, Tracks: {len(tracks)}, Detections: {len(detections)}")
            
            # Check for quit
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        logger.info(f"Test complete. Processed {frame_count} frames.")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        cv2.destroyAllWindows()
        logger.info("Test stopped")


if __name__ == "__main__":
    main()
