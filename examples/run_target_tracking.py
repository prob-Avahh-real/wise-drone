"""
Example script for running target tracking algorithm.
"""
import sys
import os
import yaml
import logging
import cv2
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.target_tracking import TargetTrackingSystem
from utils.visualization import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main target tracking example."""
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
    
    # Get input source (camera or video file)
    input_source = 0  # Default camera
    
    # Check if video file provided as argument
    if len(sys.argv) > 1:
        input_source = sys.argv[1]
        logger.info(f"Using video file: {input_source}")
    else:
        logger.info("Using default camera (0)")
    
    # Open video capture
    cap = cv2.VideoCapture(input_source)
    
    if not cap.isOpened():
        logger.error("Failed to open video source")
        logger.info("Please provide a video file as argument: python run_target_tracking.py <video_path>")
        logger.info("Or ensure camera is connected and accessible")
        return
    
    logger.info("Starting target tracking. Press 'q' to quit.")
    
    try:
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Track objects
            tracks, detections = tracker.track(frame)
            
            # Visualize results
            vis_frame = tracker.visualize(frame, tracks, detections)
            
            # Display frame
            cv2.imshow("Target Tracking", vis_frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_start_time)
                logger.info(f"FPS: {fps:.2f}, Tracks: {len(tracks)}, Detections: {len(detections)}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Target tracking stopped")


def process_video_file(video_path: str, output_path: str):
    """
    Process video file with tracking and save results.
    
    Args:
        video_path: Input video path
        output_path: Output video path
    """
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize target tracking system
    logger.info("Initializing target tracking system...")
    tracker = TargetTrackingSystem(config['target_tracking'])
    
    # Process video
    logger.info(f"Processing video: {video_path}")
    tracker.process_video(video_path, output_path)
    logger.info(f"Video processing complete. Output saved to: {output_path}")


if __name__ == "__main__":
    # Check if processing video file
    if len(sys.argv) > 2 and sys.argv[1] == '--video':
        video_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else 'output_tracking.mp4'
        process_video_file(video_path, output_path)
    else:
        main()
