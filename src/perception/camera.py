"""
Camera interface for drone perception system.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class Camera:
    """Camera interface for capturing images and video from drone."""
    
    def __init__(self, drone_interface, config: dict):
        """
        Initialize the camera interface.
        
        Args:
            drone_interface: Drone interface object
            config: Camera configuration dictionary
        """
        self.drone = drone_interface
        self.config = config
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps = config.get('fps', 30)
        self.camera_type = config.get('type', 'front')
        self._is_running = False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a single frame from the camera.
        
        Returns:
            Image as numpy array or None if unavailable
        """
        try:
            frame = self.drone.get_image(self.camera_type)
            if frame is not None:
                # Resize to configured dimensions
                frame = cv2.resize(frame, (self.width, self.height))
            return frame
        except Exception as e:
            logger.error(f"Failed to get camera frame: {e}")
            return None
    
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """
        Get a depth frame from the camera.
        
        Returns:
            Depth image as numpy array or None if unavailable
        """
        try:
            depth = self.drone.get_depth_image(self.camera_type)
            if depth is not None:
                # Resize to configured dimensions
                depth = cv2.resize(depth, (self.width, self.height))
            return depth
        except Exception as e:
            logger.error(f"Failed to get depth frame: {e}")
            return None
    
    def get_stereo_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get stereo image pair.
        
        Returns:
            Tuple of (left_image, right_image) or (None, None) if unavailable
        """
        try:
            left = self.drone.get_image('stereo_left')
            right = self.drone.get_image('stereo_right')
            
            if left is not None and right is not None:
                left = cv2.resize(left, (self.width, self.height))
                right = cv2.resize(right, (self.width, self.height))
            
            return left, right
        except Exception as e:
            logger.error(f"Failed to get stereo pair: {e}")
            return None, None
    
    def start_video_capture(self, output_path: str, codec: str = 'mp4v') -> bool:
        """
        Start video capture to file.
        
        Args:
            output_path: Path to save video file
            codec: Video codec (default: mp4v)
            
        Returns:
            True if capture started successfully, False otherwise
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, (self.width, self.height)
            )
            self._is_running = True
            logger.info(f"Started video capture to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to start video capture: {e}")
            return False
    
    def stop_video_capture(self) -> bool:
        """
        Stop video capture.
        
        Returns:
            True if capture stopped successfully, False otherwise
        """
        try:
            if hasattr(self, 'video_writer') and self.video_writer.isOpened():
                self.video_writer.release()
                self._is_running = False
                logger.info("Stopped video capture")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop video capture: {e}")
            return False
    
    def capture_frame_to_video(self, frame: np.ndarray) -> bool:
        """
        Capture a frame to the video file.
        
        Args:
            frame: Image frame to capture
            
        Returns:
            True if frame captured successfully, False otherwise
        """
        if not self._is_running:
            return False
        
        try:
            self.video_writer.write(frame)
            return True
        except Exception as e:
            logger.error(f"Failed to capture frame to video: {e}")
            return False
    
    def get_camera_info(self) -> Dict:
        """
        Get camera information.
        
        Returns:
            Dictionary with camera information
        """
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'type': self.camera_type,
            'is_running': self._is_running
        }
    
    def calibrate_camera(self, calibration_images_path: str) -> Optional[Dict]:
        """
        Calibrate camera using chessboard images.
        
        Args:
            calibration_images_path: Path to calibration images
            
        Returns:
            Calibration dictionary with camera matrix and distortion coefficients
        """
        try:
            # Chessboard dimensions
            chessboard_size = (9, 6)
            
            # Prepare object points
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
            
            objpoints = []
            imgpoints = []
            
            # Load calibration images
            import glob
            images = glob.glob(f"{calibration_images_path}/*.jpg")
            
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
            
            # Calibrate camera
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            if ret:
                calibration = {
                    'camera_matrix': mtx,
                    'distortion_coefficients': dist,
                    'rotation_vectors': rvecs,
                    'translation_vectors': tvecs
                }
                logger.info("Camera calibration successful")
                return calibration
            else:
                logger.error("Camera calibration failed")
                return None
                
        except Exception as e:
            logger.error(f"Failed to calibrate camera: {e}")
            return None
    
    def undistort_image(self, image: np.ndarray, calibration: Dict) -> np.ndarray:
        """
        Undistort image using calibration data.
        
        Args:
            image: Input image
            calibration: Calibration dictionary
            
        Returns:
            Undistorted image
        """
        try:
            mtx = calibration['camera_matrix']
            dist = calibration['distortion_coefficients']
            undistorted = cv2.undistort(image, mtx, dist, None, mtx)
            return undistorted
        except Exception as e:
            logger.error(f"Failed to undistort image: {e}")
            return image
