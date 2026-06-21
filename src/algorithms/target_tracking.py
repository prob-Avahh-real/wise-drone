"""
Target tracking algorithm using YOLO + Kalman Filter + Hungarian Algorithm.
Based on the algorithm described in the technical document.
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single object detection."""
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    """Represents a single object track."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    class_id: int
    class_name: str
    age: int  # Number of frames since last detection
    hits: int  # Number of successful detections
    time_since_update: int
    state: np.ndarray  # Kalman filter state


class KalmanFilter:
    """Kalman filter for object tracking."""
    
    def __init__(self, initial_bbox: Tuple[float, float, float, float]):
        """
        Initialize Kalman filter with initial bounding box.
        
        Args:
            initial_bbox: Initial bounding box (x, y, w, h)
        """
        # State vector: [x, y, w, h, vx, vy, vw, vh]
        self.state = np.zeros((8, 1))
        self.state[0:4] = np.array(initial_bbox).reshape(4, 1)
        
        # State transition matrix
        self.F = np.eye(8)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh
        
        # Measurement matrix (we only measure x, y, w, h)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        
        # Process noise covariance
        self.Q = np.eye(8) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(4) * 0.1
        
        # Error covariance matrix
        self.P = np.eye(8) * 1.0
    
    def predict(self) -> np.ndarray:
        """
        Predict next state.
        
        Returns:
            Predicted state
        """
        # State prediction
        self.state = self.F @ self.state
        
        # Error covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with measurement.
        
        Args:
            measurement: Measurement vector [x, y, w, h]
            
        Returns:
            Updated state
        """
        # Measurement residual
        y = measurement - self.H @ self.state
        
        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.state = self.state + K @ y
        
        # Error covariance update
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state
    
    def get_bbox(self) -> Tuple[float, float, float, float]:
        """
        Get current bounding box from state.
        
        Returns:
            Bounding box (x, y, w, h)
        """
        return tuple(self.state[0:4].flatten())


class ObjectTracker:
    """Multi-object tracker using Kalman filter and Hungarian algorithm."""
    
    def __init__(self, config: dict):
        """
        Initialize object tracker.
        
        Args:
            config: Tracker configuration
        """
        self.config = config
        self.tracks: List[Track] = []
        self.next_track_id = 1
        
        # Tracking parameters
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        
        # Kalman filter parameters
        self.process_noise = config.get('process_noise', 0.1)
        self.measurement_noise = config.get('measurement_noise', 0.1)
        self.initial_error = config.get('initial_error', 1.0)
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of new detections
            
        Returns:
            List of confirmed tracks
        """
        # Predict all tracks
        for track in self.tracks:
            track.state = track.kf.predict()
            track.time_since_update += 1
        
        # Associate detections with tracks
        matched_indices, unmatched_detections, unmatched_tracks = \
            self._associate_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            self._update_track(self.tracks[track_idx], detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._initiate_track(detections[det_idx])
        
        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_hits]
    
    def _associate_detections_to_tracks(self, detections: List[Detection]) -> \
            Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using Hungarian algorithm.
        
        Args:
            detections: List of detections
            
        Returns:
            Tuple of (matched_indices, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for track_idx, track in enumerate(self.tracks):
            for det_idx, detection in enumerate(detections):
                iou_matrix[track_idx, det_idx] = self._calculate_iou(
                    track.bbox, detection.bbox
                )
        
        # Hungarian algorithm for optimal assignment
        track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        
        # Filter by IOU threshold
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                matched_indices.append((track_idx, det_idx))
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float],
                       bbox2: Tuple[float, float, float, float]) -> float:
        """
        Calculate Intersection over Union (IOU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            IOU value
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
        
        # Calculate union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _update_track(self, track: Track, detection: Detection) -> None:
        """
        Update track with detection.
        
        Args:
            track: Track to update
            detection: Detection to update with
        """
        # Update Kalman filter
        measurement = np.array(detection.bbox).reshape(4, 1)
        track.state = track.kf.update(measurement)
        
        # Update track properties
        track.bbox = track.kf.get_bbox()
        track.confidence = detection.confidence
        track.time_since_update = 0
        track.hits += 1
    
    def _initiate_track(self, detection: Detection) -> None:
        """
        Initialize new track from detection.
        
        Args:
            detection: Detection to initialize track from
        """
        kf = KalmanFilter(detection.bbox)
        kf.Q *= self.process_noise
        kf.R *= self.measurement_noise
        kf.P *= self.initial_error
        
        track = Track(
            track_id=self.next_track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            age=0,
            hits=1,
            time_since_update=0,
            state=kf.state
        )
        track.kf = kf
        
        self.tracks.append(track)
        self.next_track_id += 1


class TargetTrackingSystem:
    """Complete target tracking system with YOLO detection and tracking."""
    
    def __init__(self, config: dict):
        """
        Initialize target tracking system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.tracking_config = config.get('tracking', {})
        
        # Initialize YOLO model
        self.model = self._load_model()
        
        # Initialize tracker
        self.tracker = ObjectTracker(self.tracking_config)
        
        # Class names for COCO dataset
        self.class_names = self._get_class_names()
    
    def _load_model(self):
        """
        Load YOLO model for object detection.
        
        Returns:
            Loaded YOLO model
        """
        try:
            from ultralytics import YOLO
            
            model_path = self.model_config.get('path', '')
            model_size = self.model_config.get('size', 's')
            
            if model_path:
                logger.info(f"Loading custom model from {model_path}")
                model = YOLO(model_path)
            else:
                logger.info(f"Loading pretrained YOLOv8{model_size}")
                model = YOLO(f'yolov8{model_size}.pt')
            
            # Set confidence threshold
            model.conf = self.model_config.get('confidence', 0.5)
            model.iou = self.model_config.get('iou_threshold', 0.45)
            
            return model
        except ImportError:
            logger.error("Ultralytics YOLO not installed. Install with: pip install ultralytics")
            return None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return None
    
    def _get_class_names(self) -> List[str]:
        """
        Get COCO class names.
        
        Returns:
            List of class names
        """
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in frame using YOLO.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detections
        """
        if self.model is None:
            logger.warning("YOLO model not loaded")
            return []
        
        try:
            # Run inference
            results = self.model(frame)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                    
                    # Get confidence
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Get class
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                    
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    )
                    detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def track(self, frame: np.ndarray) -> Tuple[List[Track], List[Detection]]:
        """
        Track objects in frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (tracks, detections)
        """
        # Detect objects
        detections = self.detect(frame)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        return tracks, detections
    
    def visualize(self, frame: np.ndarray, tracks: List[Track],
                  detections: List[Detection]) -> np.ndarray:
        """
        Visualize tracking results on frame.
        
        Args:
            frame: Input image frame
            tracks: List of tracks
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw detections
        for detection in detections:
            x, y, w, h = detection.bbox
            cv2.rectangle(vis_frame, (int(x), int(y)), (int(x + w), int(y + h)),
                         (0, 255, 0), 2)
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(vis_frame, label, (int(x), int(y - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tracks
        for track in tracks:
            x, y, w, h = track.bbox
            color = (0, 0, 255) if track.hits >= self.tracker.min_hits else (255, 0, 0)
            cv2.rectangle(vis_frame, (int(x), int(y)), (int(x + w), int(y + h)),
                         color, 2)
            label = f"ID {track.track_id}: {track.class_name}"
            cv2.putText(vis_frame, label, (int(x), int(y - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    def process_video(self, video_path: str, output_path: str) -> None:
        """
        Process video file with tracking.
        
        Args:
            video_path: Input video path
            output_path: Output video path
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track objects
            tracks, detections = self.track(frame)
            
            # Visualize
            vis_frame = self.visualize(frame, tracks, detections)
            
            # Write frame
            out.write(vis_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        logger.info(f"Video processing complete. Output saved to {output_path}")
