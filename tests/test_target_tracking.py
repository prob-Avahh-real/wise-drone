"""
Unit tests for target tracking algorithm.
"""
import sys
import os
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.target_tracking import KalmanFilter, ObjectTracker, Detection, Track


class TestKalmanFilter(unittest.TestCase):
    """Test Kalman filter implementation."""
    
    def test_initialization(self):
        """Test Kalman filter initialization."""
        bbox = (100.0, 100.0, 50.0, 50.0)
        kf = KalmanFilter(bbox)
        
        self.assertIsNotNone(kf.state)
        self.assertEqual(kf.state.shape, (8, 1))
    
    def test_prediction(self):
        """Test Kalman filter prediction."""
        bbox = (100.0, 100.0, 50.0, 50.0)
        kf = KalmanFilter(bbox)
        
        predicted_state = kf.predict()
        
        self.assertIsNotNone(predicted_state)
        self.assertEqual(predicted_state.shape, (8, 1))
    
    def test_update(self):
        """Test Kalman filter update."""
        bbox = (100.0, 100.0, 50.0, 50.0)
        kf = KalmanFilter(bbox)
        
        # Predict
        kf.predict()
        
        # Update with new measurement
        measurement = np.array([105.0, 105.0, 50.0, 50.0]).reshape(4, 1)
        updated_state = kf.update(measurement)
        
        self.assertIsNotNone(updated_state)
        self.assertEqual(updated_state.shape, (8, 1))
    
    def test_get_bbox(self):
        """Test getting bounding box from state."""
        bbox = (100.0, 100.0, 50.0, 50.0)
        kf = KalmanFilter(bbox)
        
        retrieved_bbox = kf.get_bbox()
        
        self.assertEqual(len(retrieved_bbox), 4)
        self.assertIsInstance(retrieved_bbox, tuple)


class TestObjectTracker(unittest.TestCase):
    """Test object tracker implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'process_noise': 0.1,
            'measurement_noise': 0.1,
            'initial_error': 1.0
        }
        self.tracker = ObjectTracker(self.config)
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.next_track_id, 1)
    
    def test_update_with_detections(self):
        """Test tracker update with detections."""
        detections = [
            Detection(bbox=(100.0, 100.0, 50.0, 50.0), confidence=0.9, class_id=0, class_name='person')
        ]
        
        tracks = self.tracker.update(detections)
        
        # First update should not return confirmed tracks (need min_hits)
        self.assertEqual(len(tracks), 0)
        self.assertEqual(len(self.tracker.tracks), 1)
    
    def test_track_confirmation(self):
        """Test track confirmation after multiple updates."""
        detections = [
            Detection(bbox=(100.0, 100.0, 50.0, 50.0), confidence=0.9, class_id=0, class_name='person')
        ]
        
        # Update multiple times to confirm track
        for _ in range(5):
            self.tracker.update(detections)
        
        tracks = self.tracker.update(detections)
        
        # Should have confirmed track now
        self.assertGreater(len(tracks), 0)
    
    def test_track_persistence(self):
        """Test track persistence across frames."""
        detections1 = [
            Detection(bbox=(100.0, 100.0, 50.0, 50.0), confidence=0.9, class_id=0, class_name='person')
        ]
        detections2 = [
            Detection(bbox=(105.0, 105.0, 50.0, 50.0), confidence=0.9, class_id=0, class_name='person')
        ]
        
        # Update with first detection
        self.tracker.update(detections1)
        
        # Update with second detection (slightly moved)
        self.tracker.update(detections2)
        
        # Should still have one track
        self.assertEqual(len(self.tracker.tracks), 1)
    
    def test_track_deletion(self):
        """Test track deletion after max_age."""
        detections = [
            Detection(bbox=(100.0, 100.0, 50.0, 50.0), confidence=0.9, class_id=0, class_name='person')
        ]
        
        # Confirm track
        for _ in range(5):
            self.tracker.update(detections)
        
        # Stop updating (simulate lost track)
        for _ in range(self.tracker.max_age + 5):
            self.tracker.update([])
        
        # Track should be deleted
        self.assertEqual(len(self.tracker.tracks), 0)


class TestDetection(unittest.TestCase):
    """Test detection dataclass."""
    
    def test_detection_creation(self):
        """Test detection creation."""
        detection = Detection(
            bbox=(100.0, 100.0, 50.0, 50.0),
            confidence=0.9,
            class_id=0,
            class_name='person'
        )
        
        self.assertEqual(detection.bbox, (100.0, 100.0, 50.0, 50.0))
        self.assertEqual(detection.confidence, 0.9)
        self.assertEqual(detection.class_id, 0)
        self.assertEqual(detection.class_name, 'person')


class TestTrack(unittest.TestCase):
    """Test track dataclass."""
    
    def test_track_creation(self):
        """Test track creation."""
        track = Track(
            track_id=1,
            bbox=(100.0, 100.0, 50.0, 50.0),
            confidence=0.9,
            class_id=0,
            class_name='person',
            age=0,
            hits=1,
            time_since_update=0,
            state=np.zeros((8, 1))
        )
        
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.bbox, (100.0, 100.0, 50.0, 50.0))
        self.assertEqual(track.confidence, 0.9)
        self.assertEqual(track.hits, 1)


if __name__ == '__main__':
    unittest.main()
