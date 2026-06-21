"""
Unit tests for flight controller.
"""
import sys
import os
import unittest
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.flight_controller import FlightController, FlightState, PIDController


class TestPIDController(unittest.TestCase):
    """Test PID controller implementation."""
    
    def test_initialization(self):
        """Test PID controller initialization."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.5)
        
        self.assertEqual(pid.kp, 1.0)
        self.assertEqual(pid.ki, 0.1)
        self.assertEqual(pid.kd, 0.5)
        self.assertEqual(pid.integral, 0.0)
        self.assertEqual(pid.previous_error, 0.0)
    
    def test_update(self):
        """Test PID controller update."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.5)
        
        output = pid.update(setpoint=10.0, measured=5.0)
        
        self.assertIsNotNone(output)
        self.assertIsInstance(output, float)
    
    def test_reset(self):
        """Test PID controller reset."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.5)
        
        # Update to change state
        pid.update(setpoint=10.0, measured=5.0)
        
        # Reset
        pid.reset()
        
        self.assertEqual(pid.integral, 0.0)
        self.assertEqual(pid.previous_error, 0.0)
    
    def test_output_limits(self):
        """Test PID controller output limits."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0))
        
        # Test upper limit
        output = pid.update(setpoint=100.0, measured=0.0)
        self.assertLessEqual(output, 1.0)
        
        # Test lower limit
        output = pid.update(setpoint=0.0, measured=100.0)
        self.assertGreaterEqual(output, -1.0)


class TestFlightController(unittest.TestCase):
    """Test flight controller implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock drone interface
        self.mock_drone = Mock()
        self.mock_drone.get_state.return_value = Mock(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            battery_level=100.0,
            gps_signal=True
        )
        self.mock_drone.get_position.return_value = (0.0, 0.0, 0.0)
        self.mock_drone.get_battery_level.return_value = 100.0
        
        # Create flight controller config
        self.config = {
            'pid': {
                'position': {'kp': 1.0, 'ki': 0.0, 'kd': 0.5},
                'velocity': {'kp': 2.0, 'ki': 0.1, 'kd': 0.8},
                'attitude': {'kp': 3.0, 'ki': 0.0, 'kd': 1.0}
            },
            'safety': {
                'max_altitude': 120.0,
                'max_speed': 15.0,
                'min_battery': 20,
                'emergency_landing': True,
                'rth_on_signal_loss': True
            }
        }
        
        self.flight_controller = FlightController(self.mock_drone, self.config)
    
    def test_initialization(self):
        """Test flight controller initialization."""
        self.assertIsNotNone(self.flight_controller)
        self.assertEqual(self.flight_controller.state, FlightState.IDLE)
        self.assertIsNone(self.flight_controller.target_position)
    
    def test_arm(self):
        """Test arming the drone."""
        self.mock_drone.arm.return_value = True
        
        success = self.flight_controller.arm()
        
        self.assertTrue(success)
        self.mock_drone.arm.assert_called_once()
    
    def test_arm_in_wrong_state(self):
        """Test arming in wrong state."""
        self.flight_controller.state = FlightState.NAVIGATING
        
        success = self.flight_controller.arm()
        
        self.assertFalse(success)
        self.mock_drone.arm.assert_not_called()
    
    def test_disarm(self):
        """Test disarming the drone."""
        self.mock_drone.disarm.return_value = True
        
        success = self.flight_controller.disarm()
        
        self.assertTrue(success)
        self.mock_drone.disarm.assert_called_once()
    
    def test_takeoff(self):
        """Test takeoff."""
        self.mock_drone.takeoff.return_value = True
        
        success = self.flight_controller.takeoff(altitude=10.0)
        
        self.assertTrue(success)
        self.mock_drone.takeoff.assert_called_once_with(10.0)
        self.assertEqual(self.flight_controller.state, FlightState.HOVERING)
    
    def test_takeoff_exceeds_max_altitude(self):
        """Test takeoff with altitude exceeding maximum."""
        success = self.flight_controller.takeoff(altitude=200.0)
        
        self.assertFalse(success)
        self.mock_drone.takeoff.assert_not_called()
    
    def test_land(self):
        """Test landing."""
        self.mock_drone.land.return_value = True
        self.flight_controller.state = FlightState.HOVERING
        
        success = self.flight_controller.land()
        
        self.assertTrue(success)
        self.mock_drone.land.assert_called_once()
        self.assertEqual(self.flight_controller.state, FlightState.IDLE)
    
    def test_navigate_to(self):
        """Test navigation to position."""
        self.mock_drone.move_to_position.return_value = True
        
        success = self.flight_controller.navigate_to(position=(10.0, 10.0, 10.0), speed=5.0)
        
        self.assertTrue(success)
        self.mock_drone.move_to_position.assert_called_once()
        self.assertEqual(self.flight_controller.state, FlightState.NAVIGATING)
    
    def test_navigate_exceeds_max_speed(self):
        """Test navigation with speed exceeding maximum."""
        self.mock_drone.move_to_position.return_value = True
        
        success = self.flight_controller.navigate_to(position=(10.0, 10.0, 10.0), speed=20.0)
        
        self.assertTrue(success)
        # Should call with max speed
        self.mock_drone.move_to_position.assert_called_once()
    
    def test_navigate_path(self):
        """Test navigation through path."""
        self.mock_drone.move_to_position.return_value = True
        
        path = [(10.0, 0.0, 10.0), (10.0, 10.0, 10.0), (0.0, 10.0, 10.0)]
        success = self.flight_controller.navigate_path(path, speed=5.0)
        
        self.assertTrue(success)
        self.assertEqual(len(self.flight_controller.current_path), len(path))
    
    def test_emergency_stop(self):
        """Test emergency stop."""
        self.mock_drone.emergency_stop.return_value = True
        
        success = self.flight_controller.emergency_stop()
        
        self.assertTrue(success)
        self.mock_drone.emergency_stop.assert_called_once()
        self.assertEqual(self.flight_controller.state, FlightState.EMERGENCY)
    
    def test_return_to_home(self):
        """Test return to home."""
        self.mock_drone.return_to_home.return_value = True
        self.flight_controller.home_position = (0.0, 0.0, 0.0)
        
        success = self.flight_controller.return_to_home()
        
        self.assertTrue(success)
        self.mock_drone.return_to_home.assert_called_once()
        self.assertEqual(self.flight_controller.state, FlightState.RETURNING_HOME)
    
    def test_check_safety_low_battery(self):
        """Test safety check with low battery."""
        self.mock_drone.get_battery_level.return_value = 15.0
        self.mock_drone.land.return_value = True
        self.flight_controller.state = FlightState.NAVIGATING
        
        safe = self.flight_controller.check_safety()
        
        self.assertFalse(safe)
        self.mock_drone.land.assert_called_once()
    
    def test_check_safety_gps_loss(self):
        """Test safety check with GPS loss."""
        self.mock_drone.get_state.return_value = Mock(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            battery_level=100.0,
            gps_signal=False
        )
        self.mock_drone.return_to_home.return_value = True
        self.flight_controller.home_position = (0.0, 0.0, 0.0)
        self.flight_controller.state = FlightState.NAVIGATING
        
        safe = self.flight_controller.check_safety()
        
        self.assertFalse(safe)
        self.mock_drone.return_to_home.assert_called_once()
    
    def test_start_tracking(self):
        """Test starting target tracking."""
        success = self.flight_controller.start_tracking(target_id=1)
        
        self.assertTrue(success)
        self.assertEqual(self.flight_controller.state, FlightState.TRACKING)
        self.assertEqual(self.flight_controller.tracking_target_id, 1)
    
    def test_stop_tracking(self):
        """Test stopping target tracking."""
        self.flight_controller.state = FlightState.TRACKING
        self.flight_controller.tracking_target_id = 1
        
        success = self.flight_controller.stop_tracking()
        
        self.assertTrue(success)
        self.assertEqual(self.flight_controller.state, FlightState.HOVERING)
        self.assertIsNone(self.flight_controller.tracking_target_id)


if __name__ == '__main__':
    unittest.main()
