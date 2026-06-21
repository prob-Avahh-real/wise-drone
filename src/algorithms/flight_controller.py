"""
Autonomous flight controller for drone navigation.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class FlightState(Enum):
    """Flight state machine states."""
    IDLE = "idle"
    ARMING = "arming"
    TAKING_OFF = "taking_off"
    HOVERING = "hovering"
    NAVIGATING = "navigating"
    TRACKING = "tracking"
    LANDING = "landing"
    EMERGENCY = "emergency"
    RETURNING_HOME = "returning_home"


class PIDController:
    """PID controller for stable flight control."""
    
    def __init__(self, kp: float, ki: float, kd: float, output_limits: Tuple[float, float] = (-1, 1)):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Output saturation limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()
    
    def update(self, setpoint: float, measured: float) -> float:
        """
        Update PID controller.
        
        Args:
            setpoint: Desired value
            measured: Measured value
            
        Returns:
            Control output
        """
        current_time = time.time()
        dt = current_time - self.previous_time
        
        if dt <= 0:
            return 0.0
        
        # Calculate error
        error = setpoint - measured
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Update state
        self.previous_error = error
        self.previous_time = current_time
        
        return float(output)
    
    def reset(self) -> None:
        """Reset PID controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()


class FlightController:
    """Autonomous flight controller with state machine and safety protocols."""
    
    def __init__(self, drone_interface, config: dict):
        """
        Initialize flight controller.
        
        Args:
            drone_interface: Drone interface object
            config: Flight controller configuration
        """
        self.drone = drone_interface
        self.config = config
        self.pid_config = config.get('pid', {})
        self.safety_config = config.get('safety', {})
        
        # Initialize state machine
        self.state = FlightState.IDLE
        self.target_position = None
        self.current_path = []
        self.current_path_index = 0
        
        # Initialize PID controllers
        self._init_pid_controllers()
        
        # Safety parameters
        self.max_altitude = self.safety_config.get('max_altitude', 120.0)
        self.max_speed = self.safety_config.get('max_speed', 15.0)
        self.min_battery = self.safety_config.get('min_battery', 20)
        self.emergency_landing = self.safety_config.get('emergency_landing', True)
        self.rth_on_signal_loss = self.safety_config.get('rth_on_signal_loss', True)
        
        # Home position
        self.home_position = None
        
        # Tracking target
        self.tracking_target_id = None
    
    def _init_pid_controllers(self) -> None:
        """Initialize PID controllers for position, velocity, and attitude."""
        # Position PID
        pos_config = self.pid_config.get('position', {})
        self.pid_x = PIDController(pos_config.get('kp', 1.0), pos_config.get('ki', 0.0), pos_config.get('kd', 0.5))
        self.pid_y = PIDController(pos_config.get('kp', 1.0), pos_config.get('ki', 0.0), pos_config.get('kd', 0.5))
        self.pid_z = PIDController(pos_config.get('kp', 1.0), pos_config.get('ki', 0.0), pos_config.get('kd', 0.5))
        
        # Velocity PID
        vel_config = self.pid_config.get('velocity', {})
        self.pid_vx = PIDController(vel_config.get('kp', 2.0), vel_config.get('ki', 0.1), vel_config.get('kd', 0.8))
        self.pid_vy = PIDController(vel_config.get('kp', 2.0), vel_config.get('ki', 0.1), vel_config.get('kd', 0.8))
        self.pid_vz = PIDController(vel_config.get('kp', 2.0), vel_config.get('ki', 0.1), vel_config.get('kd', 0.8))
        
        # Attitude PID
        att_config = self.pid_config.get('attitude', {})
        self.pid_roll = PIDController(att_config.get('kp', 3.0), att_config.get('ki', 0.0), att_config.get('kd', 1.0))
        self.pid_pitch = PIDController(att_config.get('kp', 3.0), att_config.get('ki', 0.0), att_config.get('kd', 1.0))
        self.pid_yaw = PIDController(att_config.get('kp', 3.0), att_config.get('ki', 0.0), att_config.get('kd', 1.0))
    
    def arm(self) -> bool:
        """
        Arm the drone.
        
        Returns:
            True if successful, False otherwise
        """
        if self.state != FlightState.IDLE:
            logger.warning(f"Cannot arm in state {self.state}")
            return False
        
        self.state = FlightState.ARMING
        success = self.drone.arm()
        
        if success:
            logger.info("Drone armed successfully")
            self.state = FlightState.IDLE
        else:
            logger.error("Failed to arm drone")
            self.state = FlightState.IDLE
        
        return success
    
    def disarm(self) -> bool:
        """
        Disarm the drone.
        
        Returns:
            True if successful, False otherwise
        """
        if self.state == FlightState.NAVIGATING or self.state == FlightState.TRACKING:
            logger.warning("Cannot disarm while navigating or tracking")
            return False
        
        success = self.drone.disarm()
        
        if success:
            logger.info("Drone disarmed successfully")
            self.state = FlightState.IDLE
        else:
            logger.error("Failed to disarm drone")
        
        return success
    
    def takeoff(self, altitude: float) -> bool:
        """
        Take off to specified altitude.
        
        Args:
            altitude: Target altitude in meters
            
        Returns:
            True if successful, False otherwise
        """
        if self.state != FlightState.IDLE:
            logger.warning(f"Cannot takeoff in state {self.state}")
            return False
        
        # Check safety
        if altitude > self.max_altitude:
            logger.warning(f"Altitude {altitude}m exceeds maximum {self.max_altitude}m")
            return False
        
        self.state = FlightState.TAKING_OFF
        success = self.drone.takeoff(altitude)
        
        if success:
            logger.info(f"Drone took off to {altitude}m")
            self.state = FlightState.HOVERING
            self.home_position = self.drone.get_position()
        else:
            logger.error("Failed to take off")
            self.state = FlightState.IDLE
        
        return success
    
    def land(self) -> bool:
        """
        Land the drone.
        
        Returns:
            True if successful, False otherwise
        """
        if self.state == FlightState.EMERGENCY:
            logger.info("Emergency landing")
        
        self.state = FlightState.LANDING
        success = self.drone.land()
        
        if success:
            logger.info("Drone landed successfully")
            self.state = FlightState.IDLE
            self._reset_pid_controllers()
        else:
            logger.error("Failed to land")
            self.state = FlightState.EMERGENCY
        
        return success
    
    def navigate_to(self, position: Tuple[float, float, float], speed: float = 5.0) -> bool:
        """
        Navigate to specified position.
        
        Args:
            position: Target position (x, y, z) in meters
            speed: Target speed in m/s
            
        Returns:
            True if navigation started successfully, False otherwise
        """
        # Check safety
        if position[2] > self.max_altitude:
            logger.warning(f"Altitude {position[2]}m exceeds maximum {self.max_altitude}m")
            return False
        
        if speed > self.max_speed:
            logger.warning(f"Speed {speed}m/s exceeds maximum {self.max_speed}m/s")
            speed = self.max_speed
        
        self.state = FlightState.NAVIGATING
        self.target_position = position
        
        success = self.drone.move_to_position(position, speed)
        
        if success:
            logger.info(f"Navigating to {position} at {speed}m/s")
        else:
            logger.error("Failed to start navigation")
            self.state = FlightState.HOVERING
        
        return success
    
    def navigate_path(self, path: List[Tuple[float, float, float]], speed: float = 5.0) -> bool:
        """
        Navigate through a path of waypoints.
        
        Args:
            path: List of waypoint positions
            speed: Target speed in m/s
            
        Returns:
            True if navigation started successfully, False otherwise
        """
        if not path:
            logger.warning("Empty path provided")
            return False
        
        self.state = FlightState.NAVIGATING
        self.current_path = path
        self.current_path_index = 0
        
        # Start navigation to first waypoint
        return self.navigate_to(path[0], speed)
    
    def update_navigation(self) -> bool:
        """
        Update navigation progress.
        
        Returns:
            True if navigation complete, False otherwise
        """
        if self.state != FlightState.NAVIGATING:
            return True
        
        # Check if reached current waypoint
        current_position = self.drone.get_position()
        if current_position is None:
            return False
        
        target = self.current_path[self.current_path_index]
        distance = np.linalg.norm(np.array(current_position) - np.array(target))
        
        if distance < 1.0:  # Within 1 meter
            logger.info(f"Reached waypoint {self.current_path_index}")
            self.current_path_index += 1
            
            # Check if path complete
            if self.current_path_index >= len(self.current_path):
                logger.info("Navigation complete")
                self.state = FlightState.HOVERING
                return True
            
            # Navigate to next waypoint
            self.navigate_to(self.current_path[self.current_path_index])
        
        return False
    
    def start_tracking(self, target_id: int) -> bool:
        """
        Start tracking a target.
        
        Args:
            target_id: Target track ID to follow
            
        Returns:
            True if tracking started successfully, False otherwise
        """
        self.state = FlightState.TRACKING
        self.tracking_target_id = target_id
        logger.info(f"Started tracking target {target_id}")
        return True
    
    def stop_tracking(self) -> bool:
        """
        Stop tracking and return to hover.
        
        Returns:
            True if successful, False otherwise
        """
        self.state = FlightState.HOVERING
        self.tracking_target_id = None
        logger.info("Stopped tracking")
        return True
    
    def update_tracking(self, target_position: Tuple[float, float, float]) -> bool:
        """
        Update tracking with new target position.
        
        Args:
            target_position: Current target position (x, y, z) in meters
            
        Returns:
            True if update successful, False otherwise
        """
        if self.state != FlightState.TRACKING:
            return False
        
        # Navigate to target position with offset
        follow_distance = 5.0  # meters
        follow_altitude = target_position[2] + 2.0  # 2m above target
        
        target = (target_position[0], target_position[1], follow_altitude)
        self.drone.move_to_position(target, speed=5.0)
        
        return True
    
    def return_to_home(self) -> bool:
        """
        Return to home position.
        
        Returns:
            True if RTH started successfully, False otherwise
        """
        if self.home_position is None:
            logger.warning("No home position set")
            return False
        
        self.state = FlightState.RETURNING_HOME
        logger.info(f"Returning to home position {self.home_position}")
        
        success = self.drone.return_to_home()
        
        if success:
            logger.info("Return to home initiated")
        else:
            logger.error("Failed to initiate return to home")
            self.state = FlightState.EMERGENCY
        
        return success
    
    def emergency_stop(self) -> bool:
        """
        Emergency stop - immediately stop the drone.
        
        Returns:
            True if successful, False otherwise
        """
        logger.warning("EMERGENCY STOP")
        self.state = FlightState.EMERGENCY
        
        success = self.drone.emergency_stop()
        
        if success:
            logger.info("Emergency stop executed")
        else:
            logger.error("Emergency stop failed")
        
        return success
    
    def check_safety(self) -> bool:
        """
        Check safety conditions and take appropriate action.
        
        Returns:
            True if safe, False if safety violation detected
        """
        # Check battery level
        battery = self.drone.get_battery_level()
        if battery is not None and battery < self.min_battery:
            logger.warning(f"Low battery: {battery}%")
            if self.emergency_landing and self.state != FlightState.LANDING:
                logger.info("Initiating emergency landing due to low battery")
                self.land()
                return False
        
        # Check GPS signal
        state = self.drone.get_state()
        if state and not state.gps_signal:
            logger.warning("GPS signal lost")
            if self.rth_on_signal_loss and self.state not in [FlightState.LANDING, FlightState.EMERGENCY]:
                logger.info("Initiating return to home due to GPS signal loss")
                self.return_to_home()
                return False
        
        # Check altitude
        if state and state.position[2] > self.max_altitude:
            logger.warning(f"Altitude {state.position[2]}m exceeds maximum {self.max_altitude}m")
            self.emergency_stop()
            return False
        
        return True
    
    def get_state(self) -> FlightState:
        """
        Get current flight state.
        
        Returns:
            Current flight state
        """
        return self.state
    
    def _reset_pid_controllers(self) -> None:
        """Reset all PID controllers."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_vx.reset()
        self.pid_vy.reset()
        self.pid_vz.reset()
        self.pid_roll.reset()
        self.pid_pitch.reset()
        self.pid_yaw.reset()
