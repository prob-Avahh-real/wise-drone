"""
Abstract drone interface for simulation and hardware control.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class DroneState:
    """Represents the current state of the drone."""
    position: Tuple[float, float, float]  # x, y, z in meters
    orientation: Tuple[float, float, float]  # roll, pitch, yaw in radians
    velocity: Tuple[float, float, float]  # vx, vy, vz in m/s
    angular_velocity: Tuple[float, float, float]  # vroll, vpitch, vyaw in rad/s
    battery_level: float  # Battery percentage (0-100)
    is_armed: bool
    is_flying: bool
    gps_signal: bool
    timestamp: float


@dataclass
class Waypoint:
    """Represents a navigation waypoint."""
    position: Tuple[float, float, float]  # x, y, z in meters
    speed: float = 5.0  # Target speed in m/s
    wait_time: float = 0.0  # Time to wait at waypoint in seconds
    tolerance: float = 1.0  # Position tolerance in meters


class DroneInterface(ABC):
    """Abstract base class for drone control interfaces."""
    
    def __init__(self, config: dict):
        """
        Initialize the drone interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._state = None
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the drone.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the drone.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to the drone.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def arm(self) -> bool:
        """
        Arm the drone motors.
        
        Returns:
            True if arming successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disarm(self) -> bool:
        """
        Disarm the drone motors.
        
        Returns:
            True if disarming successful, False otherwise
        """
        pass
    
    @abstractmethod
    def takeoff(self, altitude: float) -> bool:
        """
        Take off to specified altitude.
        
        Args:
            altitude: Target altitude in meters
            
        Returns:
            True if takeoff successful, False otherwise
        """
        pass
    
    @abstractmethod
    def land(self) -> bool:
        """
        Land the drone.
        
        Returns:
            True if landing successful, False otherwise
        """
        pass
    
    @abstractmethod
    def move_to_position(self, position: Tuple[float, float, float], 
                        speed: float = 5.0) -> bool:
        """
        Move drone to specified position.
        
        Args:
            position: Target position (x, y, z) in meters
            speed: Target speed in m/s
            
        Returns:
            True if command sent successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def set_velocity(self, velocity: Tuple[float, float, float]) -> bool:
        """
        Set drone velocity.
        
        Args:
            velocity: Target velocity (vx, vy, vz) in m/s
            
        Returns:
            True if command sent successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def set_orientation(self, orientation: Tuple[float, float, float]) -> bool:
        """
        Set drone orientation.
        
        Args:
            orientation: Target orientation (roll, pitch, yaw) in radians
            
        Returns:
            True if command sent successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Optional[DroneState]:
        """
        Get current drone state.
        
        Returns:
            DroneState object or None if unavailable
        """
        pass
    
    @abstractmethod
    def get_image(self, camera_type: str = "front") -> Optional[np.ndarray]:
        """
        Get image from specified camera.
        
        Args:
            camera_type: Camera type (front, bottom, stereo)
            
        Returns:
            Image as numpy array or None if unavailable
        """
        pass
    
    @abstractmethod
    def get_depth_image(self, camera_type: str = "front") -> Optional[np.ndarray]:
        """
        Get depth image from specified camera.
        
        Args:
            camera_type: Camera type (front, bottom, stereo)
            
        Returns:
            Depth image as numpy array or None if unavailable
        """
        pass
    
    @abstractmethod
    def navigate_waypoints(self, waypoints: List[Waypoint]) -> bool:
        """
        Navigate through a list of waypoints.
        
        Args:
            waypoints: List of Waypoint objects
            
        Returns:
            True if navigation started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def cancel_navigation(self) -> bool:
        """
        Cancel current navigation.
        
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def return_to_home(self) -> bool:
        """
        Return to home position.
        
        Returns:
            True if command sent successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def emergency_stop(self) -> bool:
        """
        Emergency stop - immediately stop the drone.
        
        Returns:
            True if emergency stop successful, False otherwise
        """
        pass
    
    def get_battery_level(self) -> Optional[float]:
        """
        Get current battery level.
        
        Returns:
            Battery percentage (0-100) or None if unavailable
        """
        state = self.get_state()
        return state.battery_level if state else None
    
    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Get current position.
        
        Returns:
            Position (x, y, z) in meters or None if unavailable
        """
        state = self.get_state()
        return state.position if state else None
    
    def is_armed(self) -> bool:
        """
        Check if drone is armed.
        
        Returns:
            True if armed, False otherwise
        """
        state = self.get_state()
        return state.is_armed if state else False
    
    def is_flying(self) -> bool:
        """
        Check if drone is flying.
        
        Returns:
            True if flying, False otherwise
        """
        state = self.get_state()
        return state.is_flying if state else False
