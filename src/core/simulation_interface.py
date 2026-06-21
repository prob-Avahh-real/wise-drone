"""
Simulation interface for AirSim drone control.
"""
import time
import numpy as np
from typing import Tuple, Optional, List
import logging

from .drone_interface import DroneInterface, DroneState, Waypoint

logger = logging.getLogger(__name__)


class SimulationInterface(DroneInterface):
    """AirSim-based drone simulation interface."""
    
    def __init__(self, config: dict):
        """
        Initialize the simulation interface.
        
        Args:
            config: Configuration dictionary with simulation settings
        """
        super().__init__(config)
        self.sim_config = config.get('simulation', {})
        self.client = None
        self.home_position = None
        self._try_import_airsim()
    
    def _try_import_airsim(self):
        """Try to import AirSim, provide fallback if not available."""
        try:
            import airsim
            self.airsim = airsim
            logger.info("AirSim imported successfully")
        except ImportError:
            logger.warning("AirSim not installed. Simulation mode will use mock implementation.")
            self.airsim = None
    
    def connect(self) -> bool:
        """
        Connect to AirSim simulation.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.airsim is None:
            logger.warning("AirSim not available, using mock connection")
            self._connected = True
            self._state = DroneState(
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                battery_level=100.0,
                is_armed=False,
                is_flying=False,
                gps_signal=True,
                timestamp=time.time()
            )
            return True
        
        try:
            self.client = self.airsim.MultirotorClient(
                ip=self.sim_config.get('host', '127.0.0.1'),
                port=self.sim_config.get('port', 41451)
            )
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self._connected = True
            logger.info("Connected to AirSim successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to AirSim: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from AirSim simulation.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        if self.client:
            try:
                self.client.enableApiControl(False)
                self._connected = False
                logger.info("Disconnected from AirSim")
                return True
            except Exception as e:
                logger.error(f"Failed to disconnect from AirSim: {e}")
                return False
        return True
    
    def is_connected(self) -> bool:
        """Check if connected to the simulation."""
        return self._connected
    
    def arm(self) -> bool:
        """
        Arm the drone motors.
        
        Returns:
            True if arming successful, False otherwise
        """
        if not self._connected:
            return False
        
        if self.airsim is None:
            self._state.is_armed = True
            logger.info("Drone armed (mock)")
            return True
        
        try:
            self.client.armDisarm(True)
            logger.info("Drone armed")
            return True
        except Exception as e:
            logger.error(f"Failed to arm drone: {e}")
            return False
    
    def disarm(self) -> bool:
        """
        Disarm the drone motors.
        
        Returns:
            True if disarming successful, False otherwise
        """
        if not self._connected:
            return False
        
        if self.airsim is None:
            self._state.is_armed = False
            logger.info("Drone disarmed (mock)")
            return True
        
        try:
            self.client.armDisarm(False)
            logger.info("Drone disarmed")
            return True
        except Exception as e:
            logger.error(f"Failed to disarm drone: {e}")
            return False
    
    def takeoff(self, altitude: float) -> bool:
        """
        Take off to specified altitude.
        
        Args:
            altitude: Target altitude in meters
            
        Returns:
            True if takeoff successful, False otherwise
        """
        if not self._connected:
            return False
        
        if self.airsim is None:
            self._state.is_flying = True
            self._state.position = (0.0, 0.0, altitude)
            logger.info(f"Drone took off to {altitude}m (mock)")
            return True
        
        try:
            self.client.takeoffAsync().join()
            # Move to target altitude
            self.client.moveToZAsync(altitude, velocity=5.0).join()
            self._state.is_flying = True
            logger.info(f"Drone took off to {altitude}m")
            return True
        except Exception as e:
            logger.error(f"Failed to take off: {e}")
            return False
    
    def land(self) -> bool:
        """
        Land the drone.
        
        Returns:
            True if landing successful, False otherwise
        """
        if not self._connected:
            return False
        
        if self.airsim is None:
            self._state.is_flying = False
            self._state.position = (0.0, 0.0, 0.0)
            logger.info("Drone landed (mock)")
            return True
        
        try:
            self.client.landAsync().join()
            self._state.is_flying = False
            logger.info("Drone landed")
            return True
        except Exception as e:
            logger.error(f"Failed to land: {e}")
            return False
    
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
        if not self._connected:
            return False
        
        if self.airsim is None:
            self._state.position = position
            logger.info(f"Drone moved to {position} (mock)")
            return True
        
        try:
            x, y, z = position
            self.client.moveToPositionAsync(
                x, y, z, speed,
                yaw_mode=self.airsim.YawMode(),
                vehicle_name=self.sim_config.get('vehicle_name', 'Drone1')
            )
            return True
        except Exception as e:
            logger.error(f"Failed to move to position: {e}")
            return False
    
    def set_velocity(self, velocity: Tuple[float, float, float]) -> bool:
        """
        Set drone velocity.
        
        Args:
            velocity: Target velocity (vx, vy, vz) in m/s
            
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self._connected:
            return False
        
        if self.airsim is None:
            self._state.velocity = velocity
            logger.info(f"Drone velocity set to {velocity} (mock)")
            return True
        
        try:
            vx, vy, vz = velocity
            self.client.moveByVelocityAsync(
                vx, vy, vz,
                duration=1.0,
                vehicle_name=self.sim_config.get('vehicle_name', 'Drone1')
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set velocity: {e}")
            return False
    
    def set_orientation(self, orientation: Tuple[float, float, float]) -> bool:
        """
        Set drone orientation.
        
        Args:
            orientation: Target orientation (roll, pitch, yaw) in radians
            
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self._connected:
            return False
        
        if self.airsim is None:
            self._state.orientation = orientation
            logger.info(f"Drone orientation set to {orientation} (mock)")
            return True
        
        try:
            roll, pitch, yaw = orientation
            self.client.rotateByYawRateAsync(yaw, duration=1.0)
            return True
        except Exception as e:
            logger.error(f"Failed to set orientation: {e}")
            return False
    
    def get_state(self) -> Optional[DroneState]:
        """
        Get current drone state.
        
        Returns:
            DroneState object or None if unavailable
        """
        if not self._connected:
            return None
        
        if self.airsim is None:
            return self._state
        
        try:
            # Get simulation data
            pose = self.client.getMultirotorState().kinematics_estimated
            position = pose.position
            orientation = pose.orientation
            
            # Convert quaternion to Euler angles
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_quat([
                orientation.x_val, orientation.y_val,
                orientation.z_val, orientation.w_val
            ])
            roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
            
            # Get velocity
            linear_vel = pose.linear_velocity
            angular_vel = pose.angular_velocity
            
            # Get battery info
            battery = self.client.getBatteryInfo()
            
            self._state = DroneState(
                position=(position.x_val, position.y_val, position.z_val),
                orientation=(roll, pitch, yaw),
                velocity=(linear_vel.x_val, linear_vel.y_val, linear_vel.z_val),
                angular_velocity=(angular_vel.x_val, angular_vel.y_val, angular_vel.z_val),
                battery_level=battery.battery_percentage * 100,
                is_armed=self.client.getMultirotorState().armed,
                is_flying=self.client.getMultirotorState().landed_state == 0,
                gps_signal=True,
                timestamp=time.time()
            )
            return self._state
        except Exception as e:
            logger.error(f"Failed to get drone state: {e}")
            return None
    
    def get_image(self, camera_type: str = "front") -> Optional[np.ndarray]:
        """
        Get image from specified camera.
        
        Args:
            camera_type: Camera type (front, bottom, stereo)
            
        Returns:
            Image as numpy array or None if unavailable
        """
        if not self._connected:
            return None
        
        if self.airsim is None:
            # Return mock image
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            camera_map = {
                'front': 0,
                'bottom': 1,
                'stereo': 2
            }
            camera_id = camera_map.get(camera_type, 0)
            
            responses = self.client.simGetImages([
                self.airsim.ImageRequest(
                    camera_id,
                    self.airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=False
                )
            ])
            
            if responses:
                response = responses[0]
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                return img_rgb
            return None
        except Exception as e:
            logger.error(f"Failed to get image: {e}")
            return None
    
    def get_depth_image(self, camera_type: str = "front") -> Optional[np.ndarray]:
        """
        Get depth image from specified camera.
        
        Args:
            camera_type: Camera type (front, bottom, stereo)
            
        Returns:
            Depth image as numpy array or None if unavailable
        """
        if not self._connected:
            return None
        
        if self.airsim is None:
            # Return mock depth image
            return np.zeros((480, 640), dtype=np.float32)
        
        try:
            camera_map = {
                'front': 0,
                'bottom': 1,
                'stereo': 2
            }
            camera_id = camera_map.get(camera_type, 0)
            
            responses = self.client.simGetImages([
                self.airsim.ImageRequest(
                    camera_id,
                    self.airsim.ImageType.DepthPlanar,
                    pixels_as_float=True,
                    compress=False
                )
            ])
            
            if responses:
                response = responses[0]
                img1d = np.array(response.image_data_float, dtype=np.float32)
                depth_img = img1d.reshape(response.height, response.width)
                return depth_img
            return None
        except Exception as e:
            logger.error(f"Failed to get depth image: {e}")
            return None
    
    def navigate_waypoints(self, waypoints: List[Waypoint]) -> bool:
        """
        Navigate through a list of waypoints.
        
        Args:
            waypoints: List of Waypoint objects
            
        Returns:
            True if navigation started successfully, False otherwise
        """
        if not self._connected:
            return False
        
        if self.airsim is None:
            logger.info(f"Navigating {len(waypoints)} waypoints (mock)")
            return True
        
        try:
            for wp in waypoints:
                self.move_to_position(wp.position, wp.speed)
                if wp.wait_time > 0:
                    time.sleep(wp.wait_time)
            logger.info(f"Completed navigation through {len(waypoints)} waypoints")
            return True
        except Exception as e:
            logger.error(f"Failed to navigate waypoints: {e}")
            return False
    
    def cancel_navigation(self) -> bool:
        """
        Cancel current navigation.
        
        Returns:
            True if cancellation successful, False otherwise
        """
        if not self._connected:
            return False
        
        if self.airsim is None:
            logger.info("Navigation cancelled (mock)")
            return True
        
        try:
            self.client.cancelLastTask()
            logger.info("Navigation cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel navigation: {e}")
            return False
    
    def return_to_home(self) -> bool:
        """
        Return to home position.
        
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self._connected:
            return False
        
        if self.home_position is None:
            self.home_position = (0.0, 0.0, 0.0)
        
        return self.move_to_position(self.home_position)
    
    def emergency_stop(self) -> bool:
        """
        Emergency stop - immediately stop the drone.
        
        Returns:
            True if emergency stop successful, False otherwise
        """
        if not self._connected:
            return False
        
        if self.airsim is None:
            logger.info("Emergency stop executed (mock)")
            return True
        
        try:
            self.client.hoverAsync().join()
            logger.info("Emergency stop executed")
            return True
        except Exception as e:
            logger.error(f"Failed to execute emergency stop: {e}")
            return False
