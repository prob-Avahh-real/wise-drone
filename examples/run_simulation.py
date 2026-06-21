"""
Example script for running drone simulation with autonomous flight.
"""
import sys
import os
import yaml
import logging
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.simulation_interface import SimulationInterface
from algorithms.flight_controller import FlightController
from algorithms.path_planning import PathPlanner
from perception.camera import Camera
from utils.visualization import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main simulation example."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize drone interface
    logger.info("Initializing drone interface...")
    drone = SimulationInterface(config['drone'])
    
    # Connect to drone
    logger.info("Connecting to drone...")
    if not drone.connect():
        logger.error("Failed to connect to drone")
        return
    
    # Initialize flight controller
    logger.info("Initializing flight controller...")
    flight_controller = FlightController(drone, config['flight_controller'])
    
    # Initialize camera
    logger.info("Initializing camera...")
    camera = Camera(drone, config['camera'])
    
    # Initialize visualizer
    logger.info("Initializing visualizer...")
    visualizer = Visualizer(config['visualization'])
    
    # Initialize path planner
    logger.info("Initializing path planner...")
    path_planner = PathPlanner(config['path_planning'])
    
    try:
        # Arm drone
        logger.info("Arming drone...")
        if not flight_controller.arm():
            logger.error("Failed to arm drone")
            return
        
        time.sleep(1)
        
        # Take off
        logger.info("Taking off...")
        if not flight_controller.takeoff(altitude=10.0):
            logger.error("Failed to take off")
            return
        
        time.sleep(2)
        
        # Define waypoints for navigation
        waypoints = [
            (10, 0, 10),
            (10, 10, 10),
            (0, 10, 10),
            (0, 0, 10)
        ]
        
        logger.info(f"Navigating through {len(waypoints)} waypoints...")
        
        # Navigate through waypoints
        for i, waypoint in enumerate(waypoints):
            logger.info(f"Moving to waypoint {i+1}/{len(waypoints)}: {waypoint}")
            
            if not flight_controller.navigate_to(waypoint, speed=5.0):
                logger.error(f"Failed to navigate to waypoint {waypoint}")
                break
            
            # Wait to reach waypoint
            time.sleep(3)
            
            # Check safety
            if not flight_controller.check_safety():
                logger.warning("Safety check failed")
                break
            
            # Get camera frame
            frame = camera.get_frame()
            if frame is not None:
                # Get drone state
                drone_state = drone.get_state()
                state_dict = {
                    'position': drone_state.position if drone_state else (0, 0, 0),
                    'velocity': drone_state.velocity if drone_state else (0, 0, 0),
                    'battery_level': drone_state.battery_level if drone_state else 0,
                    'flight_state': flight_controller.get_state().value
                }
                
                # Visualize
                vis_frame = visualizer.draw_drone_state(frame, state_dict)
                visualizer.display_frame(vis_frame)
        
        # Return to home
        logger.info("Returning to home...")
        flight_controller.return_to_home()
        time.sleep(5)
        
        # Land
        logger.info("Landing...")
        if not flight_controller.land():
            logger.error("Failed to land")
            return
        
        logger.info("Simulation complete!")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        flight_controller.emergency_stop()
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        flight_controller.emergency_stop()
    finally:
        # Disconnect
        logger.info("Disconnecting from drone...")
        drone.disconnect()


if __name__ == "__main__":
    main()
