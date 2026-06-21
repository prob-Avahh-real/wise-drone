"""
Environment modeling for drone perception.
"""
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EnvironmentModel:
    """3D environment model for obstacle representation and path planning."""
    
    def __init__(self, config: dict):
        """
        Initialize environment model.
        
        Args:
            config: Environment model configuration
        """
        self.config = config
        self.grid_resolution = config.get('grid_resolution', 0.5)
        self.map_size = config.get('map_size', 100)  # meters
        self.obstacles = []
        self.occupancy_grid = None
        self._initialize_grid()
    
    def _initialize_grid(self) -> None:
        """Initialize the occupancy grid."""
        grid_dim = int(self.map_size / self.grid_resolution)
        self.occupancy_grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.uint8)
    
    def add_obstacle(self, position: Tuple[float, float, float], 
                    size: Tuple[float, float, float],
                    obstacle_type: str = 'static') -> None:
        """
        Add an obstacle to the environment model.
        
        Args:
            position: Obstacle position (x, y, z) in meters
            size: Obstacle size (width, height, depth) in meters
            obstacle_type: Type of obstacle (static, dynamic)
        """
        obstacle = {
            'position': position,
            'size': size,
            'type': obstacle_type,
            'timestamp': time.time()
        }
        self.obstacles.append(obstacle)
        self._update_occupancy_grid(obstacle)
    
    def _update_occupancy_grid(self, obstacle: Dict) -> None:
        """
        Update occupancy grid with obstacle.
        
        Args:
            obstacle: Obstacle dictionary
        """
        try:
            pos = obstacle['position']
            size = obstacle['size']
            
            # Convert to grid coordinates
            x_start = int((pos[0] - size[0]/2) / self.grid_resolution)
            x_end = int((pos[0] + size[0]/2) / self.grid_resolution)
            y_start = int((pos[1] - size[1]/2) / self.grid_resolution)
            y_end = int((pos[1] + size[1]/2) / self.grid_resolution)
            z_start = int((pos[2] - size[2]/2) / self.grid_resolution)
            z_end = int((pos[2] + size[2]/2) / self.grid_resolution)
            
            # Ensure within bounds
            grid_dim = self.occupancy_grid.shape[0]
            x_start = max(0, min(x_start, grid_dim))
            x_end = max(0, min(x_end, grid_dim))
            y_start = max(0, min(y_start, grid_dim))
            y_end = max(0, min(y_end, grid_dim))
            z_start = max(0, min(z_start, grid_dim))
            z_end = max(0, min(z_end, grid_dim))
            
            # Mark as occupied
            self.occupancy_grid[x_start:x_end, y_start:y_end, z_start:z_end] = 1
            
        except Exception as e:
            logger.error(f"Failed to update occupancy grid: {e}")
    
    def is_position_free(self, position: Tuple[float, float, float],
                        safety_margin: float = 1.0) -> bool:
        """
        Check if a position is free of obstacles.
        
        Args:
            position: Position to check (x, y, z) in meters
            safety_margin: Safety margin in meters
            
        Returns:
            True if position is free, False otherwise
        """
        try:
            # Convert to grid coordinates
            x = int(position[0] / self.grid_resolution)
            y = int(position[1] / self.grid_resolution)
            z = int(position[2] / self.grid_resolution)
            
            # Check bounds
            grid_dim = self.occupancy_grid.shape[0]
            if not (0 <= x < grid_dim and 0 <= y < grid_dim and 0 <= z < grid_dim):
                return False
            
            # Check occupancy with safety margin
            margin_cells = int(safety_margin / self.grid_resolution)
            x_start = max(0, x - margin_cells)
            x_end = min(grid_dim, x + margin_cells + 1)
            y_start = max(0, y - margin_cells)
            y_end = min(grid_dim, y + margin_cells + 1)
            z_start = max(0, z - margin_cells)
            z_end = min(grid_dim, z + margin_cells + 1)
            
            # Check if any cell in the margin is occupied
            if np.any(self.occupancy_grid[x_start:x_end, y_start:y_end, z_start:z_end]):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to check position: {e}")
            return False
    
    def get_nearest_obstacle(self, position: Tuple[float, float, float],
                            max_distance: float = 10.0) -> Optional[Dict]:
        """
        Get the nearest obstacle to a position.
        
        Args:
            position: Query position (x, y, z) in meters
            max_distance: Maximum search distance in meters
            
        Returns:
            Nearest obstacle dictionary or None if no obstacle found
        """
        try:
            nearest_obstacle = None
            min_distance = float('inf')
            
            for obstacle in self.obstacles:
                obs_pos = obstacle['position']
                distance = np.linalg.norm(np.array(position) - np.array(obs_pos))
                
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    nearest_obstacle = obstacle
            
            return nearest_obstacle
        except Exception as e:
            logger.error(f"Failed to get nearest obstacle: {e}")
            return None
    
    def get_obstacles_in_radius(self, position: Tuple[float, float, float],
                               radius: float) -> List[Dict]:
        """
        Get all obstacles within a radius of a position.
        
        Args:
            position: Center position (x, y, z) in meters
            radius: Search radius in meters
            
        Returns:
            List of obstacles within radius
        """
        try:
            obstacles_in_radius = []
            
            for obstacle in self.obstacles:
                obs_pos = obstacle['position']
                distance = np.linalg.norm(np.array(position) - np.array(obs_pos))
                
                if distance <= radius:
                    obstacles_in_radius.append(obstacle)
            
            return obstacles_in_radius
        except Exception as e:
            logger.error(f"Failed to get obstacles in radius: {e}")
            return []
    
    def clear_dynamic_obstacles(self, max_age: float = 5.0) -> None:
        """
        Clear dynamic obstacles older than max_age seconds.
        
        Args:
            max_age: Maximum age in seconds
        """
        import time
        current_time = time.time()
        
        # Filter out old dynamic obstacles
        self.obstacles = [
            obs for obs in self.obstacles
            if obs['type'] != 'dynamic' or (current_time - obs['timestamp']) < max_age
        ]
        
        # Rebuild occupancy grid
        self._initialize_grid()
        for obstacle in self.obstacles:
            self._update_occupancy_grid(obstacle)
    
    def get_free_space(self, center: Tuple[float, float, float],
                      radius: float) -> List[Tuple[float, float, float]]:
        """
        Get free space points within a radius.
        
        Args:
            center: Center position (x, y, z) in meters
            radius: Search radius in meters
            
        Returns:
            List of free space positions
        """
        try:
            free_positions = []
            
            # Sample points in the sphere
            num_samples = 100
            for _ in range(num_samples):
                # Random point in sphere
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                r = np.random.uniform(0, radius)
                
                x = center[0] + r * np.sin(phi) * np.cos(theta)
                y = center[1] + r * np.sin(phi) * np.sin(theta)
                z = center[2] + r * np.cos(phi)
                
                if self.is_position_free((x, y, z)):
                    free_positions.append((x, y, z))
            
            return free_positions
        except Exception as e:
            logger.error(f"Failed to get free space: {e}")
            return []
    
    def export_to_point_cloud(self) -> np.ndarray:
        """
        Export environment model to point cloud.
        
        Returns:
            Point cloud as numpy array (N x 3)
        """
        try:
            points = []
            
            for obstacle in self.obstacles:
                pos = obstacle['position']
                size = obstacle['size']
                
                # Sample points from obstacle
                num_points = 10
                for _ in range(num_points):
                    x = pos[0] + np.random.uniform(-size[0]/2, size[0]/2)
                    y = pos[1] + np.random.uniform(-size[1]/2, size[1]/2)
                    z = pos[2] + np.random.uniform(-size[2]/2, size[2]/2)
                    points.append([x, y, z])
            
            return np.array(points)
        except Exception as e:
            logger.error(f"Failed to export point cloud: {e}")
            return np.array([])
