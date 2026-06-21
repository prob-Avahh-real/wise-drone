"""
Path planning algorithms for drone navigation.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import heapq
import logging
import time

logger = logging.getLogger(__name__)


class AStarPlanner:
    """A* path planning algorithm."""
    
    def __init__(self, config: dict):
        """
        Initialize A* planner.
        
        Args:
            config: Path planning configuration
        """
        self.config = config
        self.grid_resolution = config.get('grid_resolution', 0.5)
        self.max_time = config.get('max_time', 10.0)
    
    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             obstacles: List[Dict],
             map_size: Tuple[float, float, float] = (100, 100, 50)) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan path from start to goal using A*.
        
        Args:
            start: Start position (x, y, z) in meters
            goal: Goal position (x, y, z) in meters
            obstacles: List of obstacle dictionaries
            map_size: Map size (width, height, depth) in meters
            
        Returns:
            List of waypoints or None if no path found
        """
        start_time = time.time()
        
        # Convert to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)
        
        # Create grid dimensions
        grid_dim = (
            int(map_size[0] / self.grid_resolution),
            int(map_size[1] / self.grid_resolution),
            int(map_size[2] / self.grid_resolution)
        )
        
        # Create occupancy grid
        occupancy_grid = self._create_occupancy_grid(obstacles, grid_dim)
        
        # A* search
        path_grid = self._astar_search(start_grid, goal_grid, occupancy_grid, start_time)
        
        if path_grid is None:
            logger.warning("No path found")
            return None
        
        # Convert back to world coordinates
        path_world = [self._grid_to_world(node) for node in path_grid]
        
        logger.info(f"Path found with {len(path_world)} waypoints")
        return path_world
    
    def _world_to_grid(self, position: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to grid coordinates."""
        x = int(position[0] / self.grid_resolution)
        y = int(position[1] / self.grid_resolution)
        z = int(position[2] / self.grid_resolution)
        return (x, y, z)
    
    def _grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert grid coordinates to world coordinates."""
        x = grid_pos[0] * self.grid_resolution
        y = grid_pos[1] * self.grid_resolution
        z = grid_pos[2] * self.grid_resolution
        return (x, y, z)
    
    def _create_occupancy_grid(self, obstacles: List[Dict],
                               grid_dim: Tuple[int, int, int]) -> np.ndarray:
        """Create occupancy grid from obstacles."""
        grid = np.zeros(grid_dim, dtype=np.uint8)
        
        for obs in obstacles:
            pos = obs.get('position', (0, 0, 0))
            size = obs.get('size', (1, 1, 1))
            
            # Convert to grid coordinates
            x_start = int((pos[0] - size[0]/2) / self.grid_resolution)
            x_end = int((pos[0] + size[0]/2) / self.grid_resolution)
            y_start = int((pos[1] - size[1]/2) / self.grid_resolution)
            y_end = int((pos[1] + size[1]/2) / self.grid_resolution)
            z_start = int((pos[2] - size[2]/2) / self.grid_resolution)
            z_end = int((pos[2] + size[2]/2) / self.grid_resolution)
            
            # Ensure within bounds
            x_start = max(0, min(x_start, grid_dim[0]))
            x_end = max(0, min(x_end, grid_dim[0]))
            y_start = max(0, min(y_start, grid_dim[1]))
            y_end = max(0, min(y_end, grid_dim[1]))
            z_start = max(0, min(z_start, grid_dim[2]))
            z_end = max(0, min(z_end, grid_dim[2]))
            
            # Mark as occupied
            grid[x_start:x_end, y_start:y_end, z_start:z_end] = 1
        
        return grid
    
    def _astar_search(self, start: Tuple[int, int, int], goal: Tuple[int, int, int],
                     grid: np.ndarray, start_time: float) -> Optional[List[Tuple[int, int, int]]]:
        """Perform A* search on grid."""
        # Priority queue: (f_score, g_score, position)
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        
        # Track costs and parents
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        came_from = {}
        
        # 26-connected neighborhood (3D)
        neighbors = self._get_neighbors_3d()
        
        while open_set:
            # Check time limit
            if time.time() - start_time > self.max_time:
                logger.warning("A* search timed out")
                return None
            
            # Get node with lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)
            
            # Check if reached goal
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for neighbor in neighbors:
                neighbor_pos = (
                    current[0] + neighbor[0],
                    current[1] + neighbor[1],
                    current[2] + neighbor[2]
                )
                
                # Check bounds
                if not (0 <= neighbor_pos[0] < grid.shape[0] and
                        0 <= neighbor_pos[1] < grid.shape[1] and
                        0 <= neighbor_pos[2] < grid.shape[2]):
                    continue
                
                # Check if occupied
                if grid[neighbor_pos] == 1:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + self._distance(current, neighbor_pos)
                
                # Check if this path is better
                if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current
                    g_score[neighbor_pos] = tentative_g
                    f = tentative_g + self._heuristic(neighbor_pos, goal)
                    f_score[neighbor_pos] = f
                    heapq.heappush(open_set, (f, tentative_g, neighbor_pos))
        
        return None
    
    def _get_neighbors_3d(self) -> List[Tuple[int, int, int]]:
        """Get 26-connected neighborhood in 3D."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((dx, dy, dz))
        return neighbors
    
    def _heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    
    def _distance(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """Euclidean distance between two grid positions."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Reconstruct path from came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class RRTPlanner:
    """RRT (Rapidly-exploring Random Tree) path planning algorithm."""
    
    def __init__(self, config: dict):
        """
        Initialize RRT planner.
        
        Args:
            config: Path planning configuration
        """
        self.config = config
        self.grid_resolution = config.get('grid_resolution', 0.5)
        self.max_time = config.get('max_time', 10.0)
        self.rrt_config = config.get('rrt', {})
        self.max_iterations = self.rrt_config.get('max_iterations', 1000)
        self.step_size = self.rrt_config.get('step_size', 1.0)
        self.goal_sample_rate = self.rrt_config.get('goal_sample_rate', 0.1)
    
    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             obstacles: List[Dict],
             map_size: Tuple[float, float, float] = (100, 100, 50)) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan path from start to goal using RRT.
        
        Args:
            start: Start position (x, y, z) in meters
            goal: Goal position (x, y, z) in meters
            obstacles: List of obstacle dictionaries
            map_size: Map size (width, height, depth) in meters
            
        Returns:
            List of waypoints or None if no path found
        """
        start_time = time.time()
        
        # Initialize tree
        tree = {0: {'position': start, 'parent': None}}
        
        for i in range(self.max_iterations):
            # Check time limit
            if time.time() - start_time > self.max_time:
                logger.warning("RRT search timed out")
                break
            
            # Sample random point
            if np.random.random() < self.goal_sample_rate:
                random_point = goal
            else:
                random_point = self._sample_random_point(map_size)
            
            # Find nearest node in tree
            nearest_node_id = self._find_nearest_node(tree, random_point)
            nearest_position = tree[nearest_node_id]['position']
            
            # Steer towards random point
            new_position = self._steer(nearest_position, random_point)
            
            # Check if new position is valid
            if not self._is_valid_position(new_position, obstacles, map_size):
                continue
            
            # Check if path is collision-free
            if not self._is_collision_free(nearest_position, new_position, obstacles):
                continue
            
            # Add new node to tree
            new_node_id = len(tree)
            tree[new_node_id] = {
                'position': new_position,
                'parent': nearest_node_id
            }
            
            # Check if goal is reached
            if self._distance(new_position, goal) < self.step_size:
                # Add goal to tree
                goal_node_id = len(tree)
                tree[goal_node_id] = {
                    'position': goal,
                    'parent': new_node_id
                }
                
                # Reconstruct path
                path = self._reconstruct_path(tree, goal_node_id)
                logger.info(f"RRT path found with {len(path)} waypoints after {i+1} iterations")
                return path
        
        logger.warning("RRT failed to find path")
        return None
    
    def _sample_random_point(self, map_size: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Sample random point in map."""
        x = np.random.uniform(0, map_size[0])
        y = np.random.uniform(0, map_size[1])
        z = np.random.uniform(0, map_size[2])
        return (x, y, z)
    
    def _find_nearest_node(self, tree: Dict, point: Tuple[float, float, float]) -> int:
        """Find nearest node in tree to given point."""
        nearest_id = None
        min_distance = float('inf')
        
        for node_id, node_data in tree.items():
            position = node_data['position']
            distance = self._distance(position, point)
            
            if distance < min_distance:
                min_distance = distance
                nearest_id = node_id
        
        return nearest_id
    
    def _steer(self, from_pos: Tuple[float, float, float],
               to_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Steer from one position towards another by step_size."""
        direction = np.array(to_pos) - np.array(from_pos)
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_pos
        
        unit_direction = direction / distance
        new_pos = np.array(from_pos) + unit_direction * self.step_size
        return tuple(new_pos)
    
    def _is_valid_position(self, position: Tuple[float, float, float],
                          obstacles: List[Dict],
                          map_size: Tuple[float, float, float]) -> bool:
        """Check if position is within bounds and not in obstacle."""
        # Check bounds
        if not (0 <= position[0] <= map_size[0] and
                0 <= position[1] <= map_size[1] and
                0 <= position[2] <= map_size[2]):
            return False
        
        # Check obstacles
        for obs in obstacles:
            pos = obs.get('position', (0, 0, 0))
            size = obs.get('size', (1, 1, 1))
            
            # Simple bounding box check
            if (pos[0] - size[0]/2 <= position[0] <= pos[0] + size[0]/2 and
                pos[1] - size[1]/2 <= position[1] <= pos[1] + size[1]/2 and
                pos[2] - size[2]/2 <= position[2] <= pos[2] + size[2]/2):
                return False
        
        return True
    
    def _is_collision_free(self, from_pos: Tuple[float, float, float],
                          to_pos: Tuple[float, float, float],
                          obstacles: List[Dict]) -> bool:
        """Check if path between two positions is collision-free."""
        # Sample points along the path
        num_samples = 10
        for i in range(num_samples + 1):
            t = i / num_samples
            point = (
                from_pos[0] + t * (to_pos[0] - from_pos[0]),
                from_pos[1] + t * (to_pos[1] - from_pos[1]),
                from_pos[2] + t * (to_pos[2] - from_pos[2])
            )
            
            if not self._is_valid_position(point, obstacles, (100, 100, 50)):
                return False
        
        return True
    
    def _distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """Euclidean distance between two positions."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    
    def _reconstruct_path(self, tree: Dict, goal_id: int) -> List[Tuple[float, float, float]]:
        """Reconstruct path from tree."""
        path = []
        current_id = goal_id
        
        while current_id is not None:
            path.append(tree[current_id]['position'])
            current_id = tree[current_id]['parent']
        
        path.reverse()
        return path


class PathPlanner:
    """Main path planning interface."""
    
    def __init__(self, config: dict):
        """
        Initialize path planner.
        
        Args:
            config: Path planning configuration
        """
        self.config = config
        self.algorithm = config.get('algorithm', 'astar')
        
        if self.algorithm == 'astar':
            self.planner = AStarPlanner(config)
        elif self.algorithm == 'rrt':
            self.planner = RRTPlanner(config)
        else:
            logger.warning(f"Unknown algorithm {self.algorithm}, defaulting to A*")
            self.planner = AStarPlanner(config)
    
    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             obstacles: List[Dict],
             map_size: Tuple[float, float, float] = (100, 100, 50)) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan path from start to goal.
        
        Args:
            start: Start position (x, y, z) in meters
            goal: Goal position (x, y, z) in meters
            obstacles: List of obstacle dictionaries
            map_size: Map size (width, height, depth) in meters
            
        Returns:
            List of waypoints or None if no path found
        """
        return self.planner.plan(start, goal, obstacles, map_size)
    
    def smooth_path(self, path: List[Tuple[float, float, float]],
                    obstacles: List[Dict]) -> List[Tuple[float, float, float]]:
        """
        Smooth path using path simplification.
        
        Args:
            path: Original path
            obstacles: List of obstacles
            
        Returns:
            Smoothed path
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Try to skip intermediate points
            for j in range(len(path) - 1, i, -1):
                if self._is_collision_free(path[i], path[j], obstacles):
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                i += 1
        
        return smoothed
    
    def _is_collision_free(self, from_pos: Tuple[float, float, float],
                          to_pos: Tuple[float, float, float],
                          obstacles: List[Dict]) -> bool:
        """Check if path between two positions is collision-free."""
        num_samples = 10
        for i in range(num_samples + 1):
            t = i / num_samples
            point = (
                from_pos[0] + t * (to_pos[0] - from_pos[0]),
                from_pos[1] + t * (to_pos[1] - from_pos[1]),
                from_pos[2] + t * (to_pos[2] - from_pos[2])
            )
            
            # Simple obstacle check
            for obs in obstacles:
                pos = obs.get('position', (0, 0, 0))
                size = obs.get('size', (1, 1, 1))
                
                if (pos[0] - size[0]/2 <= point[0] <= pos[0] + size[0]/2 and
                    pos[1] - size[1]/2 <= point[1] <= pos[1] + size[1]/2 and
                    pos[2] - size[2]/2 <= point[2] <= pos[2] + size[2]/2):
                    return False
        
        return True
