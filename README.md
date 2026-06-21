# Smart Drone AI Algorithm Solution

智慧无人机AI算法方案 - A Python-based proof-of-concept for autonomous drone flight control with core AI algorithms.

## Features

- **Target Tracking**: YOLOv8 object detection with Kalman filter and Hungarian algorithm
- **Autonomous Flight Control**: Waypoint navigation with obstacle avoidance
- **Obstacle Detection**: Real-time obstacle detection and avoidance
- **Path Planning**: A* and RRT algorithms for optimal path generation
- **Multi-Environment Support**: Both simulation (AirSim) and real hardware (DJI/PX4/ROS)

## Project Structure

```
wise-drone/
├── config/              # Configuration files
├── src/
│   ├── core/           # Drone interface layer (simulation/hardware)
│   ├── algorithms/     # Core AI algorithms
│   ├── perception/     # Camera and sensor fusion
│   └── utils/          # Visualization and utilities
├── models/             # Trained model storage
├── tests/              # Unit and integration tests
└── examples/           # Example scripts
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Simulation Mode

```bash
python examples/run_simulation.py
```

### Target Tracking Demo

```bash
python examples/run_target_tracking.py
```

### Hardware Mode

```bash
# Edit config/config.yaml to set mode: hardware
python examples/run_simulation.py
```

## Configuration

Edit `config/config.yaml` to switch between simulation and hardware modes, and adjust algorithm parameters.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- AirSim (for simulation, optional)
- DJI SDK / PX4 (for hardware, optional)

## License

MIT License
