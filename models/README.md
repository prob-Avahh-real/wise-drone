# Models Directory

This directory is used to store trained AI models for the Smart Drone AI Algorithm Solution.

## Model Storage Structure

```
models/
├── target_detection/
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   └── custom_models/
│       └── person_tracker.pt
├── obstacle_detection/
│   ├── depth_estimation/
│   └── semantic_segmentation/
└── path_planning/
    └── trained_models/
```

## Downloading Pre-trained Models

For target tracking, YOLOv8 models will be automatically downloaded on first use from Ultralytics.

For custom models, place them in the appropriate subdirectory.

## Training Custom Models

To train custom models, use the training scripts in the `examples/` directory and specify the output path to this directory.
