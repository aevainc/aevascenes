

## Installation

### Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/aevainc/aevascenes.git
cd aevascenes

# Install the package
pip install -e .
```

### Using Conda 

```bash
# Clone the repository
git clone https://github.com/aevainc/aevascenes.git
cd aevascenes

# Create and activate conda environment
conda env create -f environment.yml
conda activate aevascenes

# Install the package in development mode
pip install -e .
```


### Basic Usage

```python
from aevascenes import AevaScenes

# Initialize dataset
avs = AevaScenes(dataroot="data/aevascenes_v0.1")

# List available sequences
avs.list_sequences()

# Load a specific sequence
sequence_uuid = "3a8ce6a1-a80e-4a59-b400-9983f2b67b08"
sequence_data = avs.load_sequence(sequence_uuid)

# Visualize the sequence
avs.visualize_sequence(
    sequence_uuid=sequence_uuid,
    pcd_color_mode="velocity",  # Options: "velocity", "reflectivity", "semantic"
    project_points_on_image=True,
    image_downsample_factor=2
)
```

## Visualization

The AevaScenes toolkit includes a powerful **Rerun-based visualizer** that provides:

- **3D LiDAR visualization** with configurable coloring modes
- **Multi-camera view** with synchronized timestamps
- **Interactive 3D environment** accessible via desktop app or web browser
- **Real-time data streaming** with temporal navigation controls

### Visualization Modes

```bash
# Visualize a single sequence
python examples/visualize_aevascenes.py --dataroot <DATA_ROOT> --viz-mode sequence --sequence-uuid <UUID> --color-mode [velocity/reflectivity/semantic]

# Visualize a single sequence with points projected
python examples/visualize_aevascenes.py --dataroot <DATA_ROOT> --viz-mode sequence --sequence-uuid <UUID> --color-mode [velocity/reflectivity/semantic] --project-points

# Visualize random sampled frames from all sequences
python examples/visualize_aevascenes.py --dataroot <DATA_ROOT> --viz-mode sampled --color-mode [velocity/reflectivity/semantic] --project-points
```

### Web Visualizer

Use the links printed out in the terminal to see launch the rerun app or the rerun web-visualizer (Separate from the web-visualizer hosted on [scenes.aeva.com/visualize](https://scenes.aeva.com/visualize))

## Dataset Structure

```
aevascenes_v0.1/
├── metadata.json                    # Dataset metadata and sequence list
├── <sequence_uuid>/
│   ├── sequence.json               # Sequence metadata and frame list
│   ├── images/
│   │   ├── front_narrow_camera_*.jpg
│   │   ├── front_wide_camera_*.jpg
│   │   ├── left_camera_*.jpg
│   │   ├── right_camera_*.jpg
│   │   ├── rear_narrow_camera_*.jpg
│   │   └── rear_wide_camera_*.jpg
│   └── pointcloud_compensated/
│       ├── front_narrow_lidar_*.npz
│       ├── front_wide_lidar_*.npz
│       ├── left_lidar_*.npz
│       ├── right_lidar_*.npz
│       ├── rear_narrow_lidar_*.npz
│       └── rear_wide_lidar_*.npz
```


### Requirements
- Python >= 3.10
- Dependencies: numpy, pandas, scipy, matplotlib, pillow, opencv-python, rerun-sdk, tqdm, rich, transformations, pyyaml, tabulate