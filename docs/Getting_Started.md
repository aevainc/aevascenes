
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

## Download Dataset

1. Visit [scenes.aeva.com/downloads](https://scenes.aeva.com/downloads).
2. Register and agree to the license terms.
3. You will receive signed URL files (`signed_urls_train.txt`, `signed_urls_validation.txt`, `signed_urls_test.txt`).
4. Download and extract the dataset:

```bash
mkdir -p data/aevascenes_v2
bash scripts/download_dataset.sh --url-file signed_urls_train.txt --output data/aevascenes_v2
bash scripts/download_dataset.sh --url-file signed_urls_validation.txt --output data/aevascenes_v2
bash scripts/download_dataset.sh --url-file signed_urls_test.txt --output data/aevascenes_v2

# Extract archives (run from data/aevascenes_v2/)
for split in train validation test; do [ -d "$split" ] && for f in "$split"/*.tar.gz; do tar -xzf "$f" -C "$split"; done; done
```

The download script is resumable — re-running it skips completed files and continues partial downloads.

## Basic Usage

Point `--dataroot` at either a **split directory** (e.g. `data/aevascenes_v2/train`) or the **dataset root** (`data/aevascenes_v2`). When using a split directory, `list_sequences()` and visualization only include sequences from that split. Sequence membership and split assignments are defined in the metadata file bundled with the SDK (`aevascenes/metadata/aevascenes_v2_metadata.json`).

```python
from aevascenes import AevaScenes

# Initialize dataset (split directory or dataset root)
avs = AevaScenes(dataroot="data/aevascenes_v2/train")

# List available sequences
avs.list_sequences()

# Load a specific sequence
sequence_uuid = "00baf481-5e77-4365-b151-f1694222c6a0"
sequence_data = avs.load_sequence(sequence_uuid)

# Visualize a sequence (compensated clouds, vehicle frame, velocity coloring)
avs.visualize_sequence(
    sequence_uuid=sequence_uuid,
    pcd_color_mode="velocity",       # "velocity", "reflectivity", or "semantic"
    pcd_type="compensated",          # "compensated", "raw", or "raw_and_compensated"
    coordinate_frame="vehicle",      # "vehicle" or "world"
    project_points_on_image=True,
    image_downsample_factor=2,
    init_visualizer=True,
)

# Raw point clouds in world frame, LiDAR only
avs.visualize_sequence(
    sequence_uuid=sequence_uuid,
    pcd_color_mode="velocity",
    pcd_type="raw",
    coordinate_frame="world",
    include_images=False,
    init_visualizer=True,
)
```

## Visualization

The AevaScenes toolkit includes a powerful **Rerun-based visualizer** that provides:

- **3D LiDAR visualization** with configurable coloring modes
- **Raw and ego-motion-compensated point clouds** (individually or side-by-side)
- **Vehicle and world coordinate frames** with ego trajectory in world mode
- **Multi-camera view** with synchronized timestamps
- **Interactive 3D environment** accessible via desktop app or web browser
- **Real-time data streaming** with temporal navigation controls

### CLI Examples

```bash
# List available sequences
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2/train --list-sequences

# Visualize both raw and compensated point clouds
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2/train --sequence-uuid <UUID> --pcd-type raw_and_compensated

# Visualize a single sequence (defaults: compensated clouds, vehicle frame, velocity coloring)
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2/train --sequence-uuid <UUID>

# Project LiDAR points onto camera images
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2/train --sequence-uuid <UUID> --color-mode semantic --project-points

# Raw point clouds in world frame, LiDAR only
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2/train --sequence-uuid <UUID> --pcd-type raw --coordinate-frame world --no-images
```

**Options:** `--color-mode` (`velocity` | `reflectivity` | `semantic`), `--pcd-type` (`compensated` | `raw` | `raw_and_compensated`), `--coordinate-frame` (`vehicle` | `world`), `--project-points`, `--no-images`, `--image-downsample-factor` (`1` | `2` | `4` | `8`), `--no-keep-alive`, `--list-sequences`.

### Web Visualizer

Use the links printed in the terminal to launch the Rerun desktop app or the Rerun web visualizer (separate from the web visualizer hosted on [scenes.aeva.com/visualize](https://scenes.aeva.com/visualize)).

## Dataset Structure

```
aevascenes_v2/
├── train/
│   └── <sequence_uuid>/
│       ├── sequence.json               # Sequence metadata and frame list
│       ├── images/
│       │   ├── front_narrow_camera_*.jpg
│       │   ├── front_wide_camera_*.jpg
│       │   ├── left_camera_*.jpg
│       │   ├── right_camera_*.jpg
│       │   ├── rear_narrow_camera_*.jpg
│       │   └── rear_wide_camera_*.jpg
│       ├── pointcloud_compensated/
│       │   ├── front_narrow_lidar_*.npz
│       │   ├── front_wide_lidar_*.npz
│       │   ├── left_lidar_*.npz
│       │   ├── right_lidar_*.npz
│       │   ├── rear_narrow_lidar_*.npz
│       │   └── rear_wide_lidar_*.npz
│       └── pointcloud_raw/
│           ├── front_narrow_lidar_*.npz
│           ├── front_wide_lidar_*.npz
│           ├── left_lidar_*.npz
│           ├── right_lidar_*.npz
│           ├── rear_narrow_lidar_*.npz
│           └── rear_wide_lidar_*.npz
├── validation/
│   └── <sequence_uuid>/...
└── test/
    └── <sequence_uuid>/...
```

**aevascenes_v2** contains 575 sequences across train (400), validation (31), and test (100) splits. See [Dataset.md](./Dataset.md) for full statistics, annotations, and schema details.

### Requirements

- Python >= 3.10
- Dependencies: numpy, pandas, scipy, matplotlib, pillow, opencv-python, rerun-sdk==0.33.0, tqdm, rich, transformations, pyyaml, tabulate
