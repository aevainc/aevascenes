## Dataset Statistics

| **Metric** | **Value** |
|------------|-----------|
| Sequences | 100 |
| Total Frames | 10,000 |
| LiDAR Sensors | 6 FMCW LiDARs |
| Camera Sensors | 6 (4K resolution) |
| Lidar Sensing Range | Up to 450m |
| Lidar & Camera Frame Rate | 10 Hz |
| Duration per Sequence | 10 seconds |

## Sensor Configuration

### FMCW LiDAR Specifications
- **Type**: Aeva FMCW (Frequency-Modulated Continuous Wave) LiDAR
- **Configuration**:
  - 4 wide-FOV LiDARs (110 degrees)
  - 2 narrow-FOV LiDARs (35 degrees)
- **Sensing Range**:
  - Up to 400m for narrow-FOV LiDARs
  - Up to 250m for front/rear and side-facing wide-FOV LiDARs
- **Point-Level Attributes**:
  - 3D position (x, y, z)
  - **Instantaneous radial velocity** (unique to FMCW)
  - Reflectivity
  - Line Index
  - Acquisition time offset (nanoseconds)

### Camera Specifications
- **Resolution**: 3840 × 2160 (4K)
- **Frame Rate**: 10 Hz
- **Field of View**: Wide-FOV and Narrow-FOV variants, aligned with corresponding LiDARs
- **Privacy**: All images anonymized with license plates and faces blurred

## Data Composition

### Sequences & Frames
- 100 sequences, each containing 100 frames (10 seconds at 10 Hz)
- Total: 10,000 frames across diverse driving scenarios

### Storage Formats
- **Point Clouds**: Raw and ego-motion compensated as compressed NumPy arrays
- **Images**: JPEG sequences (4K resolution)
- **Annotations**: Bounding boxes and semantic point classes as JSON metadata with calibration and ego-pose information

### Synchronization
- All sensors synchronized using **Precision Time Protocol (PTP)**
- All LiDARs are frame synchronized (start-of-scan) at 10 Hz

## Annotations

### 3D Bounding Boxes
- 3D object annotations in LiDAR coordinate space
- Object type labels (25 semantic classes)
- Tracking IDs and per-object velocity vectors

### Semantic Segmentation
- Per-point semantic labels for all 6 LiDARs
- 25 semantic classes including vehicles, pedestrians, road infrastructure

### Semantic Classes
```
Vehicles: car, bus, truck, trailer, vehicle_on_rails, other_vehicle
Persons: pedestrian, motorcyclist, bicyclist
Objects: bicycle, motorcycle, animal, traffic_item, traffic_sign
Structures: pole_trunk, building, other_structure, vegetation
Surfaces: road, lane_boundary, road_marking, reflective_markers, sidewalk, other_ground
```

## Calibration & Poses

### Camera Intrinsics
- Focal length, principal point, and distortion coefficients for each camera

### Sensor Extrinsics
- All LiDARs and cameras extrinsically calibrated with respect to the ego-vehicle frame (center of rear-axle)

### Vehicle Poses
- LiDAR-inertial odometry used to estimate 6-DoF pose for each frame
