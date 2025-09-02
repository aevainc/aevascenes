# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# The AevaScenes dataset is licensed separately under the AevaScenes Dataset License.
# See https://scenes.aeva.com/license for the full license text.
# fmt: off

from typing import Dict, List

# Color mapping for semantic class visualization
# RGB values in range [0, 255] for consistent rendering across platforms
class_color_map: Dict[str, List[int]] = {
    "unknown": [0, 0, 0],  # black
    
    # Vehicle categories - blue spectrum for easy grouping
    "car": [65, 200, 225],  # light blue
    "bus": [0, 140, 180],  # teal blue
    "truck": [0, 100, 150],  # deeper teal
    "trailer": [0, 60, 110],  # dark slate blue
    "vehicle_on_rails": [0, 30, 80],  # navy blue
    "other_vehicle": [100, 100, 255],  # lavender blue
    
    # Two-wheeled vehicles and riders - purple/magenta spectrum
    "bicycle": [150, 100, 200],  # soft purple
    "motorcycle": [200, 100, 200],  # light magenta
    "motorcyclist": [255, 0, 100],  # bright pink-red
    "bicyclist": [200, 0, 200],  # violet
    
    # Living entities - red/brown spectrum for visibility
    "pedestrian": [255, 0, 0],  # red
    "animal": [120, 70, 50],  # muted brown-orange
    
    # Traffic infrastructure - yellow spectrum for high visibility
    "traffic_item": [255, 255, 0],  # yellow
    "traffic_sign": [255, 255, 0],  # yellow
    
    # Vertical structures - brown/yellow spectrum
    "pole_trunk": [135, 90, 0],  # brown
    "building": [200, 200, 0],  # yellow
    "other_structure": [200, 200, 0],  # yellow
    
    # Natural elements
    "vegetation": [0, 175, 0],  # green
    
    # Road surfaces and markings - distinct colors for navigation
    "road": [255, 0, 255],  # pink/magenta
    "lane_boundary": [247, 161, 111],  # orange
    "road_marking": [247, 161, 111],  # orange
    "reflective_marker": [255, 255, 255],  # white
    "sidewalk": [85, 0, 100],  # purple
    "other_ground": [150, 100, 0],  # brown
}
