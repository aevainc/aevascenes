# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# The AevaScenes dataset is licensed separately under the AevaScenes Dataset License.
# See https://scenes.aeva.com/license for the full license text.

import copy
import json
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rerun as rr
import transformations as tf
import yaml
from rerun.datatypes import Quaternion
from scipy.spatial.transform import Rotation as scipy_rot

from aevascenes.utils import colormaps


def write_file(data: Any, filename: str, verbose: bool = True) -> None:
    """
    Write data to a file with automatic format detection based on file extension.

    Args:
        data: Data to be written. Type depends on format:
            - .pkl: Any Python object that can be pickled
            - .json: JSON-serializable object (dict, list, str, int, float, bool, None)
            - .yaml: YAML-serializable object (similar to JSON but more flexible)
            - .txt: String data
        filename: Output file path including extension
        verbose: Whether to print confirmation message after writing
    """
    if ".pkl" in filename:
        with open(filename, "wb") as file:
            pkl.dump(data, file)
    elif ".json" in filename:
        with open(filename, "w") as file:
            json.dump(data, file)
    elif ".yaml" in filename:
        with open(filename, "w") as file:
            yaml.dump(data, file, default_flow_style=False)
    elif ".txt" in filename:
        with open(filename, "w") as file:
            file.write(data)
    else:
        raise NotImplementedError
    if verbose:
        print(f"File written to: {filename}")


def read_file(filename: str) -> Any:
    """
    Read data from a file with automatic format detection based on file extension.

    Args:
        filename: Input file path including extension
    """
    if ".pkl" in filename:
        with open(filename, "rb") as file:
            data = pkl.load(file)
    elif ".json" in filename:
        with open(filename, "r") as file:
            data = json.load(file)
    elif ".yaml" in filename:
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
    elif ".npz" in filename:
        data = np.load(filename, allow_pickle=True)
    elif ".txt" in filename:
        with open(filename, "r") as file:
            data = file.read()
    else:
        raise NotImplementedError
    return data


def pose_to_matrix(pose: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Convert a pose dictionary to a 4x4 homogeneous transformation matrix.
    Transforms a pose representation containing separate translation and rotation
    (quaternion) components into a single 4x4 homogeneous transformation matrix
    suitable for 3D geometric operations.

    Args:
        pose: Pose dictionary with required structure:
            {
                "translation": {"x": float, "y": float, "z": float},
                "rotation": {"w": float, "x": float, "y": float, "z": float}
            }
            The rotation quaternion should be in [w, x, y, z] format (scalar-first).

    Returns:
        4x4 homogeneous transformation matrix as numpy.ndarray with dtype float64.
        The matrix combines rotation (top-left 3x3) and translation (top-right 3x1).
    """
    assert "translation" in pose and "rotation" in pose
    assert "x" in pose["translation"] and "y" in pose["translation"] and "z" in pose["translation"]
    assert "w" in pose["rotation"] and "x" in pose["rotation"] and "y" in pose["rotation"] and "z" in pose["rotation"]
    return tf.concatenate_matrices(
        tf.translation_matrix([pose["translation"]["x"], pose["translation"]["y"], pose["translation"]["z"]]),
        tf.quaternion_matrix(
            [pose["rotation"]["w"], pose["rotation"]["x"], pose["rotation"]["y"], pose["rotation"]["z"]]
        ),
    )


def matrix_to_pose(matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Convert a 4x4 homogeneous transformation matrix to a pose dictionary.

    Decomposes a 4x4 homogeneous transformation matrix into separate translation
    and rotation (quaternion) components, providing the inverse operation of
    pose_to_matrix().

    Args:
        matrix: A 4x4 homogeneous transformation matrix as numpy.ndarray.
                Must have shape (4, 4) and represent a valid rigid body transformation.

    Returns:
        Pose dictionary with structure:
        {
            "translation": {"x": float, "y": float, "z": float},
            "rotation": {"w": float, "x": float, "y": float, "z": float}
        }
        The rotation quaternion is in [w, x, y, z] format (scalar-first).
    """
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix."

    # Extract translation
    translation = tf.translation_from_matrix(matrix)
    translation_dict = {"x": translation[0], "y": translation[1], "z": translation[2]}

    # Extract quaternion
    quaternion = tf.quaternion_from_matrix(matrix)
    rotation_dict = {"w": quaternion[0], "x": quaternion[1], "y": quaternion[2], "z": quaternion[3]}

    # Combine into pose
    pose = {"translation": translation_dict, "rotation": rotation_dict}

    return pose


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transformation to a set of 3D points.

    Transforms 3D points using a rigid body transformation matrix. The function
    handles the conversion to homogeneous coordinates internally and returns
    the transformed 3D points.

    Args:
        transform: 4x4 homogeneous transformation matrix as numpy.ndarray.
                  Must represent a valid rigid body transformation.
        points: Array of 3D points with shape (N, 3) where N is the number of points.
               Each row represents a point [x, y, z].

    Returns:
        Transformed 3D points with same shape (N, 3) as input.
    """
    assert type(transform) == np.ndarray and transform.shape == (4, 4)
    assert points.shape[-1] == 3
    return (transform @ np.c_[points, np.ones(len(points))].T)[:3].T


def transform_bboxes(transform: np.ndarray, bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply a 4x4 homogeneous transformation to a list of 3D bounding boxes.

    Transforms 3D bounding boxes by applying the transformation to their centers,
    rotations, and linear velocities (if present). The function handles both
    the translational and rotational components of the transformation correctly.

    Args:
        transform: 4x4 homogeneous transformation matrix as numpy.ndarray.
                  Must represent a valid rigid body transformation.
        bboxes: List of bounding box dictionaries. Each box must contain:
               - "center": numpy.ndarray of shape (3,) representing box center
               - "rot_xyzw": numpy.ndarray of shape (4,) representing quaternion [x,y,z,w]
               - "linear_velocity" (optional): numpy.ndarray of shape (3,) for velocity

    Returns:
        Deep copy of input bboxes with transformed centers, rotations, and velocities.
    """
    assert type(transform) == np.ndarray and transform.shape == (4, 4)
    assert type(bboxes) == list and len(bboxes) > 0
    bboxes = copy.deepcopy(bboxes)

    transform_rotmat = copy.deepcopy(transform)
    transform_rotmat[:3, 3] = 0
    transform_scipy_rot = scipy_rot.from_matrix(transform_rotmat[:3, :3])

    for box in bboxes:
        box["center"] = transform_points(transform, box["center"].reshape([1, 3])).reshape([3])
        box["rot_xyzw"] = (scipy_rot.from_quat(box["rot_xyzw"]) * transform_scipy_rot).as_quat()

        if "linear_velocity" in box:
            box["linear_velocity"] = transform_points(transform_rotmat, box["linear_velocity"].reshape([1, 3])).reshape(
                [3]
            )

    return bboxes


def deserialize_boxes(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert serialized bounding box data to numpy arrays for computational use.

    Args:
        boxes: List of serialized bounding box dictionaries. Each box should contain:
               - "dimensions": Dict with keys "x", "y", "z" for box dimensions
               - "pose": Dict with "translation" and "rotation" subdicts
                 - "translation": Dict with keys "x", "y", "z"
                 - "rotation": Dict with keys "x", "y", "z", "w" (quaternion)
               - "velocity" (optional): Dict with keys "x", "y", "z"
               - "linear_velocity" (optional): Dict with keys "x", "y", "z"
               - "angular_velocity" (optional): Dict with keys "x", "y", "z"

    Returns:
        List of deserialized bounding boxes with numpy arrays:
        - "dimensions": numpy.ndarray of shape (3,) [x, y, z]
        - "center": numpy.ndarray of shape (3,) extracted from pose translation
        - "rot_xyzw": numpy.ndarray of shape (4,) [x, y, z, w] quaternion
        - "velocity": numpy.ndarray of shape (3,) if present
        - "linear_velocity": numpy.ndarray of shape (3,) if present
        - "angular_velocity": numpy.ndarray of shape (3,) if present
        The original "pose" dictionary is removed.
    """
    boxes = copy.deepcopy(boxes)

    for box in boxes:
        box["dimensions"] = np.array([box["dimensions"]["x"], box["dimensions"]["y"], box["dimensions"]["z"]])
        box["center"] = np.array(
            [box["pose"]["translation"]["x"], box["pose"]["translation"]["y"], box["pose"]["translation"]["z"]]
        )
        box["rot_xyzw"] = np.array(
            [
                box["pose"]["rotation"]["x"],
                box["pose"]["rotation"]["y"],
                box["pose"]["rotation"]["z"],
                box["pose"]["rotation"]["w"],
            ]
        )
        if "velocity" in box:
            box["velocity"] = np.array([box["velocity"]["x"], box["velocity"]["y"], box["velocity"]["z"]])
        if "linear_velocity" in box:
            box["linear_velocity"] = np.array(
                [box["linear_velocity"]["x"], box["linear_velocity"]["y"], box["linear_velocity"]["z"]]
            )
        if "angular_velocity" in box:
            box["angular_velocity"] = np.array(
                [box["angular_velocity"]["x"], box["angular_velocity"]["y"], box["angular_velocity"]["z"]]
            )

        del box["pose"]
    return boxes


def convert_boxes_to_rr(boxes: List[Dict[str, Any]], color_map: Optional[Dict[str, List[int]]] = None) -> rr.Boxes3D:
    """
    Convert bounding boxes to Rerun Boxes3D format for 3D visualization.
    Transforms a list of bounding boxes in AevaScenes format to a Rerun Boxes3D
    object suitable for 3D visualization. Handles color mapping for different
    object classes and provides appropriate visual styling.

    Args:
        boxes: List of deserialized bounding box dictionaries. Each box must contain:
               - "dimensions": numpy.ndarray of shape (3,) [x, y, z] full dimensions
               - "center": numpy.ndarray of shape (3,) [x, y, z] box center
               - "rot_xyzw": numpy.ndarray of shape (4,) [x, y, z, w] quaternion
               - "class" (optional): String class label for color mapping
        color_map: Optional dictionary mapping class names to RGB colors.
                  Colors should be lists of 3 integers [R, G, B] in range 0-255.
                  If None, uses default cyan color for all boxes.

    Returns:
        Rerun Boxes3D object configured for visualization
    """
    boxes_rr = {}
    boxes_rr["half_sizes"] = np.array([box["dimensions"] / 2 for box in boxes])
    boxes_rr["centers"] = np.array([box["center"] for box in boxes])
    boxes_rr["rotations"] = [Quaternion(xyzw=box["rot_xyzw"]) for box in boxes]

    if color_map is not None:
        boxes_rr["box_labels"] = [box["class"] for box in boxes]
        boxes_rr["box_colors"] = [color_map[box["class"]] for box in boxes]
    else:
        boxes_rr["box_labels"] = None
        boxes_rr["box_colors"] = [[65, 200, 225]] * len(boxes)

    boxes_rr = rr.Boxes3D(
        half_sizes=boxes_rr["half_sizes"],
        centers=boxes_rr["centers"],
        rotations=boxes_rr["rotations"],
        radii=0.05,
        colors=boxes_rr["box_colors"],
        labels=boxes_rr["box_labels"],
    )
    return boxes_rr


def convert_box_velocity_arrows_rr(
    boxes: List[Dict[str, Any]], color_map: Optional[Dict[str, List[int]]] = None
) -> rr.Arrows3D:
    """
    Convert bounding box velocity data to Rerun Arrows3D format for motion visualization.
    Creates 3D arrow vectors representing the linear velocity of bounding boxes,
    providing visual indication of object motion in the scene. Arrows originate
    from box centers and point in the direction of motion.

    Args:
        boxes: List of bounding box dictionaries. Each box must contain:
               - "center": numpy.ndarray of shape (3,) [x, y, z] box center position
               - "linear_velocity": numpy.ndarray of shape (3,) [vx, vy, vz] velocity vector
               - "class" (optional): String class label for color mapping
        color_map: Optional dictionary mapping class names to RGB colors.
                  Colors should be lists of 3 integers [R, G, B] in range 0-255.
                  If None, uses default cyan color for all arrows.

    Returns:
        Rerun Arrows3D object configured for velocity visualization with:
    """
    arrows_rr = {}
    arrows_rr["centers"] = np.array([box["center"] for box in boxes])

    linear_velocity = np.array([box["linear_velocity"] for box in boxes])
    arrows_rr["vectors"] = linear_velocity / 2  # Scale for better visualization

    if color_map is not None:
        arrows_rr["colors"] = [color_map[box["class"]] for box in boxes]
    else:
        arrows_rr["colors"] = [[65, 200, 225]] * 10

    arrows_rr = rr.Arrows3D(
        origins=arrows_rr["centers"], vectors=arrows_rr["vectors"], colors=arrows_rr["colors"], radii=0.1
    )
    return arrows_rr


def velocity_colormap(velocity: np.ndarray, clip: List[float] = [-5, 5]) -> np.ndarray:
    """
    Generate RGB colors for point cloud visualization based on velocity values.

    Creates a color mapping where:
    - Static points (velocity ≈ 0) appear gray
    - Positive velocities (approaching) appear red with intensity based on magnitude
    - Negative velocities (receding) appear blue with intensity based on magnitude

    Args:
        velocity: Velocity values as numpy.ndarray. Typically represents radial velocities in m/s.
        clip: Velocity clipping range [min_vel, max_vel] in m/s.

    Returns:
        RGB color array with shape (N, 3) where N is the number of velocity values.
        Colors are integers in range 0-255 suitable for visualization.
    """
    velocity = velocity.reshape([-1])
    static_color_list = [0.9, 0.9, 0.9]
    pos_color_list = [1, 0, 0]
    neg_color_list = [0, 0, 1]
    colors = np.full([velocity.shape[0], 3], static_color_list, dtype=np.float64)

    pos_mask = velocity > 0
    pos_norm = np.abs(velocity[pos_mask] / clip[1]).reshape([-1, 1])
    static_color = np.full([pos_norm.shape[0], 3], static_color_list, dtype=np.float64)
    pos_colors = np.full([pos_norm.shape[0], 3], pos_color_list, dtype=np.float64)
    pos_colors_interpolated = pos_norm * pos_colors + (1 - pos_norm) * static_color
    colors[pos_mask] = pos_colors_interpolated

    neg_mask = velocity < 0
    neg_norm = np.abs(velocity[neg_mask] / clip[0]).reshape([-1, 1])
    static_color = np.full([neg_norm.shape[0], 3], static_color_list, dtype=np.float64)
    neg_colors = np.full([neg_norm.shape[0], 3], neg_color_list, dtype=np.float64)
    neg_colors_interpolated = neg_norm * neg_colors + (1 - neg_norm) * static_color
    colors[neg_mask] = neg_colors_interpolated

    colors = np.clip(colors, 0, 1) * 255
    return colors.astype(np.int64)


def reflectivity_colormap(
    reflectivity: np.ndarray, clip: Optional[List[float]] = [0, 50], colormap: str = "custom_plasma"
) -> np.ndarray:
    """
    Generate RGB colors for point cloud visualization based on reflectivity values.

    Creates a plasma-like colormap for LiDAR reflectivity/intensity data, mapping
    low reflectivity values to dark purple and high values to bright yellow.
    This provides intuitive visualization of material properties and surface
    characteristics in the point cloud.

    Args:
        reflectivity: Reflectivity/intensity values as numpy.ndarray with shape (N,).
                     Typically represents LiDAR return intensity or reflectivity
                     measurements in arbitrary units or percentage.
        clip: Optional clipping range [min_val, max_val] for reflectivity values.
        colormap: Colormap style.

    Returns:
        RGB color array with shape (N, 3) where N is the number of points.
        Colors are integers in range 0-255 suitable for visualization.
    """
    if clip is not None:
        reflectivity = np.clip(reflectivity, clip[0], clip[1])

    cmap = colormaps.plasma_color_map_rgb_normalized
    reflectivity_range = np.linspace(np.min(reflectivity), np.max(reflectivity), cmap.shape[0])

    colors = np.c_[
        np.interp(reflectivity, reflectivity_range, cmap[:, 0]),
        np.interp(reflectivity, reflectivity_range, cmap[:, 1]),
        np.interp(reflectivity, reflectivity_range, cmap[:, 2]),
    ]

    colors = np.clip(colors, 0, 1) * 255
    return colors.astype(np.int64)


def velocity_reflectivity_blend_colormap(
    velocity: np.ndarray,
    reflectivity: np.ndarray,
    velocity_clip: List[float] = [0, 50],
    reflectivity_clip: List[float] = [-5, 5],
) -> np.ndarray:
    """
    Generate blended RGB colors using both velocity and reflectivity information.

    Creates a hybrid colormap that emphasizes dynamic objects through velocity
    coloring while showing static scene structure through reflectivity. This
    approach helps distinguish between moving objects (cars, pedestrians) and
    static infrastructure (buildings, road surfaces) in a single visualization.

    Color Strategy:
    - Static points (|velocity| < 3 m/s): Colored by reflectivity (grayscale-like)
    - Dynamic points (|velocity| ≥ 3 m/s): Colored by velocity (red/blue)

    Args:
        velocity: Velocity values as numpy.ndarray with shape (N,).
                 Typically Doppler radial velocities from FMCW LiDAR in m/s.
        reflectivity: Reflectivity/intensity values as numpy.ndarray with shape (N,).
        velocity_clip: Clipping range for velocity coloring [min_vel, max_vel].
        reflectivity_clip: Clipping range for reflectivity coloring [min_refl, max_refl].

    Returns:
        RGB color array with shape (N, 3) where N is the number of points.
        Colors are integers in range 0-255 suitable for visualization.
    """
    velocity_threshold = 3
    static_indices = np.where(np.abs(velocity) < velocity_threshold)[0]
    dynamic_indices = np.where(np.abs(velocity) >= velocity_threshold)[0]

    colors = np.zeros([velocity.shape[0], 3], dtype=np.int64)
    reflectivity_colors = reflectivity_colormap(reflectivity[static_indices], clip=[0, 30], colormap="gray")
    reflectivity_colors = np.clip(reflectivity_colors, 0, 170)
    colors[static_indices] = reflectivity_colors
    colors[dynamic_indices] = velocity_colormap(velocity[dynamic_indices])
    return colors.astype(np.int64)


def range_colormap(xyz: np.ndarray) -> np.ndarray:
    """
    Generate RGB colors for point cloud visualization based on distance from origin.

    Creates a rainbow colormap (turbo) where colors represent the Euclidean distance
    of each point from the coordinate system origin (0, 0, 0). This visualization
    helps understand the spatial distribution and range characteristics of the
    point cloud data.

    Args:
        xyz: 3D point coordinates as numpy.ndarray with shape (N, 3).
             Each row represents a point [x, y, z] in meters.

    Returns:
        RGB color array with shape (N, 3) where N is the number of points.
        Colors are integers in range 0-255 using the 'turbo' colormap:
    """
    range_cmap = plt.get_cmap("turbo")
    range_dist = np.linalg.norm(xyz, axis=1)
    range_normalized = range_dist / range_dist.max()
    colors = (range_cmap(range_normalized)[:, :3] * 255).astype(np.int64)
    return colors


def project_point_cloud_to_image(
    xyz: np.ndarray,
    camera_matrix: np.ndarray,
    T_S_to_V: np.ndarray,
    distortion: np.ndarray,
    image: np.ndarray,
    colors: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Project 3D LiDAR points onto a camera image with color-coded overlay.

    Transforms 3D LiDAR points to camera coordinates, projects them onto the image
    plane, and creates a blended visualization showing both the camera image and
    the projected point cloud with color coding (typically velocity or reflectivity).

    Args:
        xyz: 3D point coordinates as numpy.ndarray with shape (N, 3).
             Points in LiDAR/sensor coordinate system [x, y, z] in meters.
        camera_matrix: 3x3 camera intrinsic matrix as numpy.ndarray.
                      Contains focal lengths and principal point: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        T_S_to_V: 4x4 transformation matrix from sensor to vehicle/camera coordinates.
                  Handles both rotation and translation between coordinate systems.
        distortion: Camera distortion coefficients as numpy.ndarray.
                   Typically [k1, k2, p1, p2, k3] for radial and tangential distortion.
        image: Input camera image as numpy.ndarray with shape (H, W, 3).
               RGB image where points will be projected and overlaid.
        colors: Point colors as numpy.ndarray with shape (N, 3).
               RGB colors for each point (typically from velocity/reflectivity colormap).
               Values should be integers in range 0-255.
        alpha: Blending factor for overlay transparency (0.0 to 1.0).
               0.0 = only original image, 1.0 = only projected points.
               Default 0.7 provides good visibility of both image and points.

    Returns:
        Blended image as numpy.ndarray with same shape as input image (H, W, 3).
        Shows original camera image with color-coded projected LiDAR points overlaid.
    """
    rot_mat = T_S_to_V[:3, :3]
    trans = T_S_to_V[:3, 3]

    # Project to image plane
    uv_points = cv2.projectPoints(xyz, rot_mat, trans, camera_matrix, distortion)[0]
    uv_points = uv_points.squeeze(1)

    # Remove points outside the FoV
    mask = np.logical_and.reduce(
        (uv_points[:, 0] >= 0, uv_points[:, 0] < image.shape[1], uv_points[:, 1] >= 0, uv_points[:, 1] < image.shape[0])
    )
    uv_points = uv_points[mask].astype("int16")
    colors = colors[mask]

    # Draw circles for points
    xyz_image = np.zeros_like(image)
    for p_id in range(uv_points.shape[0]):
        xyz_image = cv2.circle(xyz_image, tuple(uv_points[p_id]), radius=1, color=colors[p_id].tolist(), thickness=-1)

    # Blend projected points with input image
    blended_image = copy.deepcopy(image)
    blended_image[xyz_image > 0] = alpha * xyz_image[xyz_image > 0] + (1 - alpha) * image[xyz_image > 0]

    return blended_image


def get_local_ip() -> str:
    """
    Retrieve the local IP address of the current machine..

    Returns:
        Local IP address as a string (e.g., "192.168.1.100").
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable — just used to get the local IP
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def pandas_map(input_array: np.ndarray, mapping: Dict[Any, Any]) -> np.ndarray:
    """
    Apply a mapping dictionary to a numpy array using pandas for efficient lookup.

    Args:
        input_array: Input numpy array of any shape containing values to be mapped.
        mapping: Dictionary defining the mapping from input values to output values.

    Returns:
        Numpy array with same shape as input_array, containing mapped values.
    """
    return pd.Series(input_array.ravel()).map(mapping).values.reshape(input_array.shape)
