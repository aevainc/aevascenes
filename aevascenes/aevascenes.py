# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# The AevaScenes dataset is licensed separately under the AevaScenes Dataset License.
# See https://scenes.aeva.com/license for the full license text.

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rerun as rr
from PIL import Image
from tqdm import tqdm

random.seed(234)

from tabulate import tabulate

import aevascenes.configs as configs
from aevascenes import utils
from aevascenes.visualizer import RRVisualizer


class AevaScenes:
    """
    Main interface for AevaScenes dataset access and visualization.

    This class provides comprehensive functionality for loading, processing,
    and visualizing multi-modal sensor data from the AevaScenes dataset.
    It handles sequence point-clouds, images, semantic labels and 3D bounding box
    visualization through an integrated Rerun visualizer.
    """

    def __init__(self, dataroot: str) -> None:
        """
        Initialize AevaScenes dataset interface.

        Args:
            dataroot: Path to the root directory of the AevaScenes dataset
        """
        self.dataroot = dataroot
        self.rr_visualizer = RRVisualizer()
        metadata_path = os.path.join(self.dataroot, "metadata.json")
        if os.path.exists(metadata_path):
            self.metadata = utils.read_file()
        else:
            print(
                f"Skipping reading metadata.json as it doesn't exist at {metadata_path}. \
                Will not be able to load full dataset automatically."
            )

    def is_sequence_uuid_valid(self, sequence_uuid: str) -> bool:
        """
        Validate if a sequence UUID exists in the dataset.

        Args:
            sequence_uuid: UUID string to validate

        Returns:
            True if the UUID is valid and exists in the dataset, False otherwise
        """
        if type(sequence_uuid) != str or len(sequence_uuid) != 36:
            return False
        return sequence_uuid in self.metadata["sequence_uuids"]

    def get_sequence_uuids(self) -> List[str]:
        """
        Get list of all sequence UUIDs in the dataset.

        Returns:
            List of sequence UUID strings available in the dataset
        """
        return self.metadata["sequence_uuids"]

    def list_sequences(self) -> None:
        """
        Print a formatted table of all sequence UUIDs in the dataset.

        Displays dataset version information, total sequence count, and
        a numbered list of all available sequence UUIDs in a grid format.
        """
        print(
            f"Listing sequence_uuid's in {self.metadata['version']} - number of sequences: {self.metadata['num_sequences']}"
        )

        sequence_uuids = self.get_sequence_uuids()
        data = [[i + 1, seq] for i, seq in enumerate(sequence_uuids)]
        print(tabulate(data, headers=["No.", "Sequence UUID"], tablefmt="grid"))

    def load_sequence(self, sequence_uuid: str) -> Dict[str, Any]:
        """
        Load complete sequence data for a given UUID.

        Args:
            sequence_uuid: UUID of the sequence to load

        Returns:
            Dictionary containing sequence metadata and frame data
        """
        sequence_data_path = os.path.join(self.dataroot, sequence_uuid, "sequence.json")
        return utils.read_file(sequence_data_path)

    def get_pcd_colors_labels(
        self,
        xyz: np.ndarray,
        pcd_color_mode: str,
        velocity: Optional[np.ndarray] = None,
        reflectivity: Optional[np.ndarray] = None,
        semantic_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate colors and labels for point cloud visualization based on the specified mode.

        Args:
            xyz: Point cloud coordinates array of shape (N, 3)
            pcd_color_mode: Coloring mode - "velocity", "reflectivity", or "semantic"
            velocity: Velocity values for each point, required if pcd_color_mode is "velocity"
            reflectivity: Reflectivity values for each point, required if pcd_color_mode is "reflectivity"
            semantic_labels: Semantic class labels for each point, required if pcd_color_mode is "semantic"

        Returns:
            Tuple containing:
                - RGB color array of shape (N, 3) with values in [0, 255]
                - List of string labels corresponding to each point
        """
        if pcd_color_mode == "velocity":
            colors = utils.velocity_colormap(velocity)
            labels = velocity.reshape([-1]).astype(np.str_).tolist()
        if pcd_color_mode == "reflectivity":
            colors = utils.reflectivity_colormap(reflectivity, clip=[0, 50])
            labels = reflectivity.reshape([-1]).astype(np.str_).tolist()
        if pcd_color_mode == "semantic":
            colors = np.array([configs.class_color_map[label] for label in semantic_labels.flatten()])
            labels = semantic_labels.reshape([-1]).astype(np.str_).tolist()

        return colors, labels

    def visualize_frame(
        self,
        frame: Dict[str, Any],
        sequence_metadata: Dict[str, Any],
        pcd_color_mode: str,
        project_points_on_image: bool = False,
        image_downsample_factor: int = 1,
    ) -> None:
        """
        Visualize a single frame with all sensor data including LiDAR, cameras, and bounding boxes.

        This method processes and visualizes all available sensor data for a single frame,
        including point cloud data from multiple LiDAR sensors, camera images with optional
        point cloud projection, and 3D bounding boxes with velocity arrows.

        Args:
            frame: Frame data dictionary containing timestamps, poses, and sensor data paths
            sequence_metadata: Sequence metadata containing sensor configurations and extrinsics
            pcd_color_mode: Point cloud coloring mode - "velocity", "reflectivity", or "semantic"
            project_points_on_image: Whether to project LiDAR points onto camera images
            image_downsample_factor: Factor to downsample images for faster rendering (1 = no downsampling)
        """
        # Load frame metadata
        timestamp_ns = frame["timestamp_ns"]
        frame_uuid = frame["frame_uuid"]
        ego_pose = frame["ego_pose"]

        # Load lidar metadata
        sequence_uuid = sequence_metadata["sequence_uuid"]

        lidar_ids = sequence_metadata["sensors"]["lidars"]
        vehicle_to_lidar_extrinsics = sequence_metadata["vehicle_to_lidar_extrinsics"]

        # Load lidar data
        pcds = []
        pointcloud_cache = {}
        for lidar_id in lidar_ids:
            rel_path = frame["point_cloud"][lidar_id]["point_cloud_path"]  # remove this hack
            pointcloud = np.load(os.path.join(self.dataroot, sequence_uuid, rel_path), allow_pickle=True)
            pointcloud_cache[lidar_id] = pointcloud

            xyz = pointcloud["xyz"]
            transform_vehicle_to_lidar = utils.pose_to_matrix(
                sequence_metadata["vehicle_to_lidar_extrinsics"][lidar_id]
            )
            xyz = utils.transform_points(transform_vehicle_to_lidar, xyz)

            reflectivity = pointcloud["reflectivity"]
            velocity = pointcloud["velocity"]
            line_index = pointcloud["line_index"]
            time_offset_ns = pointcloud["time_offset_ns"]

            # Load semantic label strings and indices
            # string to idx mapping shown in docs/dataset.md
            semantic_labels = pointcloud["semantic_labels"]
            semantic_labels_idx = pointcloud["semantic_labels_idx"]

            colors, labels = self.get_pcd_colors_labels(
                xyz=xyz,
                pcd_color_mode=pcd_color_mode,
                velocity=velocity,
                reflectivity=reflectivity,
                semantic_labels=semantic_labels,
            )

            pcds.append({"lidar_id": lidar_id, "pcd": rr.Points3D(xyz, colors=colors, labels=labels)})

        # Load camera metadata
        camera_ids = sequence_metadata["sensors"]["cameras"]
        vehicle_to_camera_extrinsics = sequence_metadata["vehicle_to_camera_extrinsics"]
        camera_intrinsics = sequence_metadata["camera_intrinsics"]

        # Load camera data
        images = []
        for camera_id in camera_ids:
            # Load image
            rel_path = frame["image"][camera_id]["image_path"]
            image = np.array(Image.open(os.path.join(self.dataroot, sequence_uuid, rel_path)))

            # Load camera parameters
            camera_matrix = np.array(sequence_metadata["camera_intrinsics"][camera_id]["matrix"]).reshape([3, 3])
            dist_coeffs = np.array(
                sequence_metadata["camera_intrinsics"][camera_id]["distortion_coefficients"]
            ).reshape([1, 5])

            if project_points_on_image:
                lidar_id_for_camera = sequence_metadata["camera_lidar_mapping"][camera_id]
                pointcloud = pointcloud_cache[lidar_id_for_camera]

                transform_vehicle_to_lidar = utils.pose_to_matrix(
                    sequence_metadata["vehicle_to_lidar_extrinsics"][lidar_id_for_camera]
                )
                xyz = utils.transform_points(transform_vehicle_to_lidar, pointcloud["xyz"])

                transform_vehicle_to_camera = utils.pose_to_matrix(
                    sequence_metadata["vehicle_to_camera_extrinsics"][camera_id]
                )
                transform_camera_to_vehicle = np.linalg.inv(transform_vehicle_to_camera)

                colors, _ = self.get_pcd_colors_labels(
                    xyz=xyz,
                    pcd_color_mode="reflectivity",
                    velocity=pointcloud["velocity"],
                    reflectivity=pointcloud["reflectivity"],
                    semantic_labels=pointcloud["semantic_labels"],
                )
                image = utils.project_point_cloud_to_image(
                    xyz, camera_matrix, transform_camera_to_vehicle, dist_coeffs, image, colors, alpha=0.7
                )

            # Undistort raw image
            undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

            # Downsample 4k images for faster viewing
            undistorted_image_resized = cv2.resize(
                undistorted_image,
                (
                    undistorted_image.shape[1] // image_downsample_factor,
                    undistorted_image.shape[0] // image_downsample_factor,
                ),
                interpolation=cv2.INTER_LINEAR,
            )
            images.append({"camera_id": camera_id, "image": rr.Image(undistorted_image_resized)})

        # Load object detection boxes
        boxes_serialized = frame["boxes"]
        boxes = utils.deserialize_boxes(boxes_serialized)
        boxes_rr = utils.convert_boxes_to_rr(boxes, color_map=configs.class_color_map)
        arrows_rr = utils.convert_box_velocity_arrows_rr(boxes)

        # Send to rerun visualizer
        self.rr_visualizer.add_data(pcds=pcds, images=images, boxes=boxes_rr, arrows=arrows_rr)

    def visualize_sampled_frames_from_dataset(
        self, pcd_color_mode: str = "velocity", project_points_on_image: bool = False, image_downsample_factor: int = 1
    ) -> None:
        """
        Visualize randomly sampled frames from all sequences in the dataset.

        This method iterates through all sequences in the dataset, randomly selects
        one frame from each sequence, and visualizes it. Useful for getting an
        overview of the dataset diversity and content.

        Args:
            pcd_color_mode: Point cloud coloring mode - "velocity", "reflectivity", or "semantic"
            project_points_on_image: Whether to project LiDAR points onto camera images
            image_downsample_factor: Factor to downsample images for faster rendering
        """
        sequence_uuids = self.get_sequence_uuids()
        random.shuffle(sequence_uuids)

        for sequence_idx in tqdm(range(len(sequence_uuids)), desc="Visualizing sampled frames"):
            sequence_data = self.load_sequence(sequence_uuids[sequence_idx])
            frames = sequence_data["frames"]

            sampled_frame_id = random.randint(0, len(frames) - 1)

            self.rr_visualizer.set_frame_counter(frame_idx=sequence_idx)
            self.visualize_frame(
                frames[sampled_frame_id],
                sequence_data["metadata"],
                pcd_color_mode=pcd_color_mode,
                project_points_on_image=project_points_on_image,
                image_downsample_factor=image_downsample_factor,
            )

    def visualize_sequence(
        self,
        sequence_uuid: str,
        pcd_color_mode: str = "velocity",
        project_points_on_image: bool = False,
        image_downsample_factor: int = 1,
    ) -> None:
        """
        Visualize all frames in a specific sequence in chronological order.

        Args:
            sequence_uuid: UUID of the sequence to visualize
            pcd_color_mode: Point cloud coloring mode - "velocity", "reflectivity", or "semantic"
            project_points_on_image: Whether to project LiDAR points onto camera images
            image_downsample_factor: Factor to downsample images for faster rendering
        """
        sequence_data = self.load_sequence(sequence_uuid)
        frames = sequence_data["frames"]

        for frame_idx in tqdm(range(len(frames)), desc="Visualizing frames"):
            self.rr_visualizer.set_frame_counter(frame_idx=frame_idx, timestamp_ns=frames[frame_idx]["timestamp_ns"])
            self.visualize_frame(
                frames[frame_idx],
                sequence_data["metadata"],
                pcd_color_mode=pcd_color_mode,
                project_points_on_image=project_points_on_image,
                image_downsample_factor=image_downsample_factor,
            )
