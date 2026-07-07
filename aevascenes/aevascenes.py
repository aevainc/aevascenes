# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# The AevaScenes dataset is licensed separately under the AevaScenes Dataset License.
# See https://scenes.aeva.com/license for the full license text.

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from tabulate import tabulate

import aevascenes.configs as configs
from aevascenes import utils

_METADATA = os.path.join(os.path.dirname(__file__), "metadata", "aevascenes_v2_metadata.json")
_SPLITS = ("train", "validation", "test")


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
        self.rr_visualizer = None

    def get_sequence_uuids(self) -> List[str]:
        """
        Get list of all sequence UUIDs in the dataset.

        Returns:
            List of sequence UUID strings available in the dataset
        """
        metadata = utils.read_file(_METADATA)
        split = os.path.basename(os.path.normpath(self.dataroot))
        if split in metadata["splits"]:
            return metadata["splits"][split]
        return [uuid for split in metadata["splits"].values() for uuid in split]

    def _dataroot_split(self) -> Optional[str]:
        split = os.path.basename(os.path.normpath(self.dataroot))
        return split if split in _SPLITS else None

    def _sequence_json_path(self, sequence_uuid: str, split: Optional[str] = None) -> str:
        base = self.dataroot if split is None else os.path.join(self.dataroot, split)
        return os.path.join(base, sequence_uuid, "sequence.json")

    def _split_dataroot(self, split: str) -> str:
        if self._dataroot_split() is None:
            return os.path.join(self.dataroot, split)
        return os.path.join(os.path.dirname(os.path.normpath(self.dataroot)), split)

    def _find_on_disk_split(self, sequence_uuid: str) -> Optional[str]:
        if os.path.exists(self._sequence_json_path(sequence_uuid)):
            return self._dataroot_split() or "."
        parent = self.dataroot if self._dataroot_split() is None else os.path.dirname(os.path.normpath(self.dataroot))
        for split in _SPLITS:
            if os.path.exists(os.path.join(parent, split, sequence_uuid, "sequence.json")):
                return split
        return None

    def _metadata_split(self, sequence_uuid: str) -> Optional[str]:
        for split, uuids in utils.read_file(_METADATA)["splits"].items():
            if sequence_uuid in uuids:
                return split
        return None

    def explain_invalid_sequence_uuid(self, sequence_uuid: str) -> Optional[str]:
        """Return an error message if the UUID is invalid, None if it can be loaded from dataroot."""
        if type(sequence_uuid) != str or len(sequence_uuid) != 36:
            return f"Invalid sequence UUID format '{sequence_uuid}' (expected a 36-character UUID)."

        if os.path.exists(self._sequence_json_path(sequence_uuid)):
            return None

        on_disk = self._find_on_disk_split(sequence_uuid)
        meta_split = self._metadata_split(sequence_uuid)
        dataroot_split = self._dataroot_split()

        if on_disk is not None and on_disk != ".":
            suggested = self._split_dataroot(on_disk)
            if dataroot_split is None:
                return (
                    f"Sequence '{sequence_uuid}' is in the '{on_disk}' split but --dataroot is "
                    f"'{self.dataroot}' (dataset root, not a split directory).\n"
                    f"  Try: --dataroot {suggested}"
                )
            return (
                f"Sequence '{sequence_uuid}' is in the '{on_disk}' split but --dataroot is "
                f"'{self.dataroot}' ({dataroot_split} split).\n"
                f"  Try: --dataroot {suggested}"
            )

        if meta_split:
            suggested = self._split_dataroot(meta_split)
            hint = f"\n  Expected under: {suggested}/" if dataroot_split != meta_split else ""
            return (
                f"Sequence '{sequence_uuid}' ({meta_split} split) is not present under '{self.dataroot}'. "
                f"Download and extract it first.{hint}"
            )

        return f"Unknown sequence UUID '{sequence_uuid}'. Use --list-sequences to see available sequences."

    def is_sequence_uuid_valid(self, sequence_uuid: str) -> bool:
        """
        Validate if a sequence UUID exists in the dataset.

        Args:
            sequence_uuid: UUID string to validate

        Returns:
            True if the UUID is valid and exists in the dataset, False otherwise
        """
        return self.explain_invalid_sequence_uuid(sequence_uuid) is None

    def list_sequences(self) -> None:
        """
        Print a formatted table of all sequence UUIDs in the dataset.

        Displays dataset version information, total sequence count, and
        a numbered list of all available sequence UUIDs in a grid format.
        """
        metadata = utils.read_file(_METADATA)
        split = os.path.basename(os.path.normpath(self.dataroot))
        if split in metadata["splits"]:
            sequence_uuids = metadata["splits"][split]
        else:
            sequence_uuids = [uuid for split in metadata["splits"].values() for uuid in split]
        name = metadata["dataset_info"]["name"]
        print(f"Listing sequence_uuid's in {name} - number of sequences: {len(sequence_uuids)}")
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

    def init_visualizer(
        self,
        name: str = "AevaScenes-Visualizer",
        web_port: int = 9590,
        grpc_port: int = 9591,
        pcd_type: str = "compensated",
        show_images: bool = True,
        coordinate_frame: str = "vehicle",
        world_look_target: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Initialize the Rerun visualizer. Call before visualize_* when using visualize mode."""
        from aevascenes.visualizer import RRVisualizer

        if RRVisualizer is None:
            raise ImportError("rerun-sdk is required for visualization. Install with: pip install rerun-sdk==0.33.0")
        self.rr_visualizer = RRVisualizer(
            name=name,
            web_port=web_port,
            grpc_port=grpc_port,
            pcd_types=pcd_type,
            show_images=show_images,
            coordinate_frame=coordinate_frame,
            world_look_target=world_look_target,
        )
        self.rr_visualizer.initialize()

    def _vehicle_to_world_transform(self, frame: Dict[str, Any]) -> np.ndarray:
        """Map vehicle-frame geometry into the sequence world frame via ego_pose."""
        return utils.pose_to_matrix(frame["ego_pose"])

    def _apply_world_transform(
        self, pcds: List[Dict[str, Any]], boxes: List[Dict[str, Any]], frame: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        transform = self._vehicle_to_world_transform(frame)
        for pcd in pcds:
            xyz = utils.transform_points(transform, pcd["xyz"])
            pcd["xyz"] = xyz
            try:
                import rerun as rr

                pcd["pcd"] = rr.Points3D(xyz, colors=pcd["colors"], labels=pcd["labels"])
            except ImportError:
                pass
        boxes = utils.transform_bboxes(transform, boxes)
        if self.rr_visualizer is not None:
            self.rr_visualizer.log_ego_pose(transform)
        return pcds, boxes

    def _world_look_target(self, frames: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        origins = np.array([utils.pose_to_matrix(frame["ego_pose"])[:3, 3] for frame in frames])
        return tuple(origins.mean(axis=0).tolist())

    def _load_pointcloud(
        self,
        rel_path: str,
        sequence_uuid: str,
        lidar_id: str,
        sequence_metadata: Dict[str, Any],
        pcd_color_mode: str,
        pcd_type: str,
    ) -> Dict[str, Any]:
        pointcloud = np.load(os.path.join(self.dataroot, sequence_uuid, rel_path), allow_pickle=True)
        xyz = pointcloud["xyz"]
        transform_vehicle_to_lidar = utils.pose_to_matrix(sequence_metadata["vehicle_to_lidar_extrinsics"][lidar_id])
        xyz = utils.transform_points(transform_vehicle_to_lidar, xyz)

        colors, labels = self.get_pcd_colors_labels(
            xyz=xyz,
            pcd_color_mode=pcd_color_mode,
            velocity=pointcloud["velocity"],
            reflectivity=pointcloud["reflectivity"],
            semantic_labels=pointcloud["semantic_labels"],
        )

        lidar_display_id = lidar_id
        pcd_data = {"lidar_id": lidar_display_id, "pcd_type": pcd_type, "xyz": xyz, "colors": colors, "labels": labels}
        try:
            import rerun as rr

            pcd_data["pcd"] = rr.Points3D(xyz, colors=colors, labels=labels)
        except ImportError:
            pass
        return {"pcd_data": pcd_data, "pointcloud": pointcloud}

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
        elif pcd_color_mode == "reflectivity":
            colors = utils.reflectivity_colormap(reflectivity, clip=[0, 50])
            labels = reflectivity.reshape([-1]).astype(np.str_).tolist()
        elif pcd_color_mode == "semantic":
            colors = np.array([configs.class_color_map[label] for label in semantic_labels.flatten()])
            labels = semantic_labels.reshape([-1]).astype(np.str_).tolist()
        else:
            raise ValueError(f"Unknown pcd_color_mode '{pcd_color_mode}' (expected velocity, reflectivity, or semantic)")

        return colors, labels

    def visualize_frame(
        self,
        frame: Dict[str, Any],
        sequence_metadata: Dict[str, Any],
        pcd_color_mode: str,
        pcd_type: str = "compensated",
        project_points_on_image: bool = False,
        image_downsample_factor: int = 1,
        include_images: bool = True,
        coordinate_frame: str = "vehicle",
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
            pcd_type: Point cloud variant - "compensated", "raw", or "raw_and_compensated"
            project_points_on_image: Whether to project LiDAR points onto camera images
            image_downsample_factor: Factor to downsample images for faster rendering (1 = no downsampling)
            include_images: Whether to load and display camera images
            coordinate_frame: "vehicle" for ego frame or "world" using frame ego_pose
        """
        both = pcd_type == "raw_and_compensated"
        types = ["compensated", "raw"] if both else [pcd_type]
        sequence_uuid = sequence_metadata["sequence_uuid"]
        lidar_ids = sequence_metadata["sensors"]["lidars"]

        pcds = []
        pointcloud_cache = {}
        for lidar_id in lidar_ids:
            pcd_entry = frame["point_cloud"][lidar_id]
            for t in types:
                key = "point_cloud_compensated_path" if t == "compensated" else "point_cloud_raw_path"
                loaded = self._load_pointcloud(
                    pcd_entry[key], sequence_uuid, lidar_id, sequence_metadata, pcd_color_mode, t
                )
                cache_key = (lidar_id, t) if both else lidar_id
                pointcloud_cache[cache_key] = loaded["pointcloud"]
                pcds.append(loaded["pcd_data"])

        # Load camera data
        images = []
        if include_images:
            camera_ids = sequence_metadata["sensors"]["cameras"]
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
                    projection_type = "compensated" if pcd_type != "raw" else "raw"
                    cache_key = (lidar_id_for_camera, projection_type) if both else lidar_id_for_camera
                    pointcloud = pointcloud_cache[cache_key]

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
                img_data = {"camera_id": camera_id, "image_np": undistorted_image_resized}
                try:
                    import rerun as rr
                    img_data["image"] = rr.Image(undistorted_image_resized)
                except ImportError:
                    pass
                images.append(img_data)

        # Load object detection boxes
        boxes_serialized = frame["boxes"]
        boxes = utils.deserialize_boxes(boxes_serialized)

        if coordinate_frame == "world":
            pcds, boxes = self._apply_world_transform(pcds, boxes, frame)

        # Send to rerun visualizer (if available)
        if self.rr_visualizer is not None:
            boxes_rr = utils.convert_boxes_to_rr(boxes, color_map=configs.class_color_map)
            arrows_rr = utils.convert_box_velocity_arrows_rr(boxes)
            self.rr_visualizer.add_data(
                pcds=pcds, images=images if include_images else None, boxes=boxes_rr, arrows=arrows_rr
            )

        return {"pcds": pcds, "images": images, "boxes": boxes}

    def visualize_sequence(
        self,
        sequence_uuid: str,
        pcd_color_mode: str = "velocity",
        pcd_type: str = "compensated",
        project_points_on_image: bool = False,
        image_downsample_factor: int = 1,
        include_images: bool = True,
        coordinate_frame: str = "vehicle",
        init_visualizer: bool = False,
        keep_alive: bool = True,
    ) -> None:
        """
        Visualize all frames in a specific sequence in chronological order.

        Args:
            sequence_uuid: UUID of the sequence to visualize
            pcd_color_mode: Point cloud coloring mode - "velocity", "reflectivity", or "semantic"
            pcd_type: Point cloud variant - "compensated", "raw", or "raw_and_compensated"
            project_points_on_image: Whether to project LiDAR points onto camera images
            image_downsample_factor: Factor to downsample images for faster rendering
            include_images: Whether to load and display camera images
            coordinate_frame: "vehicle" for ego frame or "world" using frame ego_pose
            init_visualizer: Initialize Rerun visualizer if not already initialized
            keep_alive: Keep gRPC/web servers running after streaming (until Ctrl+C)
        """
        if init_visualizer and self.rr_visualizer is None:
            sequence_data = self.load_sequence(sequence_uuid)
            frames = sequence_data["frames"]
            world_look_target = self._world_look_target(frames) if coordinate_frame == "world" else None
            if coordinate_frame == "world":
                ego_x = [utils.pose_to_matrix(frame["ego_pose"])[0, 3] for frame in frames]
                print(
                    f"World frame: ego origin X {ego_x[0]:.1f} -> {ego_x[-1]:.1f} m "
                    f"(delta {ego_x[-1] - ego_x[0]:+.1f} m along world +X)"
                )
            self.init_visualizer(
                pcd_type=pcd_type,
                show_images=include_images,
                coordinate_frame=coordinate_frame,
                world_look_target=world_look_target,
            )
            if coordinate_frame == "world":
                self.rr_visualizer.log_ego_trajectory(frames)
        else:
            sequence_data = self.load_sequence(sequence_uuid)
            frames = sequence_data["frames"]

        for frame_idx in tqdm(range(len(frames)), desc="Visualizing frames"):
            if self.rr_visualizer is not None:
                self.rr_visualizer.set_frame_counter(frame_idx=frame_idx, timestamp_ns=frames[frame_idx]["timestamp_ns"])
            self.visualize_frame(
                frames[frame_idx],
                sequence_data["metadata"],
                pcd_color_mode=pcd_color_mode,
                pcd_type=pcd_type,
                project_points_on_image=project_points_on_image,
                image_downsample_factor=image_downsample_factor,
                include_images=include_images,
                coordinate_frame=coordinate_frame,
            )

        if keep_alive and self.rr_visualizer is not None:
            self.rr_visualizer.keep_alive()
