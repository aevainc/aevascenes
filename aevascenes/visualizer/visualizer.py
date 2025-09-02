# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# The AevaScenes dataset is licensed separately under the AevaScenes Dataset License.
# See https://scenes.aeva.com/license for the full license text.

from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import quote

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rich import print

import aevascenes.utils as aeva_utils


class RRVisualizer:
    """
    Rerun-based visualizer for AevaScenes multi-modal sensor data.

    This class provides a comprehensive visualization interface for AevaScenes dataset
    components including LiDAR point clouds, camera images, 3D bounding boxes,
    flow vectors, and other sensor data. It creates an interactive 3D environment
    accessible through either the Rerun desktop application or web browser.

    The visualizer automatically handles:
    - Multi-camera layouts with synchronized timestamps
    - 3D LiDAR point cloud rendering with ground plane
    - Real-time data streaming and namespace management
    - Interactive controls for temporal navigation
    """

    def __init__(
        self,
        name: str = "AevaScenes-Visualizer",
        web_port: int = 9590,
        grpc_port: int = 9591,
    ) -> None:
        """
        Initialize the AevaScenes Rerun visualizer.

        Sets up the Rerun visualization environment with a default layout optimized
        for AevaScenes data. Creates both gRPC server and web viewer interfaces,
        configures the ground plane grid, and displays connection instructions.

        Args:
            name: Display name for the Rerun application instance
            web_port: TCP port for the web viewer interface
            grpc_port: TCP port for the gRPC server
        """
        # Initialize ground grid configuration with sensible defaults
        # Default ground grid config (meters, colors in RGB 0–1)
        self.ground_config = {
            "xmin": 0,  # Minimum X coordinate (meters)
            "xmax": 500,  # Maximum X coordinate (meters)
            "ymin": -250,  # Minimum Y coordinate (meters)
            "ymax": 250,  # Maximum Y coordinate (meters)
            "z": 0,  # Z-coordinate for grid lines
            "radii": [0.05, 0.05],  # Line thickness [start, end]
            "resolution": 100,  # Grid spacing (meters)
            "color": [1, 0.95, 0.5],  # RGB color (yellow)
            "ground_height": 0,  # Ground plane Z-offset
        }

        # UI and connection configuration
        background_color = [0, 0, 0]  # Black background for 3D views
        self.web_port = web_port
        self.grpc_port = grpc_port
        self.curr_frame_idx = 0  # Track current frame for auto-increment

        # Create optimized layout for AevaScenes multi-modal data
        # Default layout: one 3D LiDAR view + 6 camera tiles
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                # Primary 3D visualization for LiDAR and spatial data
                rrb.Spatial3DView(origin="/lidar", name="lidar", background=background_color),
                # Secondary 2D grid for all camera feeds
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="/front_wide_camera", name="front_wide_camera"),
                    rrb.Spatial2DView(origin="/front_narrow_camera", name="front_narrow_camera"),
                    rrb.Spatial2DView(origin="/right_camera", name="right_camera"),
                    rrb.Spatial2DView(origin="/rear_wide_camera", name="rear_wide_camera"),
                    rrb.Spatial2DView(origin="/rear_narrow_camera", name="rear_narrow_camera"),
                    rrb.Spatial2DView(origin="/left_camera", name="left_camera"),
                    column_shares=[1, 1, 1, 1, 1, 1],  # Equal width for all cameras
                ),
                row_shares=[7.5, 2.5],  # 75% lidar view, 25% camera grid
            ),
            # Keep side panels collapsed for a clean default UI
            rrb.TimePanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.BlueprintPanel(state="collapsed"),
        )

        # Initialize Rerun server and viewer interfaces
        # Start a local Rerun gRPC server and expose a web viewer
        rr.init(name)
        server_uri = rr.serve_grpc(grpc_port=self.grpc_port, default_blueprint=blueprint)
        rr.serve_web_viewer(connect_to=server_uri, web_port=self.web_port)

        # Initialize namespace tracking for automatic cleanup
        # Track per-stream namespaces so we can blank unused ones each frame
        self.stream_names = ["pcds", "boxes", "images", "flow", "text"]
        self.stream_namespaces = {name: set() for name in self.stream_names}
        self.empty_messages = {
            "pcds": rr.Points3D(positions=None),
            "boxes": rr.Boxes3D(),
            "images": rr.Image(image=np.empty([1, 1])),
            "flow": rr.Arrows3D(vectors=None),
            "text": "",
        }

        # Generate user-friendly connection instructions
        # Build convenient local URLs/commands for users
        ip_address = aeva_utils.get_local_ip()
        url = f"rerun+http://{ip_address}:{grpc_port}/proxy"
        web_viewer_link = f"http://{ip_address}:{web_port}/?url={quote(url)}"
        app_viewer_link = f"rerun --connect {url}"

        # Quick-start message
        message = (
            f"[green]\nWe support visualizing the dataset from both a remote server and through a web interface.\n\n"
            f"[green]1) After running visualize_sequences.py, you can launch a local rerun viewer in the terminal of "
            + "your local machine by running:\n\n"
            f"[green]\t[bright_cyan]{app_viewer_link}[/bright_cyan]\n\n"
            f"\t\t\t\t[green]OR[/green]\n\n"
            f"[green]2) You can also launch the AevaScenes web visualizer by opening the following link in a "
            + "browser[/green].\n"
            f"   This may have stability issues based on browser memory availability\n\n"
            f"[green]\t[bright_cyan]{web_viewer_link}[/bright_cyan]\n\n"
            f"[green]Please ensure that you have rerun installed on both remote machine/local python environments."
            f"[/green]\n"
            f"[green]To install rerun: pip install rerun-sdk==0.24.1[/green]"
        )
        print(message)

    def initialize(self) -> None:
        """
        Perform one-time scene setup and initialization.

        This method should be called once after creating the visualizer instance
        to set up static scene elements like the ground plane grid. It logs
        static entities that persist across all frames.
        """
        self.log_ground_line_strip()

    def set_time_sequence(self, name: str, value: int) -> None:
        """
        Set a monotonically increasing sequence clock for temporal navigation.

        Args:
            name: Identifier for this time sequence for e.g., "frame_idx"
            value: Integer sequence value (should increase monotonically)
        """
        rr.set_time_sequence(name, value)

    def set_time_nanos(self, name: str, value: int) -> None:
        """
        Set a nanosecond-precision time axis for high-accuracy synchronization.

        Args:
            name: Identifier for this time axis (e.g., "time", "lidar_timestamp")
            value: Timestamp in nanoseconds (typically from sensor data)
        """
        # NOTE: Consider rr.set_time_nanos here; current call mirrors set_time_sequence.
        rr.set_time_sequence(name, value)

    def log_ground_line_strip(self) -> None:
        """
        Draw a static 3D grid representing the ground plane.

        Creates a grid of 3D line strips based on the ground_config parameters.
        The grid helps provide spatial reference and scale in the 3D visualization.
        Lines are drawn at regular intervals in both X and Y directions at the
        specified ground height.
        """
        lines = []
        line_colors = []
        x_num_lines = (
            int(abs(self.ground_config["xmax"] - self.ground_config["xmin"]) / self.ground_config["resolution"]) + 1
        )
        y_num_lines = (
            int(abs(self.ground_config["ymax"] - self.ground_config["ymin"]) / self.ground_config["resolution"]) + 1
        )

        # Vertical grid lines (varying x)
        for i in range(x_num_lines):
            lines.append(
                [
                    [
                        self.ground_config["xmin"] + (self.ground_config["resolution"] * i),
                        self.ground_config["ymin"],
                        self.ground_config["ground_height"],
                    ],
                    [
                        self.ground_config["xmin"] + (self.ground_config["resolution"] * i),
                        self.ground_config["ymax"],
                        self.ground_config["ground_height"],
                    ],
                ]
            )
            line_colors.append(self.ground_config["color"])

        # Horizontal grid lines (varying y)
        for i in range(y_num_lines):
            lines.append(
                [
                    [
                        self.ground_config["xmin"],
                        self.ground_config["ymin"] + (self.ground_config["resolution"] * i),
                        self.ground_config["ground_height"],
                    ],
                    [
                        self.ground_config["xmax"],
                        self.ground_config["ymin"] + (self.ground_config["resolution"] * i),
                        self.ground_config["ground_height"],
                    ],
                ]
            )
            line_colors.append(self.ground_config["color"])

        rr.log(
            "lidar/ground_plane",
            rr.LineStrips3D(lines, colors=line_colors, radii=self.ground_config["radii"]),
            static=True,  # never changes across frames
        )

    def set_frame_counter(self, frame_idx: Optional[int] = None, timestamp_ns: Optional[int] = None) -> None:
        """
        Advance sequence and time clocks and update current frame tracking.

        This method synchronizes the visualization timeline by setting both
        discrete frame indices and continuous nanosecond timestamps. It's
        typically called at the beginning of each frame update to establish
        the temporal context for all subsequent data logging.

        Args:
            frame_idx: Discrete frame index. If None, auto-increments from current frame.
                Should be monotonically increasing for proper timeline navigation.
            timestamp_ns: Nanosecond timestamp for precise time synchronization.
                If None, only the frame sequence is updated.
        """
        if frame_idx is not None:
            rr.set_time_sequence("frame_idx", frame_idx)
            self.curr_frame_idx = frame_idx
        else:
            rr.set_time_sequence("frame_idx", self.curr_frame_idx)
            self.curr_frame_idx += 1

        if timestamp_ns is not None:
            rr.set_time_nanos("time", timestamp_ns)

    def clear_empty_namespaces(self, curr_stream_set: Dict[str, Set[str]]) -> None:
        """
        Clear stale data by logging empty entities to unused namespaces.

        Args:
            curr_stream_set: Dictionary mapping stream types to sets of active
                namespace strings for the current frame. Should contain entries
                for all stream types in self.stream_names.
        """
        for stream_type in self.stream_namespaces:
            for namespace in self.stream_namespaces[stream_type]:
                if namespace not in curr_stream_set[stream_type]:
                    rr.log(namespace, self.empty_messages[stream_type])
            # Track namespaces seen so far (grow-only)
            self.stream_namespaces[stream_type].update(curr_stream_set[stream_type])

    def add_data(
        self,
        pcds: Optional[List[Dict[str, Any]]] = None,
        boxes: Optional[rr.Boxes3D] = None,
        arrows: Optional[rr.Arrows3D] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        frame_idx: Optional[int] = None,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Log a complete frame of multi-modal sensor data to the visualization.

        This is the main data ingestion method that accepts various sensor data
        types and logs them to appropriate namespaces in the Rerun visualization.
        It handles temporal synchronization, namespace management, and automatic
        cleanup of stale data streams.

        Args:
            pcds: List of point cloud dictionaries. Each dict should contain:
                - "pcd": rr.Points3D object with the point cloud data
                - "lidar_id" (optional): String identifier for the LiDAR sensor
                If lidar_id is missing, points are logged as "points_<idx>"

            boxes: 3D bounding boxes as rr.Boxes3D object, typically representing
                object detections or annotations. Logged to "lidar/boxes" namespace.

            arrows: 3D arrow vectors as rr.Arrows3D object, typically representing
                velocity fields or flow data. Logged to "lidar/arrows" namespace.

            images: List of camera image dictionaries. Each dict should contain:
                - "image": rr.Image object with the camera image data
                - "camera_id" (optional): String identifier for the camera
                If camera_id is missing, images are logged as "image_<idx>"

            frame_idx: Frame sequence number for temporal navigation. If None,
                auto-increments from the current frame counter.

            timestamp_ns: Nanosecond timestamp for precise time synchronization.
                Used to align data from multiple sensors with different rates.
        """
        # Update temporal context for this frame
        self.set_frame_counter(frame_idx=frame_idx, timestamp_ns=timestamp_ns)
        curr_stream_set = {name: set() for name in self.stream_names}

        # Process and log point cloud data from LiDAR sensors
        # Point clouds (per-lidar or generic)
        if pcds is not None:
            for idx in range(len(pcds)):
                assert type(pcds[idx]["pcd"]) == rr.Points3D
                if "lidar_id" in pcds[idx]:
                    # Use sensor-specific namespace for identified LiDARs
                    rr.log("lidar/" + pcds[idx]["lidar_id"], pcds[idx]["pcd"])
                    curr_stream_set["pcds"].add("lidar/" + pcds[idx]["lidar_id"])
                else:
                    # Use generic numbering for unidentified point clouds
                    rr.log("lidar/points_" + str(idx), pcds[idx]["pcd"])
                    curr_stream_set["pcds"].add("lidar/points_" + str(idx))

        # Process and log 3D bounding box data (object detections/annotations)
        # 3D boxes (detections/annotations)
        if boxes is not None:
            assert type(boxes) == rr.Boxes3D
            rr.log("lidar/boxes", boxes)
            curr_stream_set["boxes"].add("lidar/boxes")

        # Process and log flow vector data (velocity, motion fields)
        # Flow arrows (e.g., velocities)
        if arrows is not None:
            assert type(arrows) == rr.Arrows3D
            rr.log("lidar/arrows", arrows)
            curr_stream_set["flow"].add("lidar/arrows")

        # Process and log camera image data from multiple sensors
        # Camera images (per-camera or generic)
        if images is not None:
            for idx in range(len(images)):
                assert type(images[idx]["image"]) == rr.Image
                if "camera_id" in images[idx]:
                    # Use camera-specific namespace for identified cameras
                    rr.log(f"{images[idx]['camera_id']}/image", images[idx]["image"])
                    curr_stream_set["images"].add("camera/" + images[idx]["camera_id"])
                else:
                    # Use generic numbering for unidentified images
                    rr.log("camera/image_" + str(idx), images[idx]["image"])
                    curr_stream_set["images"].add("camera/image_" + str(idx))

        # Clean up stale data from previous frames
        # Blank anything not updated this frame to avoid stale visuals
        self.clear_empty_namespaces(curr_stream_set)
