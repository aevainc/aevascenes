# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# The AevaScenes dataset is licensed separately under the AevaScenes Dataset License.
# See https://scenes.aeva.com/license for the full license text.

import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
    - 3D LiDAR point cloud rendering
    - Real-time data streaming and namespace management
    - Interactive controls for temporal navigation
    """

    def __init__(
        self,
        name: str = "AevaScenes-Visualizer",
        web_port: int = 9590,
        grpc_port: int = 9591,
        pcd_types: str = "compensated",
        show_images: bool = True,
        coordinate_frame: str = "vehicle",
        world_look_target: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Initialize the AevaScenes Rerun visualizer.

        Sets up the Rerun visualization environment with a default layout optimized
        for AevaScenes data. Creates both gRPC server and web viewer interfaces
        and displays connection instructions.

        Args:
            name: Display name for the Rerun application instance
            web_port: TCP port for the web viewer interface
            grpc_port: TCP port for the gRPC server
            pcd_types: Point cloud layout - "compensated", "raw", or "raw_and_compensated"
            show_images: Include camera panels in the Rerun layout
            coordinate_frame: "vehicle" or "world" — controls the default 3D camera framing
            world_look_target: Optional XYZ look-at point for world-frame sequences
        """
        self.dual_pcd = pcd_types == "raw_and_compensated"
        self.show_images = show_images
        self.coordinate_frame = coordinate_frame
        self.world_look_target = world_look_target or (150.0, 0.0, 0.0)

        # UI and connection configuration
        background_color = [0, 0, 0]  # Black background for 3D views
        self.web_port = web_port
        self.grpc_port = grpc_port
        self.curr_frame_idx = 0  # Track current frame for auto-increment

        blueprint = self._build_blueprint(background_color)

        # Initialize Rerun server and viewer interfaces
        # Start a local Rerun gRPC server and expose a web viewer
        rr.init(name)
        server_uri = rr.serve_grpc(
            grpc_port=self.grpc_port,
            default_blueprint=blueprint,
            cors_allow_origin=["*"],
        )
        rr.send_blueprint(blueprint, make_active=True, make_default=True)
        rr.serve_web_viewer(connect_to=server_uri, web_port=self.web_port, open_browser=False)
        self.server_uri = server_uri

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
            f"\n[bold green]AevaScenes viewer is ready. Choose how to connect:[/bold green]\n\n"
            f"[bold green]1) RECOMMENDED — Native Rerun viewer[/bold green] [green](handles full sequences)\n"
            f"   Run this in a terminal on your local machine:\n\n"
            f"   [bright_cyan]{app_viewer_link}[/bright_cyan]\n\n"
            f"   [green]For large sequences, raise the memory limit, e.g.:[/green]\n"
            f"   [bright_cyan]rerun --memory-limit 16GB --connect {url}[/bright_cyan]\n\n"
            f"[bold yellow]2) Browser viewer — NOT recommended for full sequences[/bold yellow]\n"
            f"   [yellow]The web viewer is limited to ~2 GB of RAM (a browser/WASM cap). "
            f"Long sequences will drop frames or crash with an out-of-memory error.[/yellow]\n"
            f"   [yellow]Use only for short previews. Prefer option 1 above.[/yellow]\n"
            f"   [yellow]To reduce load: --no-images, --pcd-type compensated, --image-downsample-factor 8[/yellow]\n\n"
            f"   [bright_cyan]{web_viewer_link}[/bright_cyan]\n\n"
            f"[green]Install rerun if needed: pip install rerun-sdk==0.33.0[/green]\n"
            f"[green]The server stays running after streaming finishes. Press Ctrl+C in this terminal to stop.[/green]"
        )
        print(message)

    def keep_alive(self) -> None:
        """Block until Ctrl+C so the gRPC/web servers stay reachable for remote viewers."""
        print(
            f"\n[green]Streaming complete. Servers listening on gRPC port {self.grpc_port} "
            f"and web port {self.web_port}. Press Ctrl+C to stop.[/green]"
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[green]Shutting down visualizer.[/green]")

    def _camera_row(self) -> rrb.Horizontal:
        return rrb.Horizontal(
            rrb.Spatial2DView(origin="/front_wide_camera", name="front_wide_camera"),
            rrb.Spatial2DView(origin="/front_narrow_camera", name="front_narrow_camera"),
            rrb.Spatial2DView(origin="/right_camera", name="right_camera"),
            rrb.Spatial2DView(origin="/rear_wide_camera", name="rear_wide_camera"),
            rrb.Spatial2DView(origin="/rear_narrow_camera", name="rear_narrow_camera"),
            rrb.Spatial2DView(origin="/left_camera", name="left_camera"),
            column_shares=[1, 1, 1, 1, 1, 1],
        )

    def _lidar_view_contents(self, root: Optional[str] = None) -> List[str]:
        """Entity query for lidar Spatial3DView — matches logged sensor paths."""
        if root is not None:
            return [f"+ {root}/**"]
        return [
            "+ /front_wide_lidar",
            "+ /front_narrow_lidar",
            "+ /right_lidar",
            "+ /rear_wide_lidar",
            "+ /rear_narrow_lidar",
            "+ /left_lidar",
            "+ /boxes",
            "+ /arrows",
            "+ /ego/**",
        ]

    def _spatial3d_lidar_view(
        self,
        *,
        origin: str,
        contents: List[str],
        name: str,
        background_color: List[float],
    ) -> rrb.Spatial3DView:
        return rrb.Spatial3DView(
            origin=origin,
            contents=contents,
            name=name,
            background=background_color,
        )

    def _build_blueprint(self, background_color: List[float]) -> rrb.Blueprint:
        if self.dual_pcd:
            lidar_layout = rrb.Horizontal(
                self._spatial3d_lidar_view(
                    origin="/lidar_raw",
                    contents=self._lidar_view_contents("/lidar_raw"),
                    name="raw",
                    background_color=background_color,
                ),
                self._spatial3d_lidar_view(
                    origin="/lidar_compensated",
                    contents=self._lidar_view_contents("/lidar_compensated"),
                    name="compensated",
                    background_color=background_color,
                ),
                column_shares=[1, 1],
            )
        else:
            lidar_layout = self._spatial3d_lidar_view(
                origin="/",
                contents=self._lidar_view_contents(),
                name="lidar",
                background_color=background_color,
            )

        if self.show_images:
            layout = rrb.Vertical(lidar_layout, self._camera_row(), row_shares=[7.5, 2.5])
        else:
            layout = lidar_layout

        return rrb.Blueprint(
            layout,
            rrb.TimePanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.BlueprintPanel(state="collapsed"),
        )

    def _pcd_namespace(self, pcd_data: Dict[str, Any]) -> str:
        lidar_id = pcd_data["lidar_id"].split("/")[0]
        if self.dual_pcd:
            root = "lidar_raw" if pcd_data.get("pcd_type") == "raw" else "lidar_compensated"
            return f"/{root}/{lidar_id}"
        return f"/{lidar_id}"

    def _box_namespaces(self) -> List[str]:
        if self.dual_pcd:
            return ["/lidar_raw/boxes", "/lidar_compensated/boxes"]
        return ["/boxes"]

    def _arrow_namespaces(self) -> List[str]:
        if self.dual_pcd:
            return ["/lidar_raw/arrows", "/lidar_compensated/arrows"]
        return ["/arrows"]

    def initialize(self) -> None:
        """Perform one-time scene setup. Currently a no-op."""

    def log_ego_trajectory(self, frames: List[Dict[str, Any]]) -> None:
        """Log the full ego path in world coordinates as a static green polyline."""
        origins = np.array([aeva_utils.pose_to_matrix(frame["ego_pose"])[:3, 3] for frame in frames])
        if len(origins) < 2:
            return
        line = rr.LineStrips3D([origins], colors=[[0, 255, 0]], radii=0.3)
        if self.dual_pcd:
            for root in ("/lidar_raw", "/lidar_compensated"):
                rr.log(f"{root}/ego/trajectory", line, static=True)
        else:
            rr.log("/ego/trajectory", line, static=True)

    def log_ego_pose(self, transform: np.ndarray) -> None:
        """Log the current ego origin and +X forward axis in world coordinates."""
        if self.coordinate_frame != "world":
            return
        origin = transform[:3, 3]
        forward = transform[:3, 0] * 5.0
        origin_entity = rr.Points3D([origin], colors=[[255, 255, 0]], radii=0.5)
        forward_entity = rr.Arrows3D(origins=[origin], vectors=[forward], colors=[[255, 64, 64]], radii=0.15)
        if self.dual_pcd:
            for root in ("/lidar_raw", "/lidar_compensated"):
                rr.log(f"{root}/ego/origin", origin_entity)
                rr.log(f"{root}/ego/forward", forward_entity)
        else:
            rr.log("/ego/origin", origin_entity)
            rr.log("/ego/forward", forward_entity)

    def set_time_sequence(self, name: str, value: int) -> None:
        rr.set_time(name, sequence=value)

    def set_time_nanos(self, name: str, value: int) -> None:
        rr.set_time(name, timestamp=np.datetime64(value, "ns"))

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
            rr.set_time("frame_idx", sequence=frame_idx)
            self.curr_frame_idx = frame_idx
        else:
            rr.set_time("frame_idx", sequence=self.curr_frame_idx)
            self.curr_frame_idx += 1

        if timestamp_ns is not None:
            rr.set_time("time", timestamp=np.datetime64(timestamp_ns, "ns"))

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
                object detections or annotations. Logged to "/boxes" namespace.

            arrows: 3D arrow vectors as rr.Arrows3D object, typically representing
                velocity fields or flow data. Logged to "/arrows" namespace.

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
                namespace = self._pcd_namespace(pcds[idx]) if "lidar_id" in pcds[idx] else f"/points_{idx}"
                rr.log(namespace, pcds[idx]["pcd"])
                curr_stream_set["pcds"].add(namespace)

        if boxes is not None:
            assert type(boxes) == rr.Boxes3D
            for namespace in self._box_namespaces():
                rr.log(namespace, boxes)
                curr_stream_set["boxes"].add(namespace)

        if arrows is not None:
            assert type(arrows) == rr.Arrows3D
            for namespace in self._arrow_namespaces():
                rr.log(namespace, arrows)
                curr_stream_set["flow"].add(namespace)

        # Process and log camera image data from multiple sensors
        # Camera images (per-camera or generic)
        if images is not None:
            for idx in range(len(images)):
                assert type(images[idx]["image"]) == rr.Image
                if "camera_id" in images[idx]:
                    # Use camera-specific namespace for identified cameras
                    rr.log(f"/{images[idx]['camera_id']}/image", images[idx]["image"])
                    curr_stream_set["images"].add("/" + images[idx]["camera_id"])
                else:
                    # Use generic numbering for unidentified images
                    rr.log("/camera/image_" + str(idx), images[idx]["image"])
                    curr_stream_set["images"].add("/camera/image_" + str(idx))

        # Clean up stale data from previous frames
        # Blank anything not updated this frame to avoid stale visuals
        self.clear_empty_namespaces(curr_stream_set)
