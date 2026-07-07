#!/usr/bin/env python3
# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# The AevaScenes dataset is licensed separately under the AevaScenes Dataset License.
# See https://scenes.aeva.com/license for the full license text.

import argparse
import sys
from pathlib import Path

from aevascenes import AevaScenes


def parse_arguments():
    """Parse command line arguments for AevaScenes visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize AevaScenes dataset sequences and frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset configuration
    parser.add_argument(
        "--dataroot",
        "-d",
        type=str,
        default="data/aevascenes_v0.1",
        help="Path to the AevaScenes dataset directory (for e.g. data/aevascenes_v0.1)",
    )

    # Point cloud coloring
    parser.add_argument(
        "--color-mode",
        "-c",
        choices=["velocity", "reflectivity", "semantic"],
        default="velocity",
        help="Point cloud coloring mode",
    )

    # Point cloud variant
    parser.add_argument(
        "--pcd-type",
        choices=["compensated", "raw", "raw_and_compensated"],
        default="raw_and_compensated",
        help="Point cloud variant to visualize: compensated, raw, or raw_and_compensated",
    )

    # Sequence selection
    parser.add_argument(
        "--sequence-uuid",
        "-s",
        type=str,
        default="ab87b214-a867-4e43-8d74-a2123966ed3d",
        help="UUID of the sequence to visualize",
    )

    # Image projection options
    parser.add_argument("--project-points", action="store_true", help="Project point cloud points onto images")

    parser.set_defaults(project_points=False)

    parser.add_argument("--no-images", action="store_true", help="Skip camera image loading and display")

    parser.add_argument(
        "--coordinate-frame",
        choices=["vehicle", "world"],
        default="vehicle",
        help="Coordinate frame for visualization: vehicle (ego) or world (using ego_pose)",
    )

    # Image downsampling
    parser.add_argument(
        "--image-downsample-factor",
        type=int,
        default=2,
        choices=[1, 2, 4, 8],
        help="Factor by which to downsample images",
    )

    # Utility commands
    parser.add_argument("--list-sequences", action="store_true", help="List all available sequences in the dataset")

    parser.add_argument(
        "--no-keep-alive",
        action="store_true",
        help="Exit immediately after streaming instead of keeping the rerun server open",
    )

    return parser.parse_args()


def validate_dataroot(dataroot):
    """Validate that the dataroot exists."""
    path = Path(dataroot).expanduser().resolve()

    if not path.exists():
        print(f"Error: Dataset path '{dataroot}' does not exist.")
        return False
    if not path.is_dir():
        print(f"Error: Dataset path '{dataroot}' is not a directory.")
        return False
    return True


def main():
    """Main function to run the AevaScenes visualizer."""
    args = parse_arguments()

    # Validate dataroot
    if not validate_dataroot(args.dataroot):
        sys.exit(1)

    # Initialize AevaScenes
    try:
        print(f"Loading AevaScenes dataset from: {args.dataroot}")
        avs = AevaScenes(dataroot=args.dataroot)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Handle list sequences command
    if args.list_sequences:
        print("Available sequences:")
        avs.list_sequences()
        sys.exit(0)

    sequence_uuid = args.sequence_uuid

    # Display configuration
    print("\nVisualization Configuration:")
    print(f"  Color mode: {args.color_mode}")
    print(f"  Point cloud type: {args.pcd_type}")
    print(f"  Project points on image: {args.project_points}")
    print(f"  Include images: {not args.no_images}")
    print(f"  Coordinate frame: {args.coordinate_frame}")
    print(f"  Image downsample factor: {args.image_downsample_factor}")
    print(f"  Sequence UUID: {sequence_uuid}")
    print()

    # Run visualization
    try:
        sequence_error = avs.explain_invalid_sequence_uuid(sequence_uuid)
        if sequence_error:
            print(f"Error: {sequence_error}")
            sys.exit(1)

        print(f"Visualizing sequence: {sequence_uuid}")
        avs.visualize_sequence(
            sequence_uuid=sequence_uuid,
            pcd_color_mode=args.color_mode,
            pcd_type=args.pcd_type,
            project_points_on_image=args.project_points,
            image_downsample_factor=args.image_downsample_factor,
            include_images=not args.no_images,
            coordinate_frame=args.coordinate_frame,
            init_visualizer=True,
            keep_alive=not args.no_keep_alive,
        )
    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
