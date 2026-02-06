#!/usr/bin/env python3
"""
Visualize the first RGB-D frame of a dataset episode by reconstructing a point cloud in the base frame.

This script:
1. Loads robot joint states from a dataset episode.
2. Computes forward kinematics using a URDF to obtain the base-to-chest transform.
3. Reads the first RGB frame and depth frame from chest-mounted cameras.
4. Projects RGB-D data into a 3D point cloud expressed in the robot base frame.
5. Saves the resulting colored point cloud as an ASCII PLY file.

Command-line usage:
    python 0_visualize_base_pointcloud.py \
        --dataset-dir /path/to/dataset \
        --urdf /path/to/robot.urdf \
        --episode-index 0 \
        --fx 910.447 \
        --fy 909.920 \
        --cx 651.583 \
        --cy 379.520 \
        --out-dir viz \
        --depth-scale 1000.0
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import pytorch_kinematics as pk
import torch


# ============================================================
# Camera parameters (kept identical to existing pipeline)
# ============================================================

CHEST_CAMERA_IN_CHEST = torch.tensor(
    [
        [-0.0016583, -0.49421638, 0.86933735, 0.10547365],
        [-0.99996612, 0.00782894, 0.00254325, 0.02926773],
        [-0.00806291, -0.86930368, -0.49421261, 0.41119803],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)

CAMERA_INTRINSICS = torch.tensor(
    [
        [910.4470825195312, 0.0, 651.5833740234375],
        [0.0, 909.9199829101562, 379.5197448730469],
        [0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


# ============================================================
# Dataset utilities
# ============================================================

def load_meta(dataset_dir: Path) -> Dict:
    """Load dataset metadata JSON."""
    return json.loads((dataset_dir / "meta" / "info.json").read_text())


def resolve_episode_path(dataset_dir: Path, episode_index: int) -> Path:
    """Resolve the parquet file path for a given episode index."""
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"


def resolve_video(dataset_dir: Path, episode_index: int, key: str) -> Path:
    """Resolve the video file path for a given episode and camera key."""
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    video_dir = dataset_dir / "videos" / f"chunk-{chunk:03d}" / key
    for ext in (".mp4", ".avi", ".mkv"):
        candidate = video_dir / f"episode_{ep:06d}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(str(video_dir))


def read_first_rgb_frame(video_path: Path) -> np.ndarray:
    """Read the first RGB frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read RGB frame")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_first_depth_gray16le(video_path: Path) -> np.ndarray:
    """Read the first 16-bit grayscale depth frame from a video file."""
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        return frame.to_ndarray(format="gray16le")
    raise RuntimeError("No depth frame found")


def load_actions(dataset_dir: Path, episode_index: int) -> torch.Tensor:
    """Load action tensor for a given episode."""
    parquet = pq.ParquetFile(resolve_episode_path(dataset_dir, episode_index))
    actions = np.asarray(parquet.read().to_pandas()["actions"].tolist(), np.float32)
    return torch.from_numpy(actions)


def normalize_joint_names(names: List[str]) -> Tuple[List[str], List[int]]:
    """Strip '_pos' suffixes from joint names while keeping index mapping."""
    clean, idx = [], []
    for i, name in enumerate(names):
        clean.append(name[:-4] if name.endswith("_pos") else name)
        idx.append(i)
    return clean, idx


def build_joint_mapping(
    urdf_joints: List[str], src_names: List[str], src_idx: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build index and mask tensors mapping dataset joints to URDF joints."""
    lut = {name: i for name, i in zip(src_names, src_idx)}
    indices, mask = [], []
    for joint in urdf_joints:
        if joint in lut:
            indices.append(lut[joint])
            mask.append(1.0)
        else:
            indices.append(0)
            mask.append(0.0)
    return torch.tensor(indices), torch.tensor(mask)


# ============================================================
# Core: RGB-D to base-frame point cloud
# ============================================================

def rgbd_to_base_pointcloud(rgb: np.ndarray, depth_mm: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                            T_base_chest: torch.Tensor, depth_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an RGB-D image pair into a colored point cloud in the robot base frame.

    Args:
        rgb: RGB image as uint8 array of shape (H, W, 3).
        depth_mm: Depth image in millimeters as uint16 array of shape (H, W).
        fx, fy, cx, cy: Camera intrinsic parameters.
        T_base_chest: Homogeneous transform from chest frame to base frame.
        depth_scale: Scale factor converting depth units to meters.

    Returns:
        points_base: (N, 3) array of 3D points in base frame.
        colors: (N, 3) array of RGB colors in range [0, 1].
    """
    H, W = depth_mm.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    z = depth_mm.astype(np.float32) / depth_scale
    valid = z > 0.0

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)[valid]
    colors = rgb[valid] / 255.0

    T = (T_base_chest @ CHEST_CAMERA_IN_CHEST).numpy()
    pts_base = (T @ pts_cam.T).T[:, :3]

    return pts_base, colors


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Save a colored point cloud to an ASCII PLY file."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            r, g, b = (c * 255).astype(np.uint8)
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")


# ============================================================
# Main entry point
# ============================================================

def main(args: argparse.Namespace) -> None:
    """Main execution function."""
    chain = pk.build_chain_from_urdf(args.urdf.read_bytes())
    urdf_joints = chain.get_joint_parameter_names()

    meta = load_meta(args.dataset_dir)
    actions = load_actions(args.dataset_dir, args.episode_index)
    action_names = meta["features"]["actions"]["names"]
    clean_names, src_idx = normalize_joint_names(action_names)
    joint_map, joint_mask = build_joint_mapping(urdf_joints, clean_names, src_idx)

    joints = actions[0, joint_map] * joint_mask

    fk = chain.forward_kinematics(joints.unsqueeze(0), frame_ids=[chain.frame_to_idx["BASE"], chain.frame_to_idx["CHEST"]])
    T_base_chest = fk["BASE"].get_matrix()[0].inverse() @ fk["CHEST"].get_matrix()[0]

    rgb = read_first_rgb_frame(resolve_video(args.dataset_dir, args.episode_index, "observation.images.chest_rgb"))
    depth = read_first_depth_gray16le(resolve_video(args.dataset_dir, args.episode_index, "observation.images.chest_depth"))

    points, colors = rgbd_to_base_pointcloud(rgb, depth, args.fx, args.fy, args.cx, args.cy, T_base_chest, args.depth_scale)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = args.out_dir / "base_pointcloud.ply"
    save_ply(out_ply, points, colors)
    print(f"[OK] Saved base-frame point cloud to {out_ply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("viz"))
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    main(parser.parse_args())
