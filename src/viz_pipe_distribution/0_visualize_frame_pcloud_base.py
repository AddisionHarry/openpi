#!/usr/bin/env python3
"""
0_visualize_rgbd_pointcloud.py

Visualize and save RGB-D point cloud from robot chest camera in robot base frame.

Usage:
    python 0_visualize_rgbd_pointcloud.py --dataset-dir /path/to/dataset --urdf /path/to/robot.urdf
                                        --episode-index 0 --out-dir ./output --depth-scale 1000.0

This script:
1. Loads RGB and depth frames from dataset.
2. Computes transformation from chest camera to robot base frame.
3. Projects RGB-D to base-frame point cloud.
4. Saves point cloud as PLY.
5. Renders point cloud from virtual camera and top-down orthographic view.
"""

import argparse
import json
from pathlib import Path
import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import pytorch_kinematics as pk
import torch

# ============================================================
# Camera parameters (fixed)
# ============================================================

CHEST_CAMERA_IN_CHEST = torch.tensor([
    [-0.0016583, -0.49421638, 0.86933735, 0.10547365],
    [-0.99996612, 0.00782894, 0.00254325, 0.02926773],
    [-0.00806291, -0.86930368, -0.49421261, 0.41119803],
    [0., 0., 0., 1.]
], dtype=torch.float32)

CAMERA_INTRINSICS = torch.tensor([
    [910.4470825195312, 0.0, 651.5833740234375],
    [0.0, 909.9199829101562, 379.5197448730469],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)

FX = CAMERA_INTRINSICS[0, 0].item()
FY = CAMERA_INTRINSICS[1, 1].item()
CX = CAMERA_INTRINSICS[0, 2].item()
CY = CAMERA_INTRINSICS[1, 2].item()

# ============================================================
# Dataset helpers
# ============================================================

def load_meta(dataset_dir: Path) -> dict:
    """Load dataset metadata from info.json."""
    meta_file = dataset_dir / "meta" / "info.json"
    return json.loads(meta_file.read_text())

def resolve_episode_path(dataset_dir: Path, episode_index: int) -> Path:
    """Resolve parquet file path for given episode index."""
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"

def resolve_video(dataset_dir: Path, episode_index: int, key: str) -> Path:
    """Resolve video path for given episode index and video key."""
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    video_dir = dataset_dir / "videos" / f"chunk-{chunk:03d}" / key
    for ext in (".mp4", ".avi", ".mkv"):
        video_path = video_dir / f"episode_{ep:06d}{ext}"
        if video_path.exists():
            return video_path
    raise FileNotFoundError(f"No video file found for key {key} at {video_dir}")

def read_first_rgb_frame(video_path: Path) -> np.ndarray:
    """Read first RGB frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError(f"Failed to read RGB frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def read_first_depth_gray16le(video_path: Path) -> np.ndarray:
    """Read first depth frame from Gray16LE encoded video."""
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        depth = frame.to_ndarray(format="gray16le")
        container.close()
        return depth
    container.close()
    raise RuntimeError(f"No depth frame found in {video_path}")

def load_actions(dataset_dir: Path, episode_index: int) -> torch.Tensor:
    """Load action sequence from parquet file as torch tensor."""
    parquet_file = pq.ParquetFile(resolve_episode_path(dataset_dir, episode_index))
    actions_list = parquet_file.read().to_pandas()["actions"].tolist()
    return torch.from_numpy(np.asarray(actions_list, np.float32))

# ============================================================
# Joint mapping
# ============================================================

def normalize_joint_names(names: list[str]) -> tuple[list[str], list[int]]:
    """Normalize joint names by removing '_pos' suffix and return names and indices."""
    clean_names, indices = [], []
    for i, name in enumerate(names):
        clean_names.append(name[:-4] if name.endswith("_pos") else name)
        indices.append(i)
    return clean_names, indices

def build_joint_mapping(urdf_joints: list[str], src_names: list[str], src_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Map URDF joint names to dataset joint indices."""
    name_to_idx = {name: idx for name, idx in zip(src_names, src_idx)}
    indices, mask = [], []
    for joint in urdf_joints:
        if joint in name_to_idx:
            indices.append(name_to_idx[joint])
            mask.append(1.0)
        else:
            indices.append(0)
            mask.append(0.0)
    return torch.tensor(indices), torch.tensor(mask)

# ============================================================
# RGB-D to base frame point cloud
# ============================================================

def rgbd_to_base_pointcloud(
    rgb: np.ndarray, depth_mm: np.ndarray, T_base_chest: torch.Tensor, depth_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert RGB-D image to base-frame point cloud with colors."""
    H, W = depth_mm.shape
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_mm.astype(np.float32) / depth_scale
    valid_mask = z > 0
    x = (u_grid - CX) * z / FX
    y = (v_grid - CY) * z / FY
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)[valid_mask]
    colors = rgb[valid_mask].astype(np.float32) / 255.0
    T = (T_base_chest @ CHEST_CAMERA_IN_CHEST).cpu().numpy()
    pts_base = (T @ pts_cam.T).T[:, :3]
    return pts_base, colors

def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Save point cloud as ASCII PLY file."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            r, g, b = (color * 255).astype(np.uint8)
            f.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")

# ============================================================
# Rendering functions
# ============================================================

def render_pointcloud_to_image(pts_base: np.ndarray, colors: np.ndarray, T_base_cam: np.ndarray,
                               K: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Render point cloud to RGB and depth images from virtual camera."""
    FX, FY = K[0, 0], K[1, 1]
    CX, CY = K[0, 2], K[1, 2]
    pts_hom = np.concatenate([pts_base, np.ones((len(pts_base), 1))], axis=1)
    T_cam_base = np.linalg.inv(T_base_cam)
    pts_cam = (T_cam_base @ pts_hom.T).T[:, :3]
    z = pts_cam[:, 2]
    mask = z > 1e-4
    pts_cam, colors = pts_cam[mask], colors[mask]
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    u = (FX * x / z + CX).astype(np.int32)
    v = (FY * y / z + CY).astype(np.int32)
    in_img_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, colors = u[in_img_mask], v[in_img_mask], z[in_img_mask], colors[in_img_mask]
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)
    depth_image = np.full((H, W), np.inf, dtype=np.float32)
    for ui, vi, zi, ci in zip(u, v, z, colors):
        if zi < depth_image[vi, ui]:
            depth_image[vi, ui] = zi
            rgb_image[vi, ui] = ci
    return (rgb_image * 255).astype(np.uint8), depth_image

def render_topdown_orthographic(
    pts_base: np.ndarray, colors: np.ndarray, center_xy: tuple[float, float],
    resolution: float, image_size: tuple[int, int], z_buffer: bool = True
) -> np.ndarray:
    """Render top-down orthographic RGB image from point cloud."""
    cx, cy = center_xy
    H, W = image_size
    xb, yb, zb = pts_base[:, 0], pts_base[:, 1], pts_base[:, 2]
    u = (-(yb - cy) / resolution + W / 2).astype(np.int32)
    v = (-(xb - cx) / resolution + H / 2).astype(np.int32)
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, colors = u[mask], v[mask], zb[mask], colors[mask]
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)
    if z_buffer:
        zbuf = np.full((H, W), -np.inf, dtype=np.float32)
        for ui, vi, zi, ci in zip(u, v, z, colors):
            if zi > zbuf[vi, ui]:
                zbuf[vi, ui] = zi
                rgb_image[vi, ui] = ci
    else:
        rgb_image[v, u] = colors
    return (rgb_image * 255).astype(np.uint8)

def annotate_topdown_grid_with_axes(img: np.ndarray, resolution: float, grid_m: float = 0.05, bold_every: int = 5) -> np.ndarray:
    """Draw metric grid with colored axes and labels on top-down orthographic image."""
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2
    grid_px = int(grid_m / resolution)
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.4
    th = 1

    # Vertical lines (Y axis, blue)
    for idx, x in enumerate(range(cx, W, grid_px)):
        is_bold = (idx % bold_every == 0)
        color, thickness = (255, 50, 150), 2 if is_bold else 1
        cv2.line(out, (x, 0), (x, H), color, thickness)
    for idx, x in enumerate(range(cx - grid_px, -1, -grid_px), 1):
        is_bold = (idx % bold_every == 0)
        color, thickness = (255, 50, 150), 2 if is_bold else 1
        cv2.line(out, (x, 0), (x, H), color, thickness)

    # Horizontal lines (X axis, green)
    for idx, y in enumerate(range(cy, -1, -grid_px)):
        is_bold = (idx % bold_every == 0)
        color, thickness = (120, 200, 120), 2 if is_bold else 1
        cv2.line(out, (0, y), (W, y), color, thickness)
    for idx, y in enumerate(range(cy + grid_px, H, grid_px), 1):
        is_bold = (idx % bold_every == 0)
        color, thickness = (120, 200, 120), 2 if is_bold else 1
        cv2.line(out, (0, y), (W, y), color, thickness)

    # Origin marker
    cv2.drawMarker(out, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(out, "(0,0)", (cx + 6, cy - 6), font, fs, (0, 0, 255), 1)
    return out

# ============================================================
# Main function
# ============================================================

def main(args: argparse.Namespace) -> None:
    """Main pipeline for RGB-D point cloud visualization."""
    chain = pk.build_chain_from_urdf(args.urdf.read_bytes())
    urdf_joints = chain.get_joint_parameter_names()
    meta = load_meta(args.dataset_dir)
    dataset_actions = load_actions(args.dataset_dir, args.episode_index)
    dataset_names = meta["features"]["actions"]["names"]
    clean_names, name_indices = normalize_joint_names(dataset_names)
    joint_map, joint_mask = build_joint_mapping(urdf_joints, clean_names, name_indices)
    joints = dataset_actions[0, joint_map] * joint_mask

    fk_result = chain.forward_kinematics(
        joints.unsqueeze(0),
        frame_indices=torch.tensor([chain.frame_to_idx["BASE"], chain.frame_to_idx["CHEST"]])
    )
    T_base_chest = fk_result["BASE"].get_matrix()[0].inverse() @ fk_result["CHEST"].get_matrix()[0]

    rgb_path = resolve_video(args.dataset_dir, args.episode_index, "observation.images.chest_rgb")
    depth_path = resolve_video(args.dataset_dir, args.episode_index, "observation.images.chest_depth")
    rgb_frame = read_first_rgb_frame(rgb_path)
    depth_frame = read_first_depth_gray16le(depth_path)

    points, colors = rgbd_to_base_pointcloud(rgb_frame, depth_frame, T_base_chest, args.depth_scale)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ply_output = args.out_dir / "base_pointcloud.ply"
    save_ply(ply_output, points, colors)
    print(f"[OK] Saved base-frame colored point cloud to {ply_output}")

    virtual_cam_pose = np.array([[0, -1, 0, 0.95], [-1, 0, 0, 0.05], [0, 0, -1, 1.65], [0, 0, 0, 1]])
    rendered_rgb, rendered_depth = render_pointcloud_to_image(points, colors, virtual_cam_pose,
                                                              CAMERA_INTRINSICS.numpy(), H=rgb_frame.shape[0], W=rgb_frame.shape[1])
    cv2.imwrite(str(args.out_dir / "render_rgb.png"), cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))

    topdown = render_topdown_orthographic(points, colors, center_xy=(0.95, 0.05), resolution=0.001, image_size=(800, 800))
    topdown = annotate_topdown_grid_with_axes(topdown, resolution=0.001, grid_m=0.05, bold_every=5)
    cv2.imwrite(str(args.out_dir / "topdown_ortho.png"), cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR))

# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RGB-D point cloud in robot base frame")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--urdf", type=Path, required=True, help="Path to robot URDF file")
    parser.add_argument("--episode-index", type=int, required=True, help="Episode index to visualize")
    parser.add_argument("--out-dir", type=Path, default=Path("viz"), help="Output directory")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth scale factor (mm to meters)")
    args = parser.parse_args()
    main(args)
