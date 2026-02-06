#!/usr/bin/env python3
"""
3_generate_box_topdown_masked.py

Generate top-down RGB images for each episode with a semi-transparent mask
on a specified rectangular region, and create thumbnail grids of all episodes.

Usage:
    python 3_generate_box_topdown_masked.py \
        --dataset-dir /path/to/dataset \
        --urdf /path/to/robot.urdf \
        --out-dir /path/to/output \
        --corner1 0.9 0.0 \
        --corner2 1.1 0.2 \
        --grid-size 4 \
        --depth-scale 1000.0
"""

import argparse
import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import av
import cv2
import numpy as np
import torch
import pyarrow.parquet as pq
import pytorch_kinematics as pk

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

def load_meta(dataset_dir: Path) -> Dict:
    """Load dataset metadata JSON."""
    return json.loads((dataset_dir / "meta" / "info.json").read_text())

def resolve_episode_path(dataset_dir: Path, episode_index: int) -> Path:
    """Resolve parquet file path for given episode index."""
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

# ==================== Helper functions ====================

def overlay_mask(img: np.ndarray, mask_rect: tuple[int, int, int, int], color=(0, 0, 255), alpha=0.3):
    """Overlay a semi-transparent rectangle on image."""
    x1, y1, x2, y2 = mask_rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def compute_T_base_chest(dataset_dir: Path, urdf: Path, episode_index: int):
    """Compute transformation from chest to base frame."""
    chain = pk.build_chain_from_urdf(urdf.read_bytes())
    urdf_joints = chain.get_joint_parameter_names()
    meta = load_meta(dataset_dir)
    dataset_actions = load_actions(dataset_dir, episode_index)
    dataset_names = meta["features"]["actions"]["names"]
    clean_names = [n[:-4] if n.endswith("_pos") else n for n in dataset_names]
    name_to_idx = {name: i for i, name in enumerate(clean_names)}
    joint_vec = torch.zeros(len(urdf_joints))
    for i, j in enumerate(urdf_joints):
        if j in name_to_idx:
            joint_vec[i] = dataset_actions[0, name_to_idx[j]]
    fk_result = chain.forward_kinematics(
        joint_vec.unsqueeze(0),
        frame_indices=torch.tensor([chain.frame_to_idx["BASE"], chain.frame_to_idx["CHEST"]])
    )
    return fk_result["BASE"].get_matrix()[0].inverse() @ fk_result["CHEST"].get_matrix()[0]

# ==================== Main pipeline ====================

def main(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.dataset_dir)
    num_episodes = meta["total_episodes"]

    # 保存 corner 坐标
    corners_file = out_dir / "masked_box.json"
    corners_file.write_text(json.dumps({
        "corner1": args.corner1,
        "corner2": args.corner2
    }, indent=2))

    # 创建类型文件夹，比如 topdown_masked
    type_dir = out_dir / "topdown_masked"
    type_dir.mkdir(exist_ok=True)

    episode_images = []

    # 转换世界坐标到像素
    def world2pixel(x, y, center=(0.95, 0.05), resolution=0.001, img_size=(800, 800)):
        cx, cy = center
        H, W = img_size
        u = int(-(y - cy) / resolution + W / 2)
        v = int(-(x - cx) / resolution + H / 2)
        return u, v

    # 计算像素矩形区域
    px1, py1 = world2pixel(args.corner1[0], args.corner1[1])
    px2, py2 = world2pixel(args.corner2[0], args.corner2[1])
    mask_rect = (min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2))

    for ep_idx in tqdm(range(num_episodes), desc="Episodes"):
        rgb_path = resolve_video(args.dataset_dir, ep_idx, "observation.images.chest_rgb")
        depth_path = resolve_video(args.dataset_dir, ep_idx, "observation.images.chest_depth")
        rgb = read_first_rgb_frame(rgb_path)
        depth = read_first_depth_gray16le(depth_path)
        T_base_chest = compute_T_base_chest(args.dataset_dir, args.urdf, ep_idx)

        points, colors = rgbd_to_base_pointcloud(rgb, depth, T_base_chest, args.depth_scale)
        topdown = render_topdown_orthographic(points, colors, center_xy=(0.95, 0.05),
                                              resolution=0.001, image_size=(800, 800))
        topdown = annotate_topdown_grid_with_axes(topdown, resolution=0.001, grid_m=0.05)
        topdown = overlay_mask(topdown, mask_rect, color=(0, 0, 255), alpha=0.3)

        # 保存图片到 type_dir 下
        out_path = type_dir / f"episode_{ep_idx:06d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR))
        episode_images.append(out_path)

    # ==================== Generate thumbnails ====================
    grid_size = args.grid_size
    thumbs_dir = out_dir / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)
    images_per_page = grid_size * grid_size
    pages = (len(episode_images) + images_per_page - 1) // images_per_page

    for page in range(pages):
        batch = episode_images[page * images_per_page:(page + 1) * images_per_page]
        # 以第一张图确定大小
        sample_img = cv2.imread(str(batch[0]))
        th, tw = sample_img.shape[:2]
        grid_img = np.zeros((th * grid_size, tw * grid_size, 3), dtype=np.uint8)
        for i, img_path in enumerate(batch):
            r, c = divmod(i, grid_size)
            img = cv2.imread(str(img_path))
            grid_img[r*th:(r+1)*th, c*tw:(c+1)*tw] = img
        thumb_path = thumbs_dir / f"thumbnail_page_{page+1}.png"
        cv2.imwrite(str(thumb_path), grid_img)

    print(f"[OK] Saved {len(episode_images)} top-down images in {type_dir} and {pages} thumbnails in {thumbs_dir}")


# ==================== CLI ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corner1", type=float, nargs=2, required=True, help="[x_min, y_min] in meters")
    parser.add_argument("--corner2", type=float, nargs=2, required=True, help="[x_max, y_max] in meters")
    parser.add_argument("--grid-size", type=int, default=4, help="Thumbnail grid size")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    args = parser.parse_args()
    main(args)
