#!/usr/bin/env python3
"""
3_project_box_mask_plane_camera.py

Project a plane defined by two base-frame corners + Z height
onto original chest RGB images and overlay a semi-transparent mask,
then generate thumbnail grids of all episodes.

Usage:
    python 3_project_box_mask_plane_camera.py \
        --dataset-dir /path/to/dataset \
        --urdf /path/to/robot.urdf \
        --out-dir /path/to/output \
        --corner1 0.9 0.0 \
        --corner2 1.1 0.2 \
        --z-height 0.05 \
        --grid-size 4
"""

import argparse
from pathlib import Path
import json
from tqdm import tqdm
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


def load_meta(dataset_dir: Path):
    return json.loads((dataset_dir / "meta" / "info.json").read_text())


def resolve_episode_path(dataset_dir: Path, episode_index: int) -> Path:
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"


def resolve_video(dataset_dir: Path, episode_index: int, key: str) -> Path:
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
    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError(f"Failed to read RGB frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_actions(dataset_dir: Path, episode_index: int):
    parquet_file = pq.ParquetFile(resolve_episode_path(dataset_dir, episode_index))
    actions_list = parquet_file.read().to_pandas()["actions"].tolist()
    return torch.from_numpy(np.asarray(actions_list, np.float32))


def compute_T_base_chest(dataset_dir: Path, urdf: Path, episode_index: int):
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


def project_base_points_to_image(points_base: np.ndarray, T_base_chest: torch.Tensor):
    """Project base-frame points to image pixels."""
    T = (T_base_chest @ CHEST_CAMERA_IN_CHEST).cpu().numpy()
    pts_chest = (np.linalg.inv(T) @ np.hstack([points_base, np.ones((points_base.shape[0], 1))]).T).T
    x, y, z, _ = pts_chest.T
    u = (x * FX / z + CX).astype(np.int32)
    v = (y * FY / z + CY).astype(np.int32)
    return np.stack([u, v], axis=-1)


def overlay_polygon(img: np.ndarray, pts: np.ndarray, color=(0, 0, 255), alpha=0.3):
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def main(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    type_dir = out_dir / "projected_mask"
    type_dir.mkdir(exist_ok=True)

    corners_file = out_dir / "projected_box.json"
    corners_file.write_text(json.dumps({
        "corner1": args.corner1,
        "corner2": args.corner2,
        "z_height": args.z_height
    }, indent=2))

    meta = load_meta(args.dataset_dir)
    num_episodes = meta["total_episodes"]

    episode_images = []

    for ep_idx in tqdm(range(num_episodes), desc="Episodes"):
        rgb_path = resolve_video(args.dataset_dir, ep_idx, "observation.images.chest_rgb")
        rgb = read_first_rgb_frame(rgb_path)
        T_base_chest = compute_T_base_chest(args.dataset_dir, args.urdf, ep_idx)

        x1, y1 = args.corner1
        x2, y2 = args.corner2
        z = args.z_height
        corners_base = np.array([
            [x1, y1, z],
            [x1, y2, z],
            [x2, y2, z],
            [x2, y1, z]
        ])

        pts_img = project_base_points_to_image(corners_base, T_base_chest)
        pts_img = np.clip(pts_img, 0, [rgb.shape[1]-1, rgb.shape[0]-1])

        masked_img = overlay_polygon(rgb, pts_img, color=(255, 0, 128), alpha=0.3)

        out_path = type_dir / f"episode_{ep_idx:06d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
        episode_images.append(out_path)

    # ==================== Generate thumbnails ====================
    grid_size = args.grid_size
    thumbs_dir = out_dir / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)
    images_per_page = grid_size * grid_size
    pages = (len(episode_images) + images_per_page - 1) // images_per_page

    for page in range(pages):
        batch = episode_images[page * images_per_page:(page + 1) * images_per_page]
        sample_img = cv2.imread(str(batch[0]))
        th, tw = sample_img.shape[:2]
        grid_img = np.zeros((th * grid_size, tw * grid_size, 3), dtype=np.uint8)
        for i, img_path in enumerate(batch):
            r, c = divmod(i, grid_size)
            img = cv2.imread(str(img_path))
            grid_img[r*th:(r+1)*th, c*tw:(c+1)*tw] = img
        thumb_path = thumbs_dir / f"thumbnail_page_{page+1}.png"
        cv2.imwrite(str(thumb_path), grid_img)

    print(f"[OK] Saved {num_episodes} projected masks in {type_dir} and {pages} thumbnails in {thumbs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corner1", type=float, nargs=2, required=True)
    parser.add_argument("--corner2", type=float, nargs=2, required=True)
    parser.add_argument("--z-height", type=float, required=True)
    parser.add_argument("--grid-size", type=int, default=4)
    args = parser.parse_args()
    main(args)
