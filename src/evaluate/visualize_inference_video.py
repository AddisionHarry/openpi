import argparse
import json
import subprocess
from pathlib import Path
from typing import Tuple, Dict

import cv2
import h5py
import numpy as np
import torch
import pyarrow.parquet as pq
import pytorch_kinematics as pk

# ============================================================
# Camera & projection parameters
# ============================================================

CHEST_CAMERA_IN_CHEST = torch.tensor([
    [-0.0016583  , -0.49421638, 0.86933735, 0.10547365],
    [-0.99996612 , 0.00782894, 0.00254325, 0.02926773],
    [-0.00806291 , -0.86930368, -0.49421261, 0.41119803],
    [ 0., 0., 0., 1.]
])

CAMERA_INTRINSICS = torch.tensor([
    [910.4470825195312, 0.0, 651.5833740234375, 0.0],
    [0.0, 909.9199829101562, 379.5197448730469, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

WRIST_CAMERA_IN_TCP = [
    torch.tensor([
        [0.0746, -0.9650, -0.2515, 0.05124],
        [-0.2318, 0.2285, -0.9455, -0.02052],
        [0.9699, 0.1288, -0.2066, -0.09482],
        [0., 0., 0., 1.]
    ]),
    torch.tensor([
        [0.0746, -0.9650, -0.2515, 0.05124],
        [0.2318, -0.2285, 0.9455, 0.02052],
        [0.9699, 0.1288, -0.2066, -0.09482],
        [0., 0., 0., 1.]
    ])
]

# ============================================================
# Dataset helpers
# ============================================================

def load_meta(dataset_dir: Path) -> dict:
    """Load meta information from dataset directory"""
    return json.loads((dataset_dir / "meta" / "info.json").read_text())

def resolve_episode_path(dataset_dir: Path, episode_index: int) -> Path:
    """Return parquet path for a given episode"""
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"

def resolve_chest_rgb_video(dataset_dir: Path, episode_index: int) -> Path:
    """Return video path for a given episode"""
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    video_dir = dataset_dir / "videos" / f"chunk-{chunk:03d}" / "observation.images.chest_rgb"
    for ext in (".mp4", ".mkv", ".avi"):
        p = video_dir / f"episode_{ep:06d}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No video found in {video_dir}")

def load_actions(dataset_dir: Path, episode_index: int) -> torch.Tensor:
    """Load actions from parquet file"""
    pf = pq.ParquetFile(resolve_episode_path(dataset_dir, episode_index))
    return torch.from_numpy(np.asarray(pf.read().to_pandas()["actions"].tolist(), np.float32))

# ============================================================
# Joint name normalization & mapping
# ============================================================

def normalize_joint_names(names: list[str]) -> tuple[list[str], list[int]]:
    """Normalize joint names, remove '_pos' suffix if present"""
    clean, idx = [], []
    for i, n in enumerate(names):
        if n.endswith("_pos"):
            clean.append(n[:-4])
        else:
            clean.append(n)
        idx.append(i)
    return clean, idx

def build_joint_mapping(urdf_joints: list[str], src_joint_names: list[str], src_indices: list[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build mapping from source joints to URDF joints"""
    lut = {n: i for n, i in zip(src_joint_names, src_indices)}
    indices, mask = [], []
    for j in urdf_joints:
        if j in lut:
            indices.append(lut[j])
            mask.append(1.0)
        else:
            indices.append(0)
            mask.append(0.0)
    return torch.tensor(indices), torch.tensor(mask)

# ============================================================
# FK & projection
# ============================================================

def compute_pixels(chain, joints):
    """
    Compute 2D pixel locations of left and right TCPs relative to chest camera
    joints: [T, n_joints]
    return: [2, T, 2] (left/right, T, uv)
    """
    frame_ids = torch.tensor([chain.frame_to_idx[n] for n in ("CHEST", "TCP_L", "TCP_R")])
    fk = chain.forward_kinematics(joints, frame_ids)
    chest_cam = fk["CHEST"].get_matrix() @ CHEST_CAMERA_IN_CHEST
    tcp_l = torch.linalg.inv(chest_cam) @ fk["TCP_L"].get_matrix()
    tcp_r = torch.linalg.inv(chest_cam) @ fk["TCP_R"].get_matrix()
    pts = torch.stack([tcp_l[:, :3, 3], tcp_r[:, :3, 3]], 0)
    ones = torch.ones_like(pts[..., :1])
    pts_h = torch.cat([pts, ones], -1)
    uvw = pts_h @ CAMERA_INTRINSICS.T
    return uvw[..., :2] / uvw[..., 2:3]

# ============================================================
# h5 prediction helpers
# ============================================================

def load_pred_from_h5(h5_path: Path, episode_idx: int):
    """Load predicted actions and joint names from h5"""
    with h5py.File(h5_path, "r") as f:
        grp = f[f"episode_{episode_idx:06d}"]
        actions = torch.from_numpy(grp["action"][:]).float()  # [T, chunk_len, n_pred_joints]
        joint_names = [n.decode() if isinstance(n, bytes) else n for n in grp["action_joint_names"][:]]
    return actions, joint_names

def expand_pred_actions(pred_actions: torch.Tensor, pr_map: torch.Tensor, pr_mask: torch.Tensor, n_urdf: int):
    """Expand predicted actions from [T, chunk_len, n_pred_joints] to [T, chunk_len, n_urdf]"""
    T, chunk_len, _ = pred_actions.shape
    expanded = torch.zeros((T, chunk_len, n_urdf), dtype=pred_actions.dtype)
    for i, mask in enumerate(pr_mask):
        if mask > 0:
            expanded[:, :, i] = pred_actions[:, :, pr_map[i]]
    return expanded

def compute_padding(pixels: torch.Tensor, image_size: Tuple[int, int]) -> Dict[str, float]:
    """Compute required padding for canvas"""
    H, W = image_size
    u, v = pixels[..., 0], pixels[..., 1]
    return dict(
        left=max(0.0, float(0 - u.min())),
        right=max(0.0, float(u.max() - (W - 1))),
        top=max(0.0, float(0 - v.min())),
        bottom=max(0.0, float(v.max() - (H - 1))),
    )

def compute_canvas(all_pixels: torch.Tensor, image_size: Tuple[int, int], draw: str):
    """Compute canvas size and offsets based on pixel coordinates"""
    pads = []
    if draw in ("left", "both"):
        pads.append(compute_padding(all_pixels[0], image_size))
    if draw in ("right", "both"):
        pads.append(compute_padding(all_pixels[1], image_size))
    merged = {k: max(p[k] for p in pads) for k in pads[0].keys()}
    H, W = image_size
    new_H = int(np.ceil((H + merged["top"] + merged["bottom"]) / 10) * 10)
    new_W = int(np.ceil((W + merged["left"] + merged["right"]) / 10) * 10)
    return new_H, new_W, int(merged["left"]), int(merged["top"])

# ============================================================
# Visualization helpers
# ============================================================

def draw_frame(frame: np.ndarray, pixels_ds: torch.Tensor, pixels_pred: torch.Tensor, canvas_hw, offset, image_size, draw, current_frame, trail_len=50):
    """
    Draw dataset and predicted trajectories on a canvas
    frame: [H, W, 3] uint8, pixels_ds: [2, T, 2], pixels_pred: [T, chunk_len, 2, 2] or None
    canvas_hw: (canvas_h, canvas_w), offset: (off_u, off_v), image_size: (H, W)
    draw: 'left' | 'right' | 'both', current_frame: int, trail_len: int
    """
    canvas_h, canvas_w = canvas_hw
    off_u, off_v = offset
    H, W = image_size
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # white canvas
    canvas[off_v : off_v + H, off_u : off_u + W] = frame  # embed original frame

    T = pixels_ds.shape[1]
    # Draw dataset trajectory
    for side_idx, side_name in enumerate(("left", "right")):
        if draw not in (side_name, "both"):
            continue
        pts = pixels_ds[side_idx].cpu().numpy()
        for j in range(current_frame, min(current_frame + trail_len, T - 1)):
            x0, y0 = pts[j]; x1, y1 = pts[j + 1]
            cv2.line(canvas, (int(x0 + off_u), int(y0 + off_v)), (int(x1 + off_u), int(y1 + off_v)), (0, 165, 255), 6)

    # Draw predicted trajectory
    if pixels_pred is not None:
        _, chunk_len, _, _ = pixels_pred.shape
        pred_len = min(trail_len, chunk_len)
        for side_idx, side_name in enumerate(("left", "right")):
            if draw not in (side_name, "both"):
                continue
            for i in range(pred_len - 1):
                pt0 = pixels_pred[current_frame, i, side_idx]; pt1 = pixels_pred[current_frame, i + 1, side_idx]
                x0, y0 = pt0.cpu().numpy(); x1, y1 = pt1.cpu().numpy()
                t_norm = 1 - (i / pred_len)
                color = (int(255 * t_norm), 0, int(255 * (1.0 - t_norm)))
                thickness = max(1, int(8 * t_norm))
                cv2.line(canvas, (int(x0 + off_u), int(y0 + off_v)), (int(x1 + off_u), int(y1 + off_v)), color, thickness)

    return canvas

# ============================================================
# Main
# ============================================================

def main(args):
    chain = pk.build_chain_from_urdf(args.urdf.read_bytes())
    urdf_joints = chain.get_joint_parameter_names()
    n_urdf = len(urdf_joints)

    # Dataset
    meta = load_meta(args.dataset_dir)
    img_info = meta["features"]["observation.images.chest_rgb"]["info"]
    image_size = (img_info["video.height"], img_info["video.width"])
    fps = int(img_info["video.fps"])
    ds_actions = load_actions(args.dataset_dir, args.episode_index)
    ds_names = meta["features"]["actions"]["names"]
    ds_clean, ds_idx = normalize_joint_names(ds_names)
    ds_map, ds_mask = build_joint_mapping(urdf_joints, ds_clean, ds_idx)
    ds_joints = ds_actions[:, ds_map] * ds_mask
    pixels_ds = compute_pixels(chain, ds_joints)

    # Prediction
    pixels_pred = None
    if args.pred_h5:
        pred_actions, pred_names = load_pred_from_h5(args.pred_h5, args.episode_index)
        pr_clean, pr_idx = normalize_joint_names(pred_names)
        pr_map, pr_mask = build_joint_mapping(urdf_joints, pr_clean, pr_idx)
        pred_joints = expand_pred_actions(pred_actions, pr_map, pr_mask, n_urdf)
        T, chunk_len, n_joints = pred_joints.shape
        pred_joints_flat = pred_joints.reshape(T * chunk_len, n_joints)
        pixels_flat = compute_pixels(chain, pred_joints_flat)  # [2, T*chunk_len, 2]
        pixels_pred = pixels_flat.reshape(2, T, chunk_len, 2).permute(1, 2, 0, 3).contiguous()  # [T, chunk_len, 2, 2]
        canvas_h, canvas_w, off_u, off_v = compute_canvas(torch.cat([pixels_flat, pixels_ds], dim=1), image_size, args.draw)
    else:
        canvas_h, canvas_w, off_u, off_v = compute_canvas(pixels_ds, image_size, args.draw)

    # Video
    video = cv2.VideoCapture(str(resolve_chest_rgb_video(args.dataset_dir, args.episode_index)))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{canvas_w}x{canvas_h}", "-r", str(round(fps * args.play_rate)),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", args.output],
        stdin=subprocess.PIPE
    )

    t = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        vis_frame = draw_frame(frame, pixels_ds, pixels_pred, (canvas_h, canvas_w), (off_u, off_v), \
            image_size, args.draw, t, args.trail_len)
        # ---------- draw play rate ----------
        if args.play_rate != 1.0:
            text = f"{int(fps * args.play_rate) / fps:.2f}x"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.0
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.putText(vis_frame, text, (canvas_w - text_w - 10, 10 + text_h), font, scale, (0, 0, 255), thickness)
        # ---------- draw current frame index ----------
        frame_text = f"frame {t}"
        (fw, fh), _ = cv2.getTextSize(frame_text, font, scale, thickness)
        cv2.putText(vis_frame, frame_text, (10, 10 + fh), font, scale, (255, 0, 0), thickness)

        ff.stdin.write(vis_frame[:, :, ::-1].tobytes())
        t += 1

    ff.stdin.close(); ff.wait(); video.release()
    print(f"[OK] Video saved to {args.output}")


if __name__ == "__main__":
    import os
    if os.environ.get("DEBUG_MODE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()
        print("Debugger attached, resuming execution...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--pred-h5", type=Path, default=None)
    parser.add_argument("--output", default="trajectory.mp4")
    parser.add_argument("--draw", choices=["left", "right", "both"], default="both")
    parser.add_argument("--trail-len", type=int, default=50)
    parser.add_argument("--play-rate", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
