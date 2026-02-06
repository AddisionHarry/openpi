#!/usr/bin/env python3
"""
2_compute_pipe_centroid_base_xyz.py

Compute 3D pipe centroids in robot base frame from pixel centroids and RGB-D images,
with robust Z estimation, optional fixed Z correction, Z clustering analysis,
and visualization.

Usage:
    python 2_compute_pipe_centroid_base_xyz.py
        --dataset-dir /path/to/dataset
        --urdf /path/to/robot.urdf
        --centroids-jsonl /path/to/centroids.jsonl
        --out-dir ./output
        [--radius-px 6]
        [--depth-scale 1000.0]
        [--fix-z 0.2]
        [--z-clusters 2]
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import pyarrow.parquet as pq
import pytorch_kinematics as pk
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

FX, FY = CAMERA_INTRINSICS[0, 0].item(), CAMERA_INTRINSICS[1, 1].item()
CX, CY = CAMERA_INTRINSICS[0, 2].item(), CAMERA_INTRINSICS[1, 2].item()

# ============================================================
# Dataset helper functions
# ============================================================

def load_meta(dataset_dir: Path) -> dict:
    """Load dataset metadata from info.json"""
    return json.loads((dataset_dir / "meta" / "info.json").read_text())

def resolve_episode_parquet(dataset_dir: Path, episode_index: int) -> Path:
    """Get Parquet file path for a given episode index"""
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"

def resolve_video(dataset_dir: Path, episode_index: int, key: str) -> Path:
    """Get video file path for a given episode and key"""
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    base = dataset_dir / "videos" / f"chunk-{chunk:03d}" / key
    for ext in (".mp4", ".avi", ".mkv"):
        p = base / f"episode_{ep:06d}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(base)

def read_first_depth(video_path: Path) -> np.ndarray:
    """Read first depth frame from a Gray16LE video"""
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        depth = frame.to_ndarray(format="gray16le")
        container.close()
        return depth
    raise RuntimeError(video_path)

def load_actions(dataset_dir: Path, episode_index: int) -> torch.Tensor:
    """Load action sequence from Parquet file as torch tensor"""
    pf = pq.ParquetFile(resolve_episode_parquet(dataset_dir, episode_index))
    acts = pf.read().to_pandas()["actions"].tolist()
    return torch.from_numpy(np.asarray(acts, np.float32))

# ============================================================
# Kinematics
# ============================================================

def compute_T_base_chest(dataset_dir: Path, urdf: Path, episode_index: int) -> torch.Tensor:
    """Compute transformation from chest to base frame"""
    chain = pk.build_chain_from_urdf(urdf.read_bytes())
    urdf_joints = chain.get_joint_parameter_names()
    meta = load_meta(dataset_dir)
    names = meta["features"]["actions"]["names"]
    clean = [n[:-4] if n.endswith("_pos") else n for n in names]
    name_to_idx = {n: i for i, n in enumerate(clean)}

    actions = load_actions(dataset_dir, episode_index)
    joint_vec = torch.zeros(len(urdf_joints))
    for i, j in enumerate(urdf_joints):
        if j in name_to_idx:
            joint_vec[i] = actions[0, name_to_idx[j]]

    fk = chain.forward_kinematics(
        joint_vec.unsqueeze(0),
        frame_indices=torch.tensor([chain.frame_to_idx["BASE"], chain.frame_to_idx["CHEST"]])
    )
    return fk["BASE"].get_matrix()[0].inverse() @ fk["CHEST"].get_matrix()[0]

# ============================================================
# Core computations
# ============================================================

def pixel_to_base(u: int, v: int, z_cam: float, T_base_chest: torch.Tensor) -> np.ndarray:
    """Convert a pixel and depth to base frame XYZ"""
    x = (u - CX) * z_cam / FX
    y = (v - CY) * z_cam / FY
    cam = torch.tensor([x, y, z_cam, 1.0], dtype=T_base_chest.dtype, device=T_base_chest.device)
    base = (T_base_chest @ CHEST_CAMERA_IN_CHEST @ cam)[:3]
    return base.cpu().numpy()

def collect_z_samples(depth: np.ndarray, u: int, v: int, r: int, depth_scale: float) -> np.ndarray:
    """Collect nearby Z samples around pixel (u,v)"""
    H, W = depth.shape
    zs = []
    for du in range(-r, r+1):
        for dv in range(-r, r+1):
            uu, vv = u + du, v + dv
            if 0 <= uu < W and 0 <= vv < H:
                d = depth[vv, uu]
                if d > 0:
                    zs.append(d / depth_scale)
    return np.array(zs)

# ============================================================
# Visualization functions
# ============================================================

def plot_xyz(points: np.ndarray, out: Path, prefix: str):
    """Plot 3D scatter and XY/XZ/YZ views"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=6)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{prefix} - 3D")
    ax.grid(True)
    plt.savefig(out / f"{prefix}_3d.png", bbox_inches="tight")
    plt.close()

    views = [(x, y, "X (m)", "Y (m)", "xy"), (x, z, "X (m)", "Z (m)", "xz"), (y, z, "Y (m)", "Z (m)", "yz")]
    for a, b, xl, yl, name in views:
        plt.figure(figsize=(5, 5), dpi=200)
        plt.scatter(a, b, s=6)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(f"{prefix} - {name.upper()}")
        plt.grid(True)
        plt.savefig(out / f"{prefix}_{name}.png", bbox_inches="tight")
        plt.close()

def plot_z_clusters(xyz: np.ndarray, labels: np.ndarray, out: Path):
    """Plot 3D scatter of points colored by Z cluster"""
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    for k in np.unique(labels):
        pts = xyz[labels == k]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, label=f"cluster {k}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.set_title("Z clusters in BASE frame")
    ax.grid(True)
    plt.savefig(out / "z_clusters_3d.png", bbox_inches="tight")
    plt.close()

# ============================================================
# Main pipeline
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--centroids-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--radius-px", type=int, default=6)
    ap.add_argument("--depth-scale", type=float, default=1000.0)
    ap.add_argument("--fix-z", type=float, default=None, help="Fixed Z height in BASE frame (meters)")
    ap.add_argument("--z-clusters", type=int, default=2, help="Number of Z clusters (KMeans)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / "centroids_base_xyz.jsonl"
    entries = [json.loads(l) for l in args.centroids_jsonl.read_text().splitlines()]
    results = []

    for e in tqdm(entries):
        if not e.get("valid", True):
            continue
        idx = e["episode_index"]
        u, v = int(e["cX"]), int(e["cY"])
        depth = read_first_depth(resolve_video(args.dataset_dir, idx, "observation.images.chest_depth"))
        T = compute_T_base_chest(args.dataset_dir, args.urdf, idx)
        zs = collect_z_samples(depth, u, v, args.radius_px, args.depth_scale)
        if len(zs) < 5:
            continue
        z_mean = zs.mean()
        z0 = depth[v, u] / args.depth_scale
        xyz_raw = pixel_to_base(u, v, z0, T)
        xyz_zmean = pixel_to_base(u, v, z_mean, T)

        entry = {
            "episode_index": idx,
            "pixel": {"u": u, "v": v},
            "xyz_raw": xyz_raw.tolist(),
            "xyz_zmean": xyz_zmean.tolist(),
            "z_mean": float(z_mean),
            "z_var": float(zs.var()),
            "num_samples": int(len(zs))
        }

        if args.fix_z is not None:
            entry["xyz_zfixed"] = [float(xyz_zmean[0]), float(xyz_zmean[1]), float(args.fix_z)]

        results.append(entry)

    # ========== Z clustering ==========
    xyz_mean = np.array([r["xyz_zmean"] for r in results])
    z_base = xyz_mean[:, 2].reshape(-1, 1)
    kmeans = KMeans(n_clusters=args.z_clusters, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(z_base)
    sil = silhouette_score(z_base, labels) if args.z_clusters > 1 else None

    print("\n===== Z clustering (BASE frame) =====")
    if sil is not None:
        print(f"Silhouette score: {sil:.3f}")
    for k in range(args.z_clusters):
        zs_k = z_base[labels == k].flatten()
        print(f"Cluster {k}: count={len(zs_k)}, mean={zs_k.mean():.4f} m, std={zs_k.std():.4f} m, min={zs_k.min():.4f}, max={zs_k.max():.4f}")

    for r, c in zip(results, labels):
        r["z_cluster"] = int(c)

    # ========== Write outputs ==========
    with open(out_jsonl, "w") as f:
        for r in sorted(results, key=lambda x: x["episode_index"]):
            f.write(json.dumps(r) + "\n")

    plot_xyz(np.array([r["xyz_raw"] for r in results]), args.out_dir, "raw")
    plot_xyz(xyz_mean, args.out_dir, "zmean")
    if args.fix_z is not None:
        plot_xyz(np.array([r["xyz_zfixed"] for r in results]), args.out_dir, "zfixed")
    plot_z_clusters(xyz_mean, labels, args.out_dir)

if __name__ == "__main__":
    main()
